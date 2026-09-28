"""PPO training for a Fixed-Point Reasoning Model (FPRM) actor-critic on
grid puzzles (Boxoban, Rush Hour), with adaptive per-state thinking via a policy-KL halting criterion.

The actor-critic is a weight-tied looped Transformer iterated by a
fixed-point solver ("thinking"). At every state, the network stops iterating
once the KL divergence between consecutive policy readouts drops below
`--halt_kl` — it thinks only as long as thinking still changes its mind. The
PPO loss is applied to the final actor/value readout, with full
backpropagation through every executed think iteration.

Run:
    python ppo.py --env_id=sokoban-unfiltered_train-unfiltered_valid --architecture=fprm --seed=1
"""

import os
import time
import pickle
import pprint
import functools

import numpy as np
import tyro

import jax
import jax.numpy as jnp
import flax
import optax
import distrax

from pathlib import Path
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional
from flax.training.train_state import TrainState

import utils.running_statistics as running_statistics
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from envs.utils import make_env
from models.fprm import FPRMConfig
from models.fprm_thinker import FPRMThinkerActorValue
from models.transformer_baseline import TransformerActorValue


@dataclass
class Args:
    # experiment
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    # logging and checkpointing
    track: bool = False                   # log to wandb
    wandb_project_name: str = "looped_actor"
    wandb_entity: Optional[str] = None
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50              # number of evaluation / logging / saving steps
    num_reset_steps: int = 50             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'sokoban-unfiltered_train-unfiltered_valid'  # sokoban-<train>-<eval> | rushhour-<train>-<eval>
    num_envs: int = 2048
    num_eval_envs: int = 128
    sokoban_max_train_levels: int = 0     # sokoban-*: cap on training levels (0 = whole split)
    sokoban_episode_length: int = 120     # sokoban-*: steps per episode
    rushhour_max_train_levels: int = 0    # rushhour-*: cap on training levels (0 = whole split)
    rushhour_episode_length: int = 150    # rushhour-*: steps per episode
    rushhour_shaping_weight: float = 1.0  # rushhour-*: potential-based shaping weight (0 = sparse solve reward only)

    # algorithm
    num_timesteps: int = 50000000
    rollout_length: int = 160
    num_minibatches_per_rollout: int = 32
    num_epochs_per_rollout: int = 8
    learning_rate: float = 1e-4
    discount: float = 0.99
    entropy_cost: float = 0.01
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    # stabilization (all off by default)
    max_grad_norm: float = 0.0            # 0 = no gradient clipping
    target_kl: float = 0.0                # 0 = no KL early stop; else skip remaining minibatch updates in a rollout once approx_kl exceeds this
    anneal_lr: bool = False               # linearly decay lr to 0 over training

    # model
    # 'fprm'         — looped FPRM with policy-KL adaptive halting
    # 'single_block' — FPRM core applied once (same params as fprm, no looping)
    # 'multi_block'  — num_blocks FPRM cores without weight tying (~num_blocks x params)
    architecture: str = 'fprm'
    num_blocks: int = 8                   # blocks of the multi_block baseline
    residual_scaling: bool = True         # --no-residual_scaling: plain pre-norm residuals in the FPRM core (no a1/b1 block scaling, no a2/b2 input mixing) — only sensible for non-looped baselines (no contractivity needed)

    # FPRM thinker
    fprm_d_model: int = 128
    fprm_num_heads: int = 4
    fprm_num_layers: int = 1
    max_think_iters: int = 8              # cap on core calls ("think iterations") per decision
    min_think_iters: int = 2              # halting is checked only after this many iterations
    halt_criterion: str = 'kl'            # 'kl' (policy-KL halting) | 'latent_residual' (original FPRM-paper rule: latent max-token residual)
    halt_kl: float = 1e-3                 # stop thinking once KL(pi_i || pi_{i-1}) between consecutive policy readouts < this
    halt_residual_thresh: float = 0.1     # halt_criterion='latent_residual': stop once the latent residual < this (paper fp_thresh)


@flax.struct.dataclass
class PPOTrainingState(TrainState):
    """Contains training state for the learner."""
    normalizer_params: Any
    env_steps: float


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()


def count_parameters(params):
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))


def save_params(path, params):
    with open(path, 'wb') as fout:
        fout.write(pickle.dumps(params))


def load_params(path):
    with open(path, 'rb') as fin:
        return pickle.loads(fin.read())


def make_inference_fn(actor_critic_network):
    """Creates params and inference function for the PPO agent."""
    def make_policy(params, deterministic: bool = False):

        def policy(observations, goals, key_sample):
            inputs = jnp.concatenate([observations, goals], axis=-1)
            policy_dist, value = actor_critic_network.apply(
                params['policy'],
                inputs,
                params['normalizer'],
            )

            if deterministic:
                return policy_dist.mode(), {'value': value}

            actions = policy_dist.sample(seed=key_sample)
            log_prob = policy_dist.log_prob(actions)
            return actions, {
                'log_prob': log_prob,
                'value': value,
            }

        return policy
    return make_policy


def make_optimizer(args):
    if args.anneal_lr:
        total_grad_steps = (
            args.num_training_step
            * args.num_epochs_per_rollout
            * args.num_minibatches_per_rollout
        )
        learning_rate = optax.linear_schedule(
            args.learning_rate, 0.0, total_grad_steps
        )
    else:
        learning_rate = args.learning_rate
    tx = optax.adam(learning_rate=learning_rate)
    if args.max_grad_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), tx)
    return tx


def main(args: Args):

    args.num_training_step = args.num_timesteps // (args.num_envs * args.rollout_length)
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    args.minibatch_size = args.num_envs * args.rollout_length // (args.num_minibatches_per_rollout)

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.architecture}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, key_env, key_eval, key_policy = jax.random.split(key, 4)

    # Initialize environment
    env_class, env_config = make_env(args)
    env = wrap_env(env_class(config=env_config), env_config.episode_length)
    eval_env = wrap_env(env_class(config=env_config), env_config.episode_length, train=False)
    episode_length = env_config.episode_length

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size
    obs_channels = getattr(env, 'obs_channels', 0)          # > 0 = plane-tokenized grid observation (Boxoban, Rush Hour)
    action_head = getattr(env, 'action_head', 'readout' if obs_channels > 0 else 'per_cell')  # plane-tokenized envs use the readout head by default

    # Initialize the actor-critic
    fprm_config = FPRMConfig(
        d_model=args.fprm_d_model,
        num_heads=args.fprm_num_heads,
        num_layers=args.fprm_num_layers,
        residual_scaling=args.residual_scaling,
    )
    if args.architecture == 'fprm':
        actor_critic_network = FPRMThinkerActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_criterion=args.halt_criterion,
            halt_kl=args.halt_kl,
            halt_residual_thresh=args.halt_residual_thresh,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
        )
    elif args.architecture in ('single_block', 'multi_block'):
        actor_critic_network = TransformerActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            num_blocks=1 if args.architecture == 'single_block' else args.num_blocks,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    training_state = PPOTrainingState.create(
        apply_fn=None,
        params=actor_critic_network.init(key_policy, x=jnp.zeros((1, obs_size + goal_size))),
        tx=make_optimizer(args),
        normalizer_params=running_statistics.init_state((obs_size + goal_size,)),
        env_steps=np.zeros((), dtype=np.float64),
    )
    make_policy = make_inference_fn(actor_critic_network)

    print(f'\nNumber of parameters in actor critic network are: {count_parameters(training_state.params)}\n')

    # Initialize evaluator
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=key_eval,
    )

    def generate_unroll(env, env_state, policy, key, unroll_length, extra_fields):
        """Collect trajectories of given unroll_length."""
        @jax.jit
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            actions, policy_extras = policy(env_state.obs, env_state.info['target_goal'], key)

            next_env_state = env.step(env_state, actions)
            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            transition = Transition(
                observation=jnp.concatenate([env_state.obs, env_state.info['target_goal']], axis=-1),
                action=actions,
                value=policy_extras['value'],
                reward=next_env_state.reward,
                discount=1 - next_env_state.done,
                next_observation=jnp.concatenate([next_env_state.obs, next_env_state.info['target_goal']], axis=-1),
                extras={'policy_extras': policy_extras, 'state_extras': state_extras},
            )

            return (next_env_state, next_key), transition

        (final_env_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
        return final_env_state, data

    @jax.jit
    def data_collect_step(training_state, env_state, key_generate_rollout):
        policy = make_policy({
            'policy': training_state.params,
            'normalizer': training_state.normalizer_params,
        })

        env_state, data = generate_unroll(
            env,
            env_state,
            policy,
            key_generate_rollout,
            args.rollout_length,
            extra_fields=('truncation',),
        )

        # Update normalization params.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )

        return training_state, env_state, data

    def compute_gae(
        truncation: jnp.ndarray,
        termination: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        bootstrap_value: jnp.ndarray,
        lambda_: float = 1.0,
        discount: float = 0.99,
    ):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)

        def compute_vs_minus_v_xs(carry, target_t):
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), (acc)

        (_, _), (vs_minus_v_xs) = jax.lax.scan(
            compute_vs_minus_v_xs,
            (lambda_, acc),
            (truncation_mask, deltas, termination),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        advantages = (
            rewards + discount * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)

    def compute_ppo_loss(params, normalizer_params, data, rng):
        data, value_targets, advantages = data

        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

        def ppo_terms(policy_dist, baseline):
            # Policy function loss
            target_action_log_probs = policy_dist.log_prob(data.action)
            rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
            surrogate_loss1 = rho_s * advantages
            surrogate_loss2 = (jnp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
            policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

            # Value function loss
            v_error = value_targets - baseline
            v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

            # Entropy loss
            entropy = jnp.mean(policy_dist.entropy())
            entropy_loss = args.entropy_cost * -entropy

            # k3 estimator of KL(behaviour || target), Schulman (2020).
            log_rho = target_action_log_probs - behaviour_action_log_probs
            approx_kl = jax.lax.stop_gradient(
                jnp.mean(jnp.exp(log_rho) - 1.0 - log_rho)
            )
            return policy_loss, v_loss, entropy_loss, approx_kl

        # PPO loss on the final actor / value readout. Halted examples freeze
        # inside the network, so their readout is the one they halted at.
        _, inter = actor_critic_network.apply(
            params, data.observation, normalizer_params,
            mutable=['intermediates'],
        )
        logits = inter['intermediates']['logits'][0]
        values = inter['intermediates']['value'][0]
        policy_loss, v_loss, entropy_loss, approx_kl = ppo_terms(
            distrax.Categorical(logits=logits), values
        )

        think_iters = inter['intermediates']['fp_iterations'][0]

        total_loss = policy_loss + v_loss + entropy_loss

        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl,
            'think_iters': jnp.mean(think_iters.astype(jnp.float32)),
        }

    @jax.jit
    def learn_step(training_state, data, key_sgd):

        def _learn_step(carry, unused_t):

            def _train_minibatch_step(carry, data):
                training_state, key, kl_stop = carry
                key, key_loss = jax.random.split(key)

                (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(
                    training_state.params, training_state.normalizer_params, data, key_loss
                )
                grad_norm = optax.global_norm(grads)  # pre-clip
                metrics['grad_norm'] = grad_norm
                if args.max_grad_norm > 0.0:
                    metrics['grad_clip_frac'] = (grad_norm > args.max_grad_norm).astype(jnp.float32)
                new_training_state = training_state.apply_gradients(grads=grads)
                if args.target_kl > 0.0:
                    # Once approx_kl exceeds the target, skip every remaining
                    # minibatch update until the next rollout.
                    new_training_state = jax.tree_util.tree_map(
                        lambda new, old: jnp.where(kl_stop, old, new),
                        new_training_state,
                        training_state,
                    )
                    metrics['kl_stop_frac'] = kl_stop.astype(jnp.float32)
                    kl_stop = kl_stop | (metrics['approx_kl'] > args.target_kl)
                training_state = new_training_state

                return (training_state, key, kl_stop), metrics

            training_state, data, value_targets, advantages, key, kl_stop = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def shuffle_and_reshape(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (args.num_minibatches_per_rollout, -1) + x.shape[2:])
                return x

            batch_data = (data, value_targets, advantages)
            shuffled_batch_data = jax.tree_util.tree_map(shuffle_and_reshape, batch_data)

            (training_state, _, kl_stop), metrics = jax.lax.scan(
                _train_minibatch_step,
                (training_state, key_grad, kl_stop),
                shuffled_batch_data,
                length=args.num_minibatches_per_rollout,
            )
            return (training_state, data, value_targets, advantages, key, kl_stop), metrics

        # calculate gae
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        _, bootstrap_value = actor_critic_network.apply(
            training_state.params, terminal_obs, training_state.normalizer_params
        )

        rewards = data.reward * args.reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        value_targets, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=data.value,
            bootstrap_value=bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.discount,
        )
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        (training_state, _, _, _, _, _), metrics = jax.lax.scan(
            _learn_step,
            (training_state, data, value_targets, advantages, key_sgd,
             jnp.zeros((), bool)),
            (),
            length=args.num_epochs_per_rollout,
        )

        return training_state, metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None
    for ts in range(1, args.num_training_step + 1):

        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, training_data = data_collect_step(training_state, env_state, key_generate_unroll)
        data_collect_step_time += time.time() - data_collect_start

        learn_step_start = time.time()
        training_state, training_metrics = learn_step(training_state, training_data, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, training_metrics
            )

        if args.num_reset_steps > 0 and ts % args.num_training_steps_per_real_reset == 0:
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs)

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval

            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction': data_collect_step_time / training_step_time,
                'training/learning_time_fraction': learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                **{f'training/{name}': value for name, value in metrics.items()},
            }

            metrics = evaluator.run_evaluation(
                policy_params={'policy': training_state.params, 'normalizer': training_state.normalizer_params},
                training_metrics=metrics,
            )

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                import wandb
                wandb.log(metrics, step=es)
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl",
                    params=(
                        training_state.params,
                        training_state.normalizer_params,
                    ),
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
