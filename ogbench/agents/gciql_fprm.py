"""GCIQL with an FPRM (fixed-point looped Transformer) actor.

Only the actor is replaced relative to the official GCIQL agent: the MLP
value function, twin MLP critics, target critic, GCIQL value/critic losses,
optimizer, and target updates are unchanged. The actor is trained with a
DDPG+BC loss on its final readout and one optimizer update per batch.

With `fprm_halt_on_policy_kl > 0` the actor trains with policy-KL halting
active: the think loop runs for up to `fprm_train_iters` core calls with full
backpropagation through the executed iterations, and an example freezes once
KL(pi_i || pi_{i-1}) between consecutive actor readouts drops below the
threshold. With `fprm_halt_on_policy_kl = 0` the loop runs exactly
`fprm_train_iters` core calls.

Documented deviations from vanilla GCIQL:
    - goals are oracle button-bit vectors (9-dim for 3x3), not observations;
    - the actor is a structured token model with deterministic geometry
      features ("geometry-augmented oraclerep").

With `fprm_tokenizer='cube'` (cube envs) or `fprm_tokenizer='puzzle_scalar'`
(puzzle envs on RAW observations, ignoring the legacy oraclerep path) the
actor instead tokenizes observations TQL-style (one token per obs scalar,
utils.cube_tokenizer) and goals are full observations tokenized identically
to states — standard OGBench goal-conditioning, no oracle representation.
Requires fprm_use_grid_conv=False; grid_shape is None throughout.

The latent initialization vector is stored in the parameter tree as
'latent_init' but used under stop_gradient, so its gradient is exactly zero
and Adam never changes it: a checkpointed non-trainable constant.
"""

import copy
import dataclasses
from functools import partial
from typing import Any, Tuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.fprm import (
    STOP_CONVERGED,
    FPInfo,
    FPRMConfig,
    FPRMCore,
    init_solver_state,
    make_latent_init,
    rms_norm,
    run_adaptive,
    run_fixed,
    run_halted_scan,
)
from utils.cube_tokenizer import PREFIX_LEN as CUBE_PREFIX_LEN
from utils.cube_tokenizer import CubeTokenizer
from utils.networks import GCValue, default_init
from utils.puzzle_tokenizer import PREFIX_LEN as PUZZLE_PREFIX_LEN
from utils.puzzle_tokenizer import PuzzleTokenizer

GRID_SHAPES_BY_GOAL_DIM = {9: (3, 3)}


class GCFPRMActor(nn.Module):
    """Goal-conditioned Gaussian FPRM actor.

    Reads the first (READOUT) token of the final latent, RMS-normalizes it,
    and maps it to a 5-dim diagonal Gaussian with constant unit std
    (log_stds = 0), matching the OGBench const_std actor convention.
    """

    config: FPRMConfig
    action_dim: int = 5
    # Token construction: 'puzzle' (per-cell geometry tokens over oraclerep
    # goals, 2-D lattice, size-agnostic), or TQL-style per-scalar tokens
    # with goals tokenized like states, no lattice — requires
    # use_grid_conv=False and grid_shape=None: 'cube' (cube envs) /
    # 'puzzle_scalar' (puzzle envs, raw observations). The halting
    # machinery below is tokenizer-agnostic.
    tokenizer: str = 'puzzle'
    # Readout-stability (policy-KL) halting, shared with the PPO studies
    # line: when > 0, the think loop halts once KL(pi_i || pi_{i-1}) between
    # consecutive actor readouts drops below this tolerance. For this
    # const-std (sigma = 1) Gaussian actor, KL = 0.5 * ||mu_i - mu_{i-1}||^2.
    halt_on_policy_kl: float = 0.0
    # Feedforward-stack baselines (untied stack / single core):
    # when > 0, replace the fixed-point loop by `num_blocks`
    # sequential FPRMCore calls with DISTINCT parameters (no weight tying),
    # plain composition — no solver, no halting, fixed compute per decision.
    # num_blocks=1 = "single block" (iso-params vs the looped actor);
    # num_blocks=8 = "multi block" (iso-FLOPs at ~8x params). Requires
    # halt_on_policy_kl == 0.
    num_blocks: int = 0
    # One-step flow-policy input (agents/fql_fprm.py): when > 0 the actor
    # takes `noise` [B, noise_dim] and the per-scalar tokenizer appends one
    # token per noise scalar. 0 = plain state(/goal) policy (unchanged).
    noise_dim: int = 0

    def _decode(self, mean_head, readout_latent, temperature):
        readout_latent = rms_norm(readout_latent, self.config.rms_norm_eps)
        means = mean_head(readout_latent)
        log_stds = jnp.zeros_like(means)
        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

    @nn.compact
    def __call__(
        self,
        observations,
        goals,
        temperature=1.0,
        *,
        grid_shape,
        train,
        adaptive=False,
        klhalt=False,
        num_iters=None,
        max_iters=None,
        fp_thresh=None,
        return_train_info=False,
        return_info=False,
        coord_transform=None,
        logical_coord_mode='normalized_extent',
        noise=None,
        depth_cap=None,
    ):
        """All keyword-only arguments are static compilation dimensions.

        `coord_transform` / `logical_coord_mode` are evaluation-only
        coordinate adapters of the legacy grid tokenizer; the defaults
        reproduce the original tokenizer exactly and training never sets them.
        """
        assert coord_transform is None or not train, (
            'Coordinate adapters are evaluation-only.'
        )
        cfg = self.config
        if self.tokenizer in ('cube', 'cube_paired', 'puzzle_scalar', 'puzzle_paired', 'scene_scalar', 'scene_paired', 'maze_scalar', 'maze_paired'):
            assert coord_transform is None, (
                'Coordinate adapters are legacy-puzzle-tokenizer diagnostics.'
            )
            assert grid_shape is None, (
                'The per-scalar tokenizer has no 2-D lattice; pass '
                'grid_shape=None.'
            )
            tokens, token_mask = CubeTokenizer(
                cfg.d_model,
                observations.shape[-1],
                layout=('cube' if self.tokenizer in ('cube', 'cube_paired') else 'scene' if self.tokenizer.startswith('scene') else 'maze' if self.tokenizer.startswith('maze') else 'puzzle'),
                paired=self.tokenizer in ('cube_paired', 'puzzle_paired', 'scene_paired', 'maze_paired'),
                noise_dim=self.noise_dim,
                name='tokenizer',
            )(observations, goals, noise=noise)
            prefix_len = CUBE_PREFIX_LEN
        else:
            assert self.tokenizer == 'puzzle', self.tokenizer
            assert self.noise_dim == 0 and noise is None, (
                'noise tokens are only supported by the per-scalar tokenizer'
            )
            tokens, token_mask = PuzzleTokenizer(cfg.d_model, name='tokenizer')(
                observations,
                goals,
                grid_shape,
                coord_transform=coord_transform,
                logical_coord_mode=logical_coord_mode,
            )
            prefix_len = PUZZLE_PREFIX_LEN
        core = FPRMCore(cfg, name='core')
        mean_head = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2),
            name='mean_head',
        )

        # Non-trainable latent init: checkpointed, zero-gradient constant.
        latent_init = self.param(
            'latent_init', lambda rng: make_latent_init(cfg)
        )
        latent_init = jax.lax.stop_gradient(latent_init)

        state = init_solver_state(latent_init, tokens, cfg)

        if self.num_blocks > 0:
            # Feedforward stack: `num_blocks` untied FPRMCore calls composed
            # plainly (the untied-stack semantics — the
            # unrolled FPRM iteration with weight tying removed and the
            # solver dropped). Fixed compute; halting flags invalid.
            assert not (adaptive or klhalt), (
                'The feedforward stack has fixed compute; adaptive/klhalt '
                'execution is undefined for num_blocks > 0.'
            )
            assert self.halt_on_policy_kl == 0, (
                'num_blocks > 0 requires halt_on_policy_kl == 0.'
            )
            cores = [
                FPRMCore(cfg, name=f'core_{i}')
                for i in range(self.num_blocks)
            ]
            z = state.z
            if train and return_train_info:
                # Training: compose every block, decode the final latent.
                for block in cores:
                    z_new = block(z, tokens, token_mask, grid_shape, prefix_len)
                    rms = jnp.sqrt(
                        jnp.mean(jnp.square(z_new - z), axis=(1, 2))
                    )
                    z = z_new
                dist = self._decode(mean_head, z[:, 0], temperature)
                return dist, {
                    'residual': jnp.mean(rms),
                    'rms_residual': jnp.mean(rms),
                    'nonfinite': jnp.any(~jnp.isfinite(z)),
                }

            # Eval-time depth truncation: `depth_cap` runs only the first
            # depth_cap untied blocks and decodes their latent with the
            # shared readout head (a "compute cap" for the untied stack;
            # intermediate readouts are never trained).
            # None = the full stack as trained; `num_iters` stays ignored.
            n_run = self.num_blocks if depth_cap is None else int(depth_cap)
            assert 1 <= n_run <= self.num_blocks, (
                f'depth_cap must be in [1, num_blocks={self.num_blocks}]'
            )
            z_prev = z
            for block_idx, block in enumerate(cores[:n_run]):
                z_prev = z
                z = block(z, tokens, token_mask, grid_shape, prefix_len)
            dist = self._decode(mean_head, z[:, 0], temperature)
            if not return_info:
                return dist
            B = z.shape[0]
            rms_last = jnp.sqrt(jnp.mean(jnp.square(z - z_prev), axis=(1, 2)))
            mean_prev = self._decode(mean_head, z_prev[:, 0], 1.0).mode()
            mean_last = self._decode(mean_head, z[:, 0], 1.0).mode()
            info = {
                'iterations': jnp.full((B,), n_run, jnp.int32),
                'residual': 0.5 * jnp.sum(
                    jnp.square(mean_last - mean_prev), axis=-1
                ),
                'rms_residual': rms_last,
                'paper_global_residual': rms_last,
                'converged': jnp.zeros((B,), bool),  # n/a: fixed stack
                'hit_cap': jnp.zeros((B,), bool),
                'stop_reason': jnp.zeros((B,), jnp.int32),
                'nonfinite': ~jnp.all(
                    jnp.isfinite(z.reshape(B, -1)), axis=-1
                ),
                'action_change_last_step_l2': jnp.linalg.norm(
                    mean_last - mean_prev, axis=-1
                ),
            }
            return dist, info

        halt_residual_fn = None
        if self.halt_on_policy_kl > 0:
            # Force mean-head param creation now, then close over the raw
            # kernel/bias arrays: the halt fn runs inside the lifted scan,
            # where calling a sibling module would break the transform.
            _ = mean_head(jnp.zeros((1, cfg.d_model)))
            mh = mean_head.variables['params']
            mh_kernel, mh_bias = mh['kernel'], mh['bias']

            def halt_residual_fn(z_prev, z_new):
                def mu(z):
                    readout = rms_norm(z[:, 0], cfg.rms_norm_eps)
                    return readout @ mh_kernel + mh_bias

                diff = mu(z_new) - mu(z_prev)
                # Const-std (sigma = 1) diagonal Gaussians:
                # KL(new || prev) = 0.5 * ||mu_new - mu_prev||^2.
                return 0.5 * jnp.sum(jnp.square(diff), axis=-1)

        if klhalt:
            # Policy-KL halted execution (evaluation counterpart of the
            # klhalt training objective; per-example freezing at the halt
            # step, cap defaults to the training iteration budget).
            assert not train, 'klhalt execution here is evaluation-only.'
            assert halt_residual_fn is not None, (
                'klhalt requires halt_on_policy_kl > 0.'
            )
            cap = max_iters if max_iters is not None else cfg.train_iters
            thresh = (
                fp_thresh if fp_thresh is not None else self.halt_on_policy_kl
            )
            final_state, fp_info = run_halted_scan(
                core, state, tokens, token_mask, grid_shape, cfg,
                max_iters=cap,
                fp_thresh=thresh,
                train=False,
                prefix_len=prefix_len,
                halt_residual_fn=halt_residual_fn,
            )
            dist = self._decode(mean_head, final_state.z[:, 0], temperature)
            if not return_info:
                return dist
            info = {
                'iterations': fp_info.iterations,
                'residual': fp_info.residual,
                'rms_residual': fp_info.rms_residual,
                'paper_global_residual': fp_info.paper_global_residual,
                'converged': fp_info.converged,
                'hit_cap': fp_info.hit_cap,
                'stop_reason': fp_info.stop_reason,
                'nonfinite': fp_info.nonfinite,
                # The halting residual IS 0.5 * ||dmu||^2 at each example's
                # final executed step, so the mean action change is exact.
                'action_change_last_step_l2': jnp.sqrt(
                    2.0 * jnp.maximum(fp_info.residual, 0.0)
                ),
            }
            return dist, info

        if adaptive:
            assert not train, 'Adaptive halting is evaluation-only.'
            final_state, fp_info, (readout_prev, readout_last) = run_adaptive(
                core, state, tokens, token_mask, grid_shape, cfg,
                max_iters=max_iters, fp_thresh=fp_thresh,
                prefix_len=prefix_len,
            )
            dist = self._decode(mean_head, final_state.z[:, 0], temperature)
            if not return_info:
                return dist
            mean_prev = self._decode(mean_head, readout_prev, 1.0).mode()
            mean_last = self._decode(mean_head, readout_last, 1.0).mode()
            info = {
                'iterations': fp_info.iterations,
                'residual': fp_info.residual,
                'rms_residual': fp_info.rms_residual,
                'paper_global_residual': fp_info.paper_global_residual,
                'converged': fp_info.converged,
                'hit_cap': fp_info.hit_cap,
                'stop_reason': fp_info.stop_reason,
                'nonfinite': fp_info.nonfinite,
                'action_change_last_step_l2': jnp.linalg.norm(
                    mean_last - mean_prev, axis=-1
                ),
            }
            return dist, info

        if train and return_train_info and halt_residual_fn is not None:
            # Policy-KL halted training: one differentiable scan over the
            # full iteration budget with full BPTT through every executed
            # iteration; halted examples freeze at their halt step and the
            # loss is taken on the final readout.
            k_total = num_iters if num_iters is not None else cfg.train_iters
            state, fp_info = run_halted_scan(
                core, state, tokens, token_mask, grid_shape, cfg,
                max_iters=k_total,
                fp_thresh=self.halt_on_policy_kl,
                train=True,
                prefix_len=prefix_len,
                halt_residual_fn=halt_residual_fn,
            )
            dist = self._decode(mean_head, state.z[:, 0], temperature)
            info = {
                # residual holds the policy KL of each example's final
                # executed step (the halting quantity).
                'residual': jnp.mean(state.residual),
                'rms_residual': jnp.mean(fp_info.rms_residual),
                'nonfinite': jnp.any(state.nonfinite),
                'think_iters': jnp.mean(
                    state.iterations.astype(jnp.float32)
                ),
                'kl_converged_frac': jnp.mean(
                    (state.stop_reason == STOP_CONVERGED).astype(jnp.float32)
                ),
            }
            return dist, info

        if train and return_train_info:
            # Fixed-depth training: exactly `train_iters` core calls, full
            # BPTT, loss on the final readout.
            k_total = num_iters if num_iters is not None else cfg.train_iters
            state, diags = run_fixed(
                core, state, tokens, token_mask, grid_shape, k_total, cfg,
                train=True, prefix_len=prefix_len,
            )
            dist = self._decode(mean_head, state.z[:, 0], temperature)
            return dist, {
                'residual': jnp.mean(state.residual),
                'rms_residual': jnp.mean(diags.rms_residual[-1]),
                'nonfinite': jnp.any(state.nonfinite),
            }

        # Plain fixed-depth execution (evaluation).
        k_total = num_iters if num_iters is not None else cfg.train_iters
        final_state, diags = run_fixed(
            core, state, tokens, token_mask, grid_shape, k_total, cfg,
            train=train, prefix_len=prefix_len,
        )
        dist = self._decode(mean_head, final_state.z[:, 0], temperature)
        if not return_info:
            return dist

        # Diagnostics from the per-step scan outputs.
        if k_total >= 2:
            mean_prev = self._decode(mean_head, diags.readout[-2], 1.0).mode()
        else:
            mean_prev = self._decode(mean_head, state.z[:, 0], 1.0).mode()
        mean_last = self._decode(mean_head, diags.readout[-1], 1.0).mode()
        info = {
            'iterations': final_state.iterations,
            'residual': final_state.residual,
            'rms_residual': diags.rms_residual[-1],
            'paper_global_residual': diags.paper_global_residual[-1],
            'converged': jnp.zeros_like(final_state.nonfinite),  # n/a in fixed mode
            'hit_cap': jnp.zeros_like(final_state.nonfinite),
            'stop_reason': jnp.zeros_like(final_state.iterations),
            'nonfinite': final_state.nonfinite,
            'action_change_last_step_l2': jnp.linalg.norm(
                mean_last - mean_prev, axis=-1
            ),
        }
        return dist, info


class GCIQLFPRMAgent(flax.struct.PyTreeNode):
    """GCIQL agent with an FPRM actor (DDPG+BC actor loss only)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """IQL value loss (identical to official GCIQL)."""
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['value_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """IQL critic loss (identical to official GCIQL)."""
        next_v = self.network.select('value')(
            batch['next_observations'], batch['value_goals']
        )
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def _ddpg_bc_loss(self, dist, batch):
        """DDPG+BC loss on an action distribution.

        The critic is evaluated with the stored parameter tree (never
        grad_params), so the actor Q term contributes no critic gradient while
        gradients still flow through q_actions into the actor.
        """
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['actor_goals'], q_actions
        )
        q = jnp.minimum(q1, q2)

        q_norm = jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])
        q_loss = -q.mean() / q_norm
        bc_loss = -(self.config['alpha'] * log_prob).mean()

        return q_loss + bc_loss, {
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """DDPG+BC actor loss on the final readout (one optimizer update)."""
        dist, train_info = self.network.select('actor')(
            batch['observations'],
            batch['actor_goals'],
            params=grad_params,
            grid_shape=self.config['grid_shape'],
            train=True,
            return_train_info=True,
        )
        actor_loss, info = self._ddpg_bc_loss(dist, batch)
        info['mean_residual'] = train_info['residual']
        info['mean_rms_residual'] = train_info['rms_residual']
        if self.config['fprm_halt_on_policy_kl'] > 0:
            # Halting statistics of the training batch.
            info['think_iters'] = train_info['think_iters']
            info['kl_converged_frac'] = train_info['kl_converged_frac']
        info['actor_loss'] = actor_loss
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Official-API compatibility wrapper (used by main.py evaluation).

        Runs the actor at the configured default grid and, for a klhalt agent
        (fprm_halt_on_policy_kl > 0), with policy-KL halting at the training
        threshold and cap — so in-training evaluation measures the adaptive
        policy actually being trained. Otherwise fixed training depth.
        Accepts the official unbatched (D,) observation/goal shape and
        returns (action_dim,); batched inputs pass through unchanged.
        """
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
            goals = goals[None]
        observations = observations.astype(jnp.float32)
        goals = goals.astype(jnp.float32)

        klhalt = self.config['fprm_halt_on_policy_kl'] > 0
        dist = self.network.select('actor')(
            observations,
            goals,
            temperature=temperature,
            grid_shape=self.config['grid_shape'],
            train=False,
            adaptive=False,
            klhalt=klhalt,
            num_iters=None if klhalt else self.config['fprm_train_iters'],
        )
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        if unbatched:
            actions = actions[0]
        return actions

    @partial(
        jax.jit,
        static_argnames=(
            'grid_shape',
            'deterministic',
            'halt_mode',
            'num_iters',
            'max_iters',
            'fp_thresh',
            'coord_transform',
            'logical_coord_mode',
            'depth_cap',
        ),
    )
    def sample_actions_with_info(
        self,
        observations,
        goals,
        *,
        grid_shape,
        deterministic=True,
        halt_mode='fixed',
        num_iters=None,
        max_iters=None,
        fp_thresh=None,
        seed=None,
        temperature=1.0,
        coord_transform=None,
        logical_coord_mode='normalized_extent',
        depth_cap=None,
    ):
        """Compute-scaling evaluation API: actions plus fixed-point diagnostics.

        halt_mode 'fixed' runs exactly `num_iters` core calls; 'fixed_point'
        runs adaptive halting with cap `max_iters` and threshold `fp_thresh`
        (defaults from the FPRM config); 'policy_kl' runs readout-stability
        halting (requires fprm_halt_on_policy_kl > 0; `fp_thresh` then
        overrides the KL threshold and `max_iters` defaults to the training
        cap). Inputs must be batched [B, D].

        `coord_transform` (a hashable utils.coord_adapter.CoordinateTransform,
        static: one compilation per distinct transform) and
        `logical_coord_mode` apply the evaluation-time coordinate adapters;
        the returned action is in POLICY (virtual) space - any inverse
        planar action mapping happens outside, immediately before env.step.
        """
        assert halt_mode in ('fixed', 'fixed_point', 'policy_kl'), halt_mode
        assert observations.ndim == 2, 'sample_actions_with_info expects [B, D]'
        observations = observations.astype(jnp.float32)
        goals = goals.astype(jnp.float32)

        adaptive = halt_mode == 'fixed_point'
        klhalt = halt_mode == 'policy_kl'
        if halt_mode == 'fixed':
            assert num_iters is not None, 'fixed mode requires num_iters'

        dist, info = self.network.select('actor')(
            observations,
            goals,
            temperature=temperature,
            grid_shape=grid_shape,
            train=False,
            adaptive=adaptive,
            klhalt=klhalt,
            num_iters=num_iters if halt_mode == 'fixed' else None,
            max_iters=max_iters,
            fp_thresh=fp_thresh,
            return_info=True,
            coord_transform=coord_transform,
            logical_coord_mode=logical_coord_mode,
            depth_cap=depth_cap,
        )
        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        info['action_mean_preclip'] = dist.mode()
        actions = jnp.clip(actions, -1, 1)
        return actions, info

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create the agent from a sampled example batch.

        The example batch must contain 'observations', 'actions', and
        'actor_goals' (oracle goals, e.g. 9-dim for 3x3): goal-conditioned
        networks are initialized from the oracle goal width, never from
        `ex_goals = ex_observations`.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals']

        goal_dim = ex_goals.shape[-1]
        if config['fprm_tokenizer'] in ('cube', 'cube_paired', 'puzzle_scalar', 'puzzle_paired', 'scene_scalar', 'scene_paired', 'maze_scalar', 'maze_paired'):
            # Goal-conditioned per-scalar path: goals are full observations,
            # tokenized identically to states. No lattice, so no grid
            # conv/grid shape.
            assert goal_dim == ex_observations.shape[-1], (
                f'per-scalar-tokenizer goals must be full observations: goal '
                f'dim {goal_dim} != obs dim {ex_observations.shape[-1]}'
            )
            assert not config['fprm_use_grid_conv'], (
                'The per-scalar tokenizer has no 2-D lattice; set '
                '--agent.fprm_use_grid_conv=False explicitly.'
            )
            grid_shape = None
        else:
            grid_shape = GRID_SHAPES_BY_GOAL_DIM[goal_dim]
        with config.unlocked():
            config['grid_shape'] = grid_shape
        action_dim = ex_actions.shape[-1]

        fprm_config = FPRMConfig(
            d_model=config['fprm_d_model'],
            num_heads=config['fprm_num_heads'],
            num_layers=config['fprm_num_layers'],
            expansion=config['fprm_expansion'],
            softmax_temp=config['fprm_softmax_temp'],
            rms_norm_eps=config['fprm_rms_norm_eps'],
            alpha_1_init=config['fprm_alpha_1_init'],
            alpha_2_init=config['fprm_alpha_2_init'],
            use_grid_conv=config['fprm_use_grid_conv'],
            conv_kernel=tuple(config['fprm_conv_kernel']),
            conv_bias=config['fprm_conv_bias'],
            train_iters=config['fprm_train_iters'],
            eval_max_iters=config['fprm_eval_max_iters'],
            fp_thresh=config['fprm_fp_thresh'],
            eps=config['fprm_eps'],
            adaptive_min_iters=config['fprm_adaptive_min_iters'],
            init_mode=config['fprm_init_mode'],
            init_std=config['fprm_init_std'],
            init_seed=config['fprm_init_seed'],
            additive_noise_std=config['fprm_additive_noise_std'],
        )

        # Feedforward-stack baselines (.get: key absent in configs of runs
        # predating the option, which are all looped).
        num_blocks = config.get('fprm_num_blocks', 0)
        if num_blocks > 0:
            assert config['fprm_halt_on_policy_kl'] == 0, (
                'fprm_num_blocks > 0 (feedforward stack) is incompatible '
                'with klhalt.'
            )

        # MLP value and twin critics, exactly as in official GCIQL.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )
        actor_def = GCFPRMActor(
            config=fprm_config,
            action_dim=action_dim,
            tokenizer=config['fprm_tokenizer'],
            halt_on_policy_kl=config['fprm_halt_on_policy_kl'],
            num_blocks=num_blocks,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            actor=(
                actor_def,
                dict(
                    observations=ex_observations,
                    goals=ex_goals,
                    grid_shape=grid_shape,
                    train=True,
                    return_train_info=True,
                ),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['max_grad_norm'] > 0:
            network_tx = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters (puzzle GCIQL defaults).
            agent_name='gciql_fprm',
            create_from_example_batch=True,  # main.py passes the example batch.
            lr=3e-4,
            max_grad_norm=0.0,  # > 0: global-norm gradient clipping.
            batch_size=1024,
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,  # Target network update rate.
            expectile=0.9,
            actor_loss='ddpgbc',  # Only DDPG+BC is supported.
            # NOTE: `alpha` is the GCIQL DDPG+BC BC coefficient (iql_alpha in
            # the spec), NOT an FPRM residual scale (fprm_alpha_*_init below).
            alpha=1.0,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            # FPRM actor hyperparameters (spec Section 8.1).
            # Tokenizer: 'puzzle' (per-cell geometry tokens, oraclerep), or
            # TQL-style per-scalar tokens (require fprm_use_grid_conv=False
            # and full-observation goals): 'cube' (e.g.
            # cube-double-play-v0) / 'cube_paired' / 'puzzle_scalar' / 'puzzle_paired'
            # (one token per scalar holding [s_i, g_i]; 1 + D tokens) (e.g.
            # puzzle-3x3-play-v0 raw observations).
            fprm_tokenizer='puzzle',
            fprm_d_model=128,
            fprm_num_heads=4,
            fprm_num_layers=2,
            fprm_expansion=4.0,
            fprm_softmax_temp=1.0,
            fprm_rms_norm_eps=1e-5,
            fprm_alpha_1_init=0.75,
            fprm_alpha_2_init=0.25,
            fprm_use_grid_conv=True,
            fprm_conv_kernel=(3, 3),
            fprm_conv_bias=False,
            fprm_train_iters=8,
            fprm_eval_max_iters=64,
            fprm_fp_thresh=0.1,
            fprm_eps=1e-8,
            fprm_adaptive_min_iters=2,
            fprm_init_mode='fixed_random',
            fprm_init_std=6.0,
            fprm_init_seed=0,
            fprm_additive_noise_std=0.0,
            # Policy-KL halting threshold (0 = exactly fprm_train_iters core
            # calls, no halting).
            fprm_halt_on_policy_kl=0.0,
            # Feedforward-stack baselines (untied stack
            # port): 0 = looped FPRM (default); 1 = single untied block
            # (iso-params); 8 = eight untied blocks (iso-FLOPs, ~8x params).
            fprm_num_blocks=0,
            # Dataset hyperparameters (identical to official puzzle GCIQL).
            dataset_class='GCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            oracle_success_mode='index',  # 'index' (primary) | 'semantic' (ablation)
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
