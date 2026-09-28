"""Evaluate a saved checkpoint on its OGBench environment, optionally under a
test-time cap on think iterations (looped actor) or a depth cap (untied stack).

Success = OGBench's per-task success flag at episode end, averaged over
`--episodes` episodes per task and over the tasks. Also reports the mean
number of think iterations (core applications) actually executed per step.

Examples
  python eval_checkpoint.py --ckpt_dir exp/cube_double/dummy/looped/sd000_<stamp>
  python eval_checkpoint.py --ckpt_dir exp/cube_double/dummy/looped/sd000_<stamp> --max_iters 8
  python eval_checkpoint.py --ckpt_dir exp/cube_double/dummy/iso_flops/sd000_<stamp> --max_iters 8   # first 8 of 16 blocks
"""
import argparse
import json
import os
import pickle
import time

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

import ogbench
from agents import agents

EPISODE_STEPS = {  # OGBench episode budgets (max_episode_steps)
    'cube-double-play-v0': 500, 'cube-triple-play-v0': 1000,
    'puzzle-3x3-play-v0': 500, 'scene-play-v0': 750,
}


def derive_seed(base_seed, env_name, task_key, ep_idx):
    """Stable per-(env, task, episode) reset seed (process-salt free)."""
    import hashlib
    digest = hashlib.sha256(f'{base_seed}|{env_name}|{task_key}|{ep_idx}'.encode()).digest()
    return int.from_bytes(digest[:4], 'little') % (2**31 - 1)


def build_agent(ckpt_dir, ckpt, obs_dim, action_dim):
    with open(os.path.join(ckpt_dir, 'flags.json')) as f:
        flag_dict = json.load(f)
    config = ml_collections.ConfigDict(flag_dict['agent'])
    example_batch = {
        'observations': np.zeros((1, obs_dim), np.float32),
        'actions': np.zeros((1, action_dim), np.float32),
        'actor_goals': np.zeros((1, obs_dim), np.float32),
    }
    agent = agents[config['agent_name']].create(0, example_batch, config)
    with open(os.path.join(ckpt_dir, f'params_{ckpt}.pkl'), 'rb') as f:
        load_dict = pickle.load(f)
    state = flax.serialization.to_state_dict(agent)
    loaded = load_dict['agent']
    if 'network' in loaded and set(loaded['network'].get('params', {})) < set(state['network']['params']):
        # actor-only pickle: keep the freshly initialized critics (unused at evaluation) and load the actor
        for k, v in loaded['network']['params'].items():
            state['network']['params'][k] = v
        loaded = state
    return flax.serialization.from_state_dict(agent, loaded), config, flag_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True, help='directory holding flags.json and params_<ckpt>.pkl')
    ap.add_argument('--ckpt', type=int, default=1000000)
    ap.add_argument('--env_name', default=None, help='default: the env the checkpoint was trained on')
    ap.add_argument('--tasks', default='1,2,3,4,5')
    ap.add_argument('--episodes', type=int, default=50, help='episodes per task (run in lockstep)')
    ap.add_argument('--max_iters', type=int, default=0,
                    help='looped actor: hard cap on think iterations (0 = training cap); '
                         'untied stack: run only the first N blocks (0 = all)')
    ap.add_argument('--seed_base', type=int, default=50000, help='episode-init seed base')
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()

    with open(os.path.join(args.ckpt_dir, 'flags.json')) as f:
        env_name = args.env_name or json.load(f)['env_name']
    max_steps = EPISODE_STEPS[env_name]
    envs = [ogbench.make_env_and_datasets(env_name, env_only=True, max_episode_steps=max_steps)
            for _ in range(args.episodes)]
    obs_dim = envs[0].observation_space.shape[-1]
    action_dim = envs[0].action_space.shape[-1]
    agent, config, flag_dict = build_agent(args.ckpt_dir, args.ckpt, obs_dim, action_dim)

    feedforward = config.get('fprm_num_blocks', 0) > 0
    if feedforward:
        nb = int(config['fprm_num_blocks'])
        assert 0 <= args.max_iters <= nb, f'depth cap must be in [1, {nb}]'
        cap = args.max_iters or nb
        halt_kwargs = dict(halt_mode='fixed', num_iters=cap, depth_cap=cap)
    else:
        assert config['fprm_halt_on_policy_kl'] > 0, 'expected a KL-halting looped actor'
        cap = args.max_iters or int(config['fprm_train_iters'])
        halt_kwargs = dict(halt_mode='policy_kl', max_iters=cap)
    print(f'{args.ckpt_dir} params_{args.ckpt} env={env_name} feedforward={feedforward} '
          f'cap={cap} episodes={args.episodes}', flush=True)

    results = {'ckpt_dir': args.ckpt_dir, 'ckpt': args.ckpt, 'env_name': env_name,
               'run_group': flag_dict['run_group'], 'seed': flag_dict['seed'],
               'episodes': args.episodes, 'cap': cap, 'feedforward': feedforward, 'tasks': {}}
    t0 = time.time()
    for task_id in [int(t) for t in args.tasks.split(',')]:
        n = len(envs)
        obs, goals = [], []
        for i, env in enumerate(envs):
            np.random.seed(derive_seed(args.seed_base, env_name, f'task{task_id}', i))
            o, info = env.reset(options=dict(task_id=task_id))
            obs.append(o); goals.append(info['goal'])
        obs, goals = np.stack(obs), np.stack(goals)
        done = np.zeros(n, bool); success = np.zeros(n, bool)
        iters_sum = np.zeros(n); steps = np.full(n, max_steps, np.int64)
        for t in range(max_steps):
            actions, info = agent.sample_actions_with_info(
                obs.astype(np.float32), goals.astype(np.float32),
                grid_shape=None, deterministic=True, **halt_kwargs)
            acts = np.clip(np.asarray(actions), -1, 1)
            its = np.asarray(info['iterations'])
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                o, r, term, trunc, inf = env.step(acts[i])
                obs[i] = o
                iters_sum[i] += its[i]
                if term or trunc:
                    # OGBench protocol: success is the goal predicate at episode end.
                    done[i] = True; steps[i] = t + 1
                    success[i] = bool(inf.get('success', False))
            if done.all():
                break
        rec = {'success': float(success.mean()),
               'mean_steps': float(steps.mean()),
               'mean_iters': float((iters_sum / steps).mean())}
        results['tasks'][f'task{task_id}'] = rec
        print(f'task {task_id}: success={rec["success"]:.3f} iters={rec["mean_iters"]:.2f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    results['overall_success'] = float(np.mean([v['success'] for v in results['tasks'].values()]))
    results['overall_mean_iters'] = float(np.mean([v['mean_iters'] for v in results['tasks'].values()]))
    print(f'overall success {results["overall_success"]:.3f}  mean iters {results["overall_mean_iters"]:.2f}')
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
