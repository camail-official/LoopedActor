"""Evaluate a Rush Hour PPO checkpoint on a held-out bank, each level exactly once.

Deterministic policy (argmax), 150-step episodes, levels rolled out in parallel with the JAX env. Reports success rate,
mean realized think iterations (looped arm), KL-convergence rate and mean episode length, plus success broken down by
the level's minimum move count (Fogleman moves = slides of any length).

Examples (repo root)
  python rushhour/eval_checkpoint.py --ckpt exp/rushhour/looped_s1/checkpoints/<run>/params_101.pkl
  python rushhour/eval_checkpoint.py --ckpt exp/rushhour/iso_flops_s1/checkpoints/<run>/params_101.pkl --arch multi_block --num_blocks 16
  python rushhour/eval_checkpoint.py --ckpt exp/rushhour/looped_s1/checkpoints/<run>/params_101.pkl --max_iters 8   # test-time cap
"""
import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.rushhour_env import RushHourEnv, default_config  # noqa: E402
from models.fprm import FPRMConfig  # noqa: E402
from models.fprm_thinker import FPRMThinkerActorValue  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='params_<k>.pkl written by ppo.py: (policy params, normalizer params)')
    ap.add_argument('--arch', default='fprm', choices=['fprm', 'multi_block', 'single_block'])
    ap.add_argument('--max_iters', type=int, default=16, help='looped: think-iteration cap at evaluation (training cap 16)')
    ap.add_argument('--halt_kl', type=float, default=1e-3)
    ap.add_argument('--num_blocks', type=int, default=16, help='multi_block: number of untied cores')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--splits', nargs='+', default=['easy_test'])
    ap.add_argument('--max_levels', type=int, default=2000, help='levels per split, taken from the start of the bank (0 = whole bank)')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--episode_steps', type=int, default=150)
    ap.add_argument('--out', default=None, help='JSON path (default: next to the checkpoint, eval_<name>[_cap<N>].json)')
    args = ap.parse_args()

    cfg = FPRMConfig(d_model=args.d_model, num_heads=4, num_layers=args.num_layers)
    kw = dict(m=6, n=6, config=cfg, output_dim_1=72, obs_channels=7, action_head='per_cell')
    if args.arch == 'fprm':
        mod = FPRMThinkerActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_criterion='kl',
                                    halt_kl=args.halt_kl, **kw)
    else:
        mod = TransformerActorValue(num_blocks=args.num_blocks if args.arch == 'multi_block' else 1,
                                    **kw)
    with open(args.ckpt, 'rb') as f:
        policy_vars, norm = pickle.load(f)

    @jax.jit
    def act(obs):
        (dist, _), inter = mod.apply(policy_vars, obs, norm, mutable=['intermediates'])
        I = inter['intermediates']
        its = I['fp_iterations'][0] if 'fp_iterations' in I else jnp.zeros(obs.shape[0], jnp.int32)
        conv = I['fp_converged'][0] if 'fp_converged' in I else jnp.zeros(obs.shape[0], bool)
        return dist.mode(), its, conv

    results = {}
    for split in args.splits:
        ecfg = default_config(); ecfg.train_split = split; ecfg.eval_split = split; ecfg.episode_length = args.episode_steps
        env = RushHourEnv(ecfg)
        n_lv = env.num_eval_levels if args.max_levels <= 0 else min(args.max_levels, env.num_eval_levels)
        reset_v = jax.jit(jax.vmap(lambda i: env.reset_level(env.eval_bank, i)))
        step_v = jax.jit(jax.vmap(env.step))
        T = args.episode_steps
        succ = np.zeros(n_lv, bool); length = np.full(n_lv, T, np.int32); think = np.zeros(n_lv); cnt = np.zeros(n_lv)
        conv_sum = 0.0; think_n = 0
        for s0 in range(0, n_lv, args.batch):
            idx = jnp.arange(s0, min(s0 + args.batch, n_lv))
            st = reset_v(idx); done = np.zeros(len(idx), bool)
            for t in range(T):
                a, its, cv = act(st.obs)
                alive = ~done; rows = s0 + np.flatnonzero(alive)
                think[rows] += np.asarray(its)[alive]; cnt[rows] += 1
                conv_sum += float(np.asarray(cv)[alive].sum()); think_n += int(alive.sum())
                st = step_v(st, a)
                d_now = np.asarray(st.done) > 0.5; newly = d_now & ~done
                succ[s0 + np.flatnonzero(newly)] = True; length[s0 + np.flatnonzero(newly)] = t + 1; done |= d_now
                if done.all():
                    break
        moves = np.asarray(env.eval_bank['moves'][:n_lv])
        by_moves = {int(m): float(succ[moves == m].mean()) for m in np.unique(moves)}
        results[split] = dict(levels=int(n_lv), success=float(succ.mean()), mean_len=float(length.mean()),
                              think=float(think.sum() / max(cnt.sum(), 1)), converged=float(conv_sum / max(think_n, 1)),
                              solved_len=float(length[succ].mean()) if succ.any() else None, success_by_moves=by_moves)
        print(f'{args.ckpt} {split:10s} n={n_lv:5d} success {succ.mean():.3f} think {results[split]["think"]:.2f} '
              f'conv {results[split]["converged"]:.3f} len {length.mean():.1f}', flush=True)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f'eval_{os.path.splitext(os.path.basename(args.ckpt))[0]}'
                                   + (f'_cap{args.max_iters}' if args.arch == 'fprm' and args.max_iters != 16 else '') + '.json')
    json.dump(dict(ckpt=args.ckpt, arch=args.arch, max_iters=args.max_iters if args.arch == 'fprm' else None, results=results),
              open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
