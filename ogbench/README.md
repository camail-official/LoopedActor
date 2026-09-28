# Looped actors for OGBench manipulation

Offline goal-conditioned RL on four OGBench manipulation environments with a
**looped Transformer actor** (a weight-tied FPRM core iterated
with policy-KL halting) inside GCIQL, against two untied controls at matched
parameters and matched compute. Value and critic stay the standard GCIQL MLPs;
only the actor changes.

| arm | actor | parameters | compute / decision |
|---|---|---|---|
| `looped` (ours) | one weight-tied 2-layer core, up to 16 think iterations, stops early once consecutive policy readouts agree (KL < 1e-3) | 1x | adaptive, <= 16 core calls |
| `iso_flops` | 16 untied 2-layer cores stacked, loss on the final readout | 16x | 16 core calls |
| `iso_param` | the 2-layer core applied once | 1x | 1 core call |

Observations and goals are tokenized per scalar (TQL-style); on the puzzle each
button's observation and goal scalars share one token (`puzzle_paired`) so the
actor can compare them. The loss is on the final readout of the looped actor
and of the untied stack alike.

## Layout

```
ogbench/
  main.py                  training loop (OGBench reference trainer + FPRM agent)
  agents/gciql_fprm.py     GCIQL with the looped / untied FPRM actor (+ get_config)
  utils/fprm.py            FPRM core, fixed-point solver, halted scan
  utils/cube_tokenizer.py  per-scalar tokenizer (cube, scene, paired puzzles)
  utils/puzzle_tokenizer.py grid tokenizer (legacy oraclerep puzzle path, unused here)
  eval_checkpoint.py       evaluate a checkpoint, optionally under a think-iteration cap
  scripts/train.sh         one command per (env, arm, seed)
  tests/                   unit tests (tokenizers, halting, untied stacks)
```

## Setup

```bash
pip install -r ogbench/requirements.txt        # plus the CUDA build of jax
export MUJOCO_GL=egl
cd ogbench && python -m pytest -q tests/        # ~3 min on CPU
```

OGBench downloads datasets to `~/.ogbench/data` on first use
(`cube-double-play-v0`, `cube-triple-play-v0`, `puzzle-3x3-play-v0`,
`scene-play-v0`, each with its `-val` split).

## Training

```bash
scripts/train.sh cube_double looped 0        # -> exp/cube_double/dummy/looped/sd000_<stamp>/
scripts/train.sh cube_double iso_flops 0
scripts/train.sh cube_double iso_param 0
```

Environments: `cube_double`, `cube_triple`, `puzzle_3x3`, `scene`;
seeds 0-4 in the paper. Every run evaluates 50 episodes per task every 100k
gradient steps (`eval.csv`) and saves `params_<step>.pkl` alongside `flags.json`.
Wall-clock on one B200: `iso_param` ~1-3 h, `iso_flops` ~5-10 h, `looped` ~12-24 h
per seed (the puzzle and scene have the most tokens).

## Evaluation

```bash
python eval_checkpoint.py --ckpt_dir exp/scene/dummy/looped/sd000_<stamp>
python eval_checkpoint.py --ckpt_dir exp/scene/dummy/looped/sd000_<stamp> --max_iters 4    # cap think iterations
python eval_checkpoint.py --ckpt_dir exp/scene/dummy/iso_flops/sd000_<stamp> --max_iters 8 # first 8 of 16 blocks
```

`eval_checkpoint.py` rebuilds the agent from the run's `flags.json`, loads
`params_1000000.pkl`, runs 50 episodes per task from fixed episode seeds and
reports the success rate and the mean number of core calls actually executed
per step. Success rate (%) of the paper's runs under this evaluator (50
episodes x 5 tasks), mean +- std over seeds 0-4:

| env | iso_param | iso_flops | looped (cap 16) |
|---|---|---|---|
| cube_double | 14.9 +- 7.6 | 64.9 +- 3.6 | 66.7 +- 2.6 |
| cube_triple | 2.7 +- 5.2 | 47.0 +- 9.1 | 49.0 +- 7.8 |
| puzzle_3x3 | 90.6 +- 7.3 | 91.6 +- 12.5 | 90.6 +- 7.7 |
| scene | 25.5 +- 7.5 | 69.4 +- 4.5 | 68.6 +- 6.4 |

The in-training evaluation logged to `eval.csv` uses different episode
initializations and differs from these by evaluation noise (a few points per
seed). GPU evaluation is not bit-reproducible (XLA non-determinism), so repeated
evaluations of the same checkpoint on the same hardware can differ by a few
points on `cube_double` and `cube_triple` and by about +-1 point on
`puzzle_3x3` and `scene`.
