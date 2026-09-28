# Looped Actor: Depth-Recurrent Reasoning Models for Reinforcement Learning

Code and data pipelines for the paper's three studies. The
actor of an RL agent is a **depth-recurrent (looped) Transformer**: one
weight-tied core is applied repeatedly to a latent, with repeated input
injection and a fixed-point update between applications (the FPRM core,
[arXiv:2606.18206](https://arxiv.org/abs/2606.18206)). A **policy-KL halting
rule** lets every state choose its own depth: iteration stops once consecutive
policy readouts agree, `KL(pi_i || pi_{i-1}) < halt_kl`. Halted examples are
frozen inside a differentiable fixed-length scan, so the same rule is active
during rollouts and inside the loss, and gradients flow through exactly the
iterations that were executed. Every study compares the same three arms:

| arm | actor | parameters | compute per decision |
|---|---|---|---|
| **looped** (ours) | one core, up to 16 applications, KL halting | 1x | adaptive, <= 16 |
| **iso_flops** | 16 untied cores stacked, no halting | 16x | fixed, 16 |
| **iso_param** | the core applied once | 1x | fixed, 1 |

- **OGBench** (`ogbench/`): offline goal-conditioned RL (GCIQL) on four
  manipulation environments (`cube_double`, `cube_triple`, `puzzle_3x3`,
  `scene`); only the actor changes, value and critic stay the standard MLPs.
  Seeds 0-4.
- **Boxoban** (`sokoban/`): online PPO on Boxoban levels, trained and
  evaluated on the unfiltered generator (`unfiltered_train` / `_valid` / `_test`). Seeds 1-3.
- **Rush Hour** (`rushhour/`): online PPO on Fogleman's 6x6 Rush Hour puzzles
  with at most 15 moves (`easy_train` / `easy_valid` / `easy_test`). Seeds 1-3.

The PPO studies share `ppo.py`, `models/`, `envs/` and `utils/`; the OGBench
study is self-contained under `ogbench/`. Each study's README gives the exact
training and evaluation commands and the numbers reported in the paper.

## Layout

```
ppo.py                      PPO trainer (Boxoban, Rush Hour), architecture switch --architecture=fprm|multi_block|single_block
models/fprm.py              FPRM core, fixed-point solver, halted differentiable scan
models/fprm_thinker.py      looped actor-critic (tokenizer, KL halting, readout / per-cell heads)
models/transformer_baseline.py   untied stack (iso_flops) and single core (iso_param)
envs/sokoban_env.py, envs/rushhour_env.py   JAX environments; envs/utils.py env registry; utils/ PPO helpers
data_scripts/               build the Boxoban and Rush Hour level banks from their public sources
sokoban/, rushhour/         per-study train script, evaluator and README
ogbench/                    OGBench study (trainer, agent, tokenizers, evaluator, tests, README)
tests/                      environment tests (Boxoban)
```

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # PPO studies
pip install -r ogbench/requirements.txt    # OGBench study (adds ogbench, mujoco)
pip install -U "jax[cuda12]==0.10.2"       # GPU build of JAX (training; evaluation also runs on CPU, slowly)
python -m pytest -q tests/                 # Boxoban environment tests
(cd ogbench && python -m pytest -q tests/)  # OGBench tokenizer / halting / untied-stack tests, ~3 min on CPU
```

All reported runs used Python 3.11, JAX 0.10.2, Flax 0.12.8 on a single
NVIDIA H100 or B200 per run (`MUJOCO_GL=egl` for OGBench).

## Reproducing the tables

1. Build the data (`data_scripts/`, see the study READMEs; OGBench datasets download automatically).
2. Train with `<study>/scripts/train.sh <arm> <seed>`; every run is seeded and
   writes its checkpoints and evaluation curves. Training is deterministic up
   to XLA's GPU non-determinism (re-running a seed reproduces the reported
   number within the seed-to-seed spread).
3. Evaluate the final checkpoint of each run with `sokoban/eval_checkpoint.py`,
   `rushhour/eval_checkpoint.py` or `ogbench/eval_checkpoint.py` (commands and
   the numbers to expect in each README; a few minutes per checkpoint on one GPU).

All hyperparameters live in the train scripts and in the `Args` / `get_config`
defaults; nothing is set through environment variables except `MUJOCO_GL`
and the optional `RUSHHOUR_DATA_DIR`.
