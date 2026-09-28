# Looped actor on Boxoban (Sokoban)

Online PPO on Boxoban (Guez et al. 2018): 10x10 rooms, 4 boxes, 4 targets. The
agent trains on the 900k **unfiltered** training levels and is evaluated on the
held-out `unfiltered_valid` (100k) and `unfiltered_test` (1k) banks of the same
generator. Only the actor-critic trunk differs between arms; PPO, the
tokenizer, the heads and the budget are shared.

| arm | trunk | parameters | compute / decision |
|---|---|---|---|
| `looped` (ours) | one weight-tied 2-layer core, up to 16 think iterations, stops once consecutive policy readouts agree (KL < 1e-3) | 1x (0.53M) | adaptive, <= 16 core calls |
| `iso_flops` | 16 untied 2-layer cores stacked | 16x (8.4M) | 16 core calls |
| `iso_param` | the core applied once | 1x (0.53M) | 1 core call |

The PPO loss is on the final readout for every arm. The
observation is four planes per cell (wall, box, target, player), tokenized per
cell with grid geometry features plus one readout token; the four move logits
and the value are read from the readout token.

## Layout

```
sokoban/
  scripts/train.sh              one command per (arm, seed)
  eval_checkpoint.py            whole-bank evaluation, optional test-time cap, optional trajectories
envs/sokoban_env.py             JAX environment (banks from data/boxoban/*.npz)
data_scripts/build_boxoban_banks.py   builds the banks from google-deepmind/boxoban-levels
tests/test_sokoban_env.py
```

## Setup

```bash
pip install -r requirements.txt                    # plus the CUDA build of jax for training
python data_scripts/build_boxoban_banks.py         # downloads boxoban-levels (33 MB), writes data/boxoban/<split>.npz
python -m pytest -q tests/test_sokoban_env.py
```

## Training

```bash
sokoban/scripts/train.sh looped 1        # -> exp/sokoban/looped_s1/{train.log, checkpoints/<run>/params_<k>.pkl}
sokoban/scripts/train.sh iso_flops 1
sokoban/scripts/train.sh iso_param 1
```

100M environment steps, 1024 envs x 64 steps per rollout, 64 minibatches x 4
epochs, Adam 1e-4, grad-norm clip 1.0; one checkpoint and one 256-episode
evaluation on the training bank (`eval/train_*`) and on `unfiltered_valid`
(`eval/test_*`) per ~1M steps. Seeds 1-3 in the paper. Wall-clock on one
B200: `looped` ~13 h, `iso_flops` ~10 h, `iso_param` ~1 h. The final
checkpoint is `params_101.pkl`.

## Evaluation

```bash
python sokoban/eval_checkpoint.py --ckpt exp/sokoban/looped_s1/checkpoints/<run>/params_101.pkl
python sokoban/eval_checkpoint.py --ckpt exp/sokoban/iso_flops_s1/checkpoints/<run>/params_101.pkl --arch multi_block --num_blocks 16
python sokoban/eval_checkpoint.py --ckpt exp/sokoban/iso_param_s1/checkpoints/<run>/params_101.pkl --arch single_block
python sokoban/eval_checkpoint.py --ckpt exp/sokoban/looped_s1/checkpoints/<run>/params_101.pkl --max_iters 8   # test-time cap
```

Every level of a bank is played once with the argmax policy for at most 120
steps (`--max_levels 2000` by default; `0` = whole bank). Success rate (%) of
the paper's runs (final checkpoint, first 2000 levels), mean +- std over seeds 1-3:

| bank | iso_param | iso_flops | looped (cap 16) |
|---|---|---|---|
| unfiltered_valid | 0.3 +- 0.3 | 45.0 +- 5.5 | 80.8 +- 4.3 |
| unfiltered_test | 0.2 +- 0.3 | 42.8 +- 6.2 | 81.5 +- 5.7 |

Realized think iterations of the looped arm at cap 16: 9.8 on average
(91% of decisions halt before the cap). Per seed, unfiltered_test:
looped 75.1 / 83.1 / 86.2, iso_flops 49.4 / 42.2 / 36.8, iso_param 0.0 / 0.5 / 0.0.
