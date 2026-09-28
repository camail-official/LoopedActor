# Looped actor on Rush Hour

Online PPO on Fogleman's 6x6 Rush Hour puzzles. The agent trains on
`easy_train` (all puzzles with at most 15 moves, minus two held-out sets;
1.98M levels) and is evaluated on the held-out `easy_test` bank of the same
distribution. Only the actor-critic trunk differs between arms; PPO, the
tokenizer, the heads and the budget are shared.

| arm | trunk | parameters | compute / decision |
|---|---|---|---|
| `looped` (ours) | one weight-tied 2-layer core, up to 16 think iterations, stops once consecutive policy readouts agree (KL < 1e-3) | 1x | adaptive, <= 16 core calls |
| `iso_flops` | 16 untied 2-layer cores stacked | 16x | 16 core calls |
| `iso_param` | the core applied once | 1x | 1 core call |

The PPO loss is on the final readout for every arm. The
observation is seven planes per cell (occupied, red car, wall, horizontal
piece, vertical piece, link-right, link-down), tokenized per cell with grid
geometry features plus one readout token; the action is one of 72 = 36 cells x
2 directions (slide the piece at that cell one step up/left or down/right,
invalid moves are no-ops), read as two logits per cell token. Reward: -0.1 per
step, +10 on solving, plus potential-based shaping (weight 1.0) on the number of
blockers between the red car and the exit and the red car's distance to it;
150-step episodes.

## Layout

```
rushhour/
  scripts/train.sh              one command per (arm, seed)
  eval_checkpoint.py            held-out evaluation with success by move count, optional test-time cap
envs/rushhour_env.py            JAX environment (banks from data/rushhour/*.npz)
data_scripts/build_rushhour_banks.py   builds the banks from Fogleman's rush.txt
```

## Setup

```bash
pip install -r requirements.txt
curl -L -o rush.txt.gz https://www.michaelfogleman.com/static/rush/rush.txt.gz && gunzip rush.txt.gz   # 2,577,412 puzzles
python data_scripts/build_rushhour_banks.py --src rush.txt        # -> data/rushhour/{easy_train,easy_valid,easy_test}.npz
```

## Training

```bash
rushhour/scripts/train.sh looped 1        # -> exp/rushhour/looped_s1/{train.log, checkpoints/<run>/params_<k>.pkl}
rushhour/scripts/train.sh iso_flops 1
rushhour/scripts/train.sh iso_param 1
```

100M environment steps, 1024 envs x 64 steps per rollout, 64 minibatches x 4
epochs, Adam 1e-4, grad-norm clip 1.0; one checkpoint and one 256-episode
evaluation on the training bank (`eval/train_*`) and on `easy_valid`
(`eval/test_*`) per ~1M steps. Seeds 1-3 in the paper. Wall-clock on one H100:
`looped` ~11 h, `iso_flops` ~7 h, `iso_param` ~0.5 h. The final checkpoint is
`params_101.pkl`.

## Evaluation

```bash
python rushhour/eval_checkpoint.py --ckpt exp/rushhour/looped_s1/checkpoints/<run>/params_101.pkl
python rushhour/eval_checkpoint.py --ckpt exp/rushhour/iso_flops_s1/checkpoints/<run>/params_101.pkl --arch multi_block --num_blocks 16
python rushhour/eval_checkpoint.py --ckpt exp/rushhour/iso_param_s1/checkpoints/<run>/params_101.pkl --arch single_block
```

The first 2000 levels of `easy_test` (moves 3-15) are each played once with the
argmax policy for at most 150 steps. Success rate (%) of the paper's runs
(final checkpoint), mean +- std over seeds 1-3:

| bank | iso_param | iso_flops | looped (cap 16) |
|---|---|---|---|
| easy_test | 41.1 +- 1.3 | 75.2 +- 9.4 | 87.5 +- 1.0 |

Success by minimum move count (pooled over seeds): at 10 moves 95 / 85 / 6
(looped / iso_flops / iso_param), at 13 moves 78 / 60 / 23, at 15 moves
54 / 32 / 6. Realized think iterations of the looped arm at cap 16: 15.1
(21% of decisions halt before the cap).
