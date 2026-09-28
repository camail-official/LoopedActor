#!/bin/bash
# Train one arm of the Rush Hour study.
#   rushhour/scripts/train.sh <arm> <seed> [out_dir]
#   arm: looped | iso_flops | iso_param
# All arms: PPO on the easy_train bank (Fogleman puzzles with <= 15 moves), in-training evaluation on easy_valid
# (eval/test_* metrics), 1024 envs x 64 steps per rollout, 64 minibatches x 4 epochs, 100M env steps, 2-layer d=128
# core, plane tokenizer (7 planes per cell), per-cell action head (2 logits per cell = 72 actions), global grad-norm
# clip 1.0, potential-based reward shaping (weight 1.0), loss on the final readout for every arm.
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3)
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
# Requires data/rushhour/{easy_train,easy_valid,easy_test}.npz (python data_scripts/build_rushhour_banks.py --src rush.txt).
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/rushhour}"
case "$ARM" in
  looped)    ARM_FLAGS=(--architecture=fprm --max_think_iters=16 --halt_criterion=kl --halt_kl=1e-3) ;;
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks=16) ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/${ARM}_s${SEED}"
exec python -u ppo.py --env_id=rushhour-easy_train-easy_valid --seed="$SEED" --wandb_dir="$OUT/${ARM}_s${SEED}" \
  --fprm_num_layers=2 --max_grad_norm=1.0 \
  --num_envs=1024 --rollout_length=64 --num_minibatches_per_rollout=64 --num_epochs_per_rollout=4 \
  --num_timesteps=100000000 --num_eval_steps=100 --num_reset_steps=100 --num_eval_envs=256 \
  "${ARM_FLAGS[@]}" 2>&1 | tee -a "$OUT/${ARM}_s${SEED}/train.log"
