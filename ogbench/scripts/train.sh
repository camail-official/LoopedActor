#!/bin/bash
# Train one arm of the OGBench study.
#   scripts/train.sh <env> <arm> <seed> [save_dir]
#   env: cube_double | cube_triple | puzzle_3x3 | scene
#   arm: looped | iso_flops | iso_param
# All arms: GCIQL (alpha=1, discount 0.99, batch 1024, 1M gradient steps, eval every
# 100k steps on 50 episodes/task), a 2-layer d=128 FPRM core, TQL-style per-scalar
# tokenizer (paired obs/goal tokens on the puzzle), global grad-norm clip 1.0.
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3),
#               full BPTT through the executed iterations, loss on the final readout.
#   iso_flops : 16 untied cores stacked (16x the parameters), loss on the final readout.
#   iso_param : one core applied once (same parameters as the looped actor).
set -euo pipefail
ENV="${1:?env}"; ARM="${2:?arm}"; SEED="${3:?seed}"; SAVE_DIR="${4:-exp}"
case "$ENV" in
  cube_double) ENV_NAME=cube-double-play-v0; TOK=cube ;;
  cube_triple) ENV_NAME=cube-triple-play-v0; TOK=cube ;;
  puzzle_3x3)  ENV_NAME=puzzle-3x3-play-v0;  TOK=puzzle_paired ;;
  scene)       ENV_NAME=scene-play-v0;       TOK=scene_scalar ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
case "$ARM" in
  looped)    ARM_FLAGS=(--agent.fprm_train_iters=16 --agent.fprm_halt_on_policy_kl=0.001) ;;
  iso_flops) ARM_FLAGS=(--agent.fprm_num_blocks=16 --agent.fprm_train_iters=16) ;;
  iso_param) ARM_FLAGS=(--agent.fprm_num_blocks=1 --agent.fprm_train_iters=1) ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/.."
exec python main.py \
  --env_name="$ENV_NAME" \
  --agent=agents/gciql_fprm.py \
  --train_steps=1000000 --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
  --eval_episodes=50 --video_episodes=0 --eval_on_cpu=0 \
  --save_dir="$SAVE_DIR/$ENV" --run_group="$ARM" --seed="$SEED" \
  --agent.alpha=1.0 --agent.max_grad_norm=1.0 \
  --agent.fprm_tokenizer="$TOK" --agent.fprm_use_grid_conv=False \
  "${ARM_FLAGS[@]}"
