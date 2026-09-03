#!/usr/bin/env bash
# Formal 100-step DAPO-Math LoRA recipe; orchestration stays in the shared launcher.

set -euo pipefail

export TRAINING_STEPS=${TRAINING_STEPS:-100}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export USE_KL_LOSS=false
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-10}
export REWARD_MODE=math_dapo
export VAL_BEFORE_TRAIN=true
export RESUME_MODE=${RESUME_MODE:-auto}
export ENABLE_MTP=true
export ENFORCE_EAGER=true
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.76}
export EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-17917}
export EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-30}
export PROJECT_NAME=${PROJECT_NAME:-qwen38_full_dapo_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fsdp16_lora_r32_mtp_eager_4k}

exec "$(dirname "$0")/run_qwen_full_verl_adapter_multinode_slurm.sh"
