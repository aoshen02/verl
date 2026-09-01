#!/usr/bin/env bash
# Four-step FSDP2 + merged-LoRA + vLLM rollout gate for the verified L8 checkpoint.

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the derived L8 BF16 trainer checkpoint}"
: "${TRAIN_FILE:?set TRAIN_FILE to the frozen smoke train parquet}"
: "${VAL_FILE:?set VAL_FILE to the frozen smoke validation parquet}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a writable checkpoint directory}"
: "${UV_PROJECT_ENVIRONMENT:?set this to a target-image system-site uv venv}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
SMOKE_STEPS=${SMOKE_STEPS:-4}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-32}
EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
PROJECT_NAME=${PROJECT_NAME:-qwen38_l8_rl_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-fsdp2_lora_merge_mtp}
EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-0.1.dev20073+g8e685d198}
EXPECTED_TORCH_VERSION=${EXPECTED_TORCH_VERSION:-2.13.0+cu130}
EXPECTED_TRANSFORMERS_VERSION=${EXPECTED_TRANSFORMERS_VERSION:-5.16.0}

if ((NNODES != 1 || NGPUS_PER_NODE != 8)); then
    echo "the L8 smoke is fixed to one eight-GPU node" >&2
    exit 2
fi
if ((TRAIN_BATCH_SIZE <= 0 || ROLLOUT_N <= 1 || ROLLOUT_TP <= 0 || SMOKE_STEPS <= 0)); then
    echo "batch size, rollout count, TP, and smoke steps must be positive; rollout count must exceed one" >&2
    exit 2
fi
if ((NGPUS_PER_NODE % ROLLOUT_TP != 0)); then
    echo "NGPUS_PER_NODE must be divisible by ROLLOUT_TP" >&2
    exit 2
fi
if ((EXPECTED_TRAIN_ROWS != TRAIN_BATCH_SIZE * SMOKE_STEPS || EXPECTED_VAL_ROWS <= 0)); then
    echo "expected train rows must equal TRAIN_BATCH_SIZE * SMOKE_STEPS and expected val rows must be positive" >&2
    exit 2
fi

# This provisional gate inherits the immutable image runtime through a uv venv
# created with `uv venv --system-site-packages`. It deliberately never asks uv
# to install vLLM: the image contains the model-specific build. The final
# evidence gate will replace this with an upstream-supported frozen uv image.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_ALLOW_UNSUPPORTED_VLLM_VERSION=1
UV_PY=(uv run --frozen --no-sync python3)
runtime_versions=$("${UV_PY[@]}" -c '
import torch
import transformers
import vllm
print(vllm.__version__)
print(torch.__version__)
print(transformers.__version__)
print(vllm.__file__)
')
mapfile -t runtime_lines <<< "$runtime_versions"
if [ "${runtime_lines[0]}" != "$EXPECTED_VLLM_VERSION" ] || \
    [ "${runtime_lines[1]}" != "$EXPECTED_TORCH_VERSION" ] || \
    [ "${runtime_lines[2]}" != "$EXPECTED_TRANSFORMERS_VERSION" ] || \
    [[ "${runtime_lines[3]}" != /usr/local/lib/python3.12/dist-packages/vllm/* ]]; then
    printf 'unexpected image runtime: %s\n' "$runtime_versions" >&2
    exit 2
fi
train_rows=$("${UV_PY[@]}" -c 'import pyarrow.parquet as p, sys; print(p.ParquetFile(sys.argv[1]).metadata.num_rows)' "$TRAIN_FILE")
val_rows=$("${UV_PY[@]}" -c 'import pyarrow.parquet as p, sys; print(p.ParquetFile(sys.argv[1]).metadata.num_rows)' "$VAL_FILE")
if [ "$train_rows" != "$EXPECTED_TRAIN_ROWS" ] || [ "$val_rows" != "$EXPECTED_VAL_ROWS" ]; then
    echo "smoke parquet cardinality differs: train=$train_rows/$EXPECTED_TRAIN_ROWS val=$val_rows/$EXPECTED_VAL_ROWS" >&2
    exit 2
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=8
    actor_rollout_ref.model.lora_alpha=16
    'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj]'
    actor_rollout_ref.model.lora.merge=True
    actor_rollout_ref.model.mtp.enable=True
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=True
    actor_rollout_ref.model.mtp.method=mtp
    actor_rollout_ref.model.mtp.num_speculative_tokens=3
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.layered_summon=False
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=1024
    actor_rollout_ref.rollout.max_num_batched_tokens=1024
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024
    actor_rollout_ref.rollout.seed=17
    actor_rollout_ref.rollout.full_determinism=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024
)

TRAINER=(
    trainer.logger='["console","file"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.total_epochs=1
    trainer.val_before_train=False
    trainer.save_freq=${SMOKE_STEPS}
    trainer.test_freq=${SMOKE_STEPS}
    trainer.default_local_dir=${OUTPUT_DIR}
)

RAY=(
    '+ray_kwargs.ray_init.num_gpus=8'
    'ray_kwargs.ray_init.runtime_env.py_executable=uv -v run --frozen --no-sync'
)

EXTRA=()
if [ "${DRY_RUN:-0}" = 1 ]; then
    EXTRA+=(--cfg job --resolve)
fi

exec "${UV_PY[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "${EXTRA[@]}" \
    "$@"
