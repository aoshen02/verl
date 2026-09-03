#!/usr/bin/env bash
# FSDP2 LoRA + FP8 vLLM adapter-update recipe.

set -euo pipefail

: "${TRAIN_MODEL_PATH:?set TRAIN_MODEL_PATH to the derived L8 BF16 trainer checkpoint}"
: "${ROLLOUT_MODEL_PATH:?set ROLLOUT_MODEL_PATH to the matching L8 FP8 serving checkpoint}"
: "${TRAIN_FILE:?set TRAIN_FILE to the frozen smoke train parquet}"
: "${VAL_FILE:?set VAL_FILE to the frozen smoke validation parquet}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a writable checkpoint directory}"
: "${UV_PROJECT_ENVIRONMENT:?set this to the baked specialized-image uv venv}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TRAINING_STEPS=${TRAINING_STEPS:-${SMOKE_STEPS:-4}}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-}
EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
USE_KL_LOSS=${USE_KL_LOSS:-true}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
SAVE_FREQ=${SAVE_FREQ:-$TRAINING_STEPS}
TEST_FREQ=${TEST_FREQ:-$TRAINING_STEPS}
REWARD_MODE=${REWARD_MODE:-math_dapo}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
RESUME_MODE=${RESUME_MODE:-disable}
PROJECT_NAME=${PROJECT_NAME:-qwen38_l8_rl_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-fsdp2_lora_adapter_only}
ENABLE_MTP=${ENABLE_MTP:-false}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-0.1.dev20073+g8e685d198}
EXPECTED_TORCH_VERSION=${EXPECTED_TORCH_VERSION:-2.13.0+cu130}
EXPECTED_TRANSFORMERS_VERSION=${EXPECTED_TRANSFORMERS_VERSION:-5.16.0}

if ((NNODES <= 0 || NGPUS_PER_NODE <= 0)); then
    echo "NNODES and NGPUS_PER_NODE must be positive" >&2
    exit 2
fi
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
if ((TRAIN_BATCH_SIZE <= 0 || ROLLOUT_N <= 1 || ROLLOUT_TP <= 0 || TRAINING_STEPS <= 0)); then
    echo "batch size, rollout count, TP, and training steps must be positive; rollout count must exceed one" >&2
    exit 2
fi
if [[ "$ENABLE_MTP" != false && "$ENABLE_MTP" != true ]]; then
    echo "ENABLE_MTP must be false or true" >&2
    exit 2
fi
if [[ "$ENFORCE_EAGER" != false && "$ENFORCE_EAGER" != true ]]; then
    echo "ENFORCE_EAGER must be false or true" >&2
    exit 2
fi
if [[ "$USE_KL_LOSS" != false && "$USE_KL_LOSS" != true ]]; then
    echo "USE_KL_LOSS must be false or true" >&2
    exit 2
fi
if ((TOTAL_GPUS % ROLLOUT_TP != 0)); then
    echo "total GPUs must be divisible by ROLLOUT_TP" >&2
    exit 2
fi
if ((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN)); then
    echo "prompt plus response length exceeds MAX_MODEL_LEN" >&2
    exit 2
fi
if ((SAVE_FREQ <= 0 || TEST_FREQ <= 0 || LORA_RANK <= 0 || LORA_ALPHA <= 0)); then
    echo "save/test frequency and LoRA rank/alpha must be positive" >&2
    exit 2
fi
if ((PPO_MINI_BATCH_SIZE <= 0 || PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE)); then
    echo "PPO_MINI_BATCH_SIZE must be in [1, TRAIN_BATCH_SIZE]" >&2
    exit 2
fi
if [[ "$REWARD_MODE" != math_dapo && "$REWARD_MODE" != smoke ]]; then
    echo "REWARD_MODE must be math_dapo or smoke" >&2
    exit 2
fi
if [[ "$VAL_BEFORE_TRAIN" != false && "$VAL_BEFORE_TRAIN" != true ]]; then
    echo "VAL_BEFORE_TRAIN must be false or true" >&2
    exit 2
fi
if [[ "$RESUME_MODE" != auto && "$RESUME_MODE" != disable ]]; then
    echo "RESUME_MODE must be auto or disable" >&2
    exit 2
fi

# This gate uses the uv environment baked into the specialized image. It never
# installs vLLM: the image contains the model-specific build.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_ALLOW_UNSUPPORTED_VLLM_VERSION=1
UV_PY=(uv run --active --frozen --no-sync python)
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
if [[ -n "$EXPECTED_TRAIN_ROWS" && "$train_rows" != "$EXPECTED_TRAIN_ROWS" ]]; then
    echo "train parquet cardinality differs: train=$train_rows/$EXPECTED_TRAIN_ROWS" >&2
    exit 2
fi
if [[ -n "$EXPECTED_VAL_ROWS" && "$val_rows" != "$EXPECTED_VAL_ROWS" ]]; then
    echo "validation parquet cardinality differs: val=$val_rows/$EXPECTED_VAL_ROWS" >&2
    exit 2
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.dataloader_num_workers=0
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path=${TRAIN_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj]'
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.mtp.enable=${ENABLE_MTP}
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=${ENABLE_MTP}
    actor_rollout_ref.model.mtp.method=mtp
    actor_rollout_ref.model.mtp.num_speculative_tokens=3
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.model_path=${ROLLOUT_MODEL_PATH}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.seed=17
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.total_epochs=1
    trainer.total_training_steps=${TRAINING_STEPS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.resume_mode=${RESUME_MODE}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.default_local_dir=${OUTPUT_DIR}
)

RAY=(
    'ray_kwargs.ray_init.runtime_env.py_executable=/opt/verl-uv-final/bin/python'
)
if [[ -n "${RAY_ADDRESS:-}" ]]; then
    RAY+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
else
    RAY+=(
        'ray_kwargs.ray_init.num_cpus=48'
        "+ray_kwargs.ray_init.num_gpus=${TOTAL_GPUS}"
    )
fi

REWARD=()
if [[ "$REWARD_MODE" == smoke ]]; then
    REWARD=(
        reward.custom_reward_function.path=${REPO_ROOT}/scripts/qwen_l8_smoke_reward.py
        reward.custom_reward_function.name=compute_score
    )
fi

HYDRA=(
    hydra.run.dir=/run/hydra
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
    "${REWARD[@]}" \
    "${HYDRA[@]}" \
    "${EXTRA[@]}" \
    "$@"
