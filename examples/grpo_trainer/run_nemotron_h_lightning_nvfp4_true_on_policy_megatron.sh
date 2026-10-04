#!/usr/bin/env bash
# Nemotron-H Lightning NVFP4 true-on-policy for VERL.
# aligned/quick_alignment_test use batch-invariant vLLM for exact probabilities.
# Modes: quick_alignment_test (1x4, short workload, three steps), aligned.
# Hardware: gb200 (1x4, PP1/EP4 actor, rollout DP4/EP4).
# Image builds on `Dockerfile.nemotron_h_true_on_policy`. Unlike the DeepSeek-V4
# preview, vLLM, Megatron-LM and VERL are installed in the image; nothing is mounted.
set -euo pipefail

# Required env: MODEL_PATH, BF16_MASTER_PATH, TRAIN_FILES, VAL_FILES.
# Optional env: training steps, batch size, lengths, output paths, WANDB_API_KEY,
# WANDB_ENTITY, WANDB_MODE, WANDB_BASE_URL, and Hydra overrides.
SEED="${SEED:-42}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
PROJECT_NAME="${PROJECT_NAME:-verl-nemotron-h-true-on-policy}"

usage() {
  echo "usage: $0 --hardware gb200 --mode {quick_alignment_test|aligned} [Hydra overrides...]"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

MODE="${MODE:-}"
HARDWARE="${HARDWARE:-}"
HYDRA_OVERRIDES=()
while (( $# > 0 )); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --hardware)
      [[ $# -ge 2 ]] || die "--hardware requires a value"
      HARDWARE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      HYDRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

[[ -n "${MODE}" ]] || die "set --mode quick_alignment_test or aligned"
[[ -n "${HARDWARE}" ]] || die "set --hardware gb200"
[[ "${HARDWARE}" == gb200 ]] || die "unknown hardware: ${HARDWARE}"

# --- Required inputs ---
: "${MODEL_PATH:?set MODEL_PATH to the Nemotron-H Lightning NVFP4 checkpoint}"
# The BF16 release the NVFP4 checkpoint was quantized from: the actor's masters.
: "${BF16_MASTER_PATH:?set BF16_MASTER_PATH to the Nemotron-H Lightning BF16 release}"
: "${TRAIN_FILES:?set TRAIN_FILES to DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to DAPO-format validation parquet}"
[[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"

# --- Mode presets ---
# quick_alignment_test keeps the full model (the actor requires the complete
# Lightning checkpoint); only the workload shrinks.
case "${MODE}" in
  quick_alignment_test)
    : "${TOTAL_TRAINING_STEPS:=3}"
    : "${TRAIN_BATCH_SIZE:=16}"
    : "${PPO_MINI_BATCH_SIZE:=8}"
    : "${OVERLONG_BUFFER_LEN:=512}"
    : "${ROLLOUT_N:=4}"
    : "${MAX_RESPONSE_LENGTH:=2048}"
    : "${ROLLOUT_MAX_NUM_SEQS:=64}"
    : "${PPO_MAX_TOKEN_LEN_PER_GPU:=8192}"
    : "${SAVE_FREQ:=-1}"
    : "${TEST_FREQ:=-1}"
    ;;
  aligned)
    : "${TOTAL_TRAINING_STEPS:=100}"
    : "${TRAIN_BATCH_SIZE:=128}"
    : "${PPO_MINI_BATCH_SIZE:=32}"
    : "${OVERLONG_BUFFER_LEN:=4096}"
    : "${ROLLOUT_N:=8}"
    : "${MAX_RESPONSE_LENGTH:=14000}"
    : "${ROLLOUT_MAX_NUM_SEQS:=128}"
    # Without context parallelism one micro-batch holds the longest sequence.
    : "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"
    : "${SAVE_FREQ:=5}"
    : "${TEST_FREQ:=10}"
    ;;
  *)
    die "unknown mode '${MODE}'"
    ;;
esac
# The FSDP2 actor keeps its FP32 shards on the GPU during rollout; at 0.7 vLLM's
# FlashInfer autotune warmup runs out of memory.
: "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.65}"

: "${ACTOR_OPTIMIZER:=fsdp2}"
case "${ACTOR_OPTIMIZER}" in
  dist_opt|fsdp2) ;;
  *) die "ACTOR_OPTIMIZER must be dist_opt or fsdp2, got '${ACTOR_OPTIMIZER}'" ;;
esac

if [[ -n "${TRAINER_LOGGERS:-}" ]]; then
  :
elif [[ "${WANDB_MODE:-}" == disabled ]]; then
  TRAINER_LOGGERS='[console,file]'
elif [[ -v WANDB_API_KEY || -v WANDB_ENTITY || -v WANDB_MODE ]]; then
  TRAINER_LOGGERS='[console,file,wandb]'
else
  TRAINER_LOGGERS='[console,file]'
fi

# --- Alignment behavior ---
: "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=16384}"
# Routed experts: FlashInfer CuTe-DSL W4A16, the kernel the actor calls.
: "${ROLLOUT_MOE_BACKEND:=flashinfer_cutedsl}"
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER=1
MODE_ARGS=(
  actor_rollout_ref.actor.engine.impl=vllm
  +actor_rollout_ref.actor.engine.seed="${SEED}"
  +actor_rollout_ref.actor.engine.full_determinism=True
  actor_rollout_ref.actor.engine.attention_backend_override=null
  # True gives the n samples of a prompt one vLLM seed; batch invariance, not
  # full determinism, makes rollout and actor agree.
  actor_rollout_ref.rollout.full_determinism=False
  actor_rollout_ref.rollout.seed="${SEED}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=flashinfer_nvlink_one_sided
)

OPTIMIZER_ARGS=(
  +actor_rollout_ref.actor.engine.impl_cfg.optimizer="${ACTOR_OPTIMIZER}"
  actor_rollout_ref.actor.engine.param_offload=False
  actor_rollout_ref.actor.engine.optimizer_offload=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True
)

# The ModelOpt checkpoint refits through vLLM's layerwise reload.
VLLM_WORKER_EXTENSION="verl_mlite.rollout.vllm_worker.MLiteVLLMColocateWorkerExtension"

# --- Validated hardware profile ---
: "${NNODES:=1}"
: "${NGPUS_PER_NODE:=4}"
: "${ACTOR_PP:=1}"
: "${ACTOR_CP:=1}"
: "${ACTOR_EP:=4}"
: "${ROLLOUT_DP:=4}"
: "${ROLLOUT_EP:=4}"
: "${ROLLOUT_AGENT_WORKERS:=8}"

runtime_config_root="$(mktemp -d /tmp/nemotron-h-true-on-policy-config.XXXXXX)"
trap 'rm -rf "${runtime_config_root}"' EXIT
mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
printf '%s\n' \
  '# @package _global_' \
  'model_engine: mlite' \
  >"${runtime_config_root}/model_engine/mlite.yaml"
printf '%s\n' \
  '_target_: verl.workers.config.CriticConfig' \
  'enable: false' \
  'strategy: mlite' \
  >"${runtime_config_root}/critic/mlite_critic.yaml"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_ARGS=(
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
  actor_rollout_ref.actor.engine.cp="${ACTOR_CP}"
  actor_rollout_ref.actor.engine.ep="${ACTOR_EP}"
  '~actor_rollout_ref.actor.engine.grad_offload'
  '~actor_rollout_ref.ref.engine.grad_offload'
  # The actor exports its NVFP4/FP8 deployment bytes as stored.
  actor_rollout_ref.actor.engine.export_dtype=null
  +actor_rollout_ref.actor.engine.impl_cfg.bf16_master_path="${BF16_MASTER_PATH}"
  +actor_rollout_ref.actor.engine.impl_cfg.recompute=full
)
ENGINE_ARGS=("hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]" model_engine=mlite)
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/nemotron_h_true_on_policy/${HARDWARE}/${MODE}}"
RUN_NAME="${RUN_NAME:-nemotron_h_${HARDWARE}_${MODE}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
if [[ "${DRY_RUN:-0}" != 1 && "${NNODES}" -gt 1 ]]; then
  : "${RAY_ADDRESS:?multi-node modes require an existing Ray cluster}"
fi

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${CKPT_DIR}" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${JSONL_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

# --- Internal container/Ray environment; normally do not edit ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
# Cold starts (no compile caches) take longer than vLLM's 600 s default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"

# Exporting in this launcher is not enough for an existing Ray cluster.
RAY_ENV_NAMES=(
  PYTHONNOUSERSITE PYTHONHASHSEED
  VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER VLLM_ENGINE_READY_TIMEOUT_S
  VERL_FILE_LOGGER_PATH
)
RAY_RUNTIME_ENV=()
for name in "${RAY_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
  )
done

for name in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL HF_HUB_OFFLINE NCCL_MNNVL_ENABLE; do
  if [[ -v "${name}" ]]; then
    RAY_RUNTIME_ENV+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
    )
  fi
done

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -s "${BF16_MASTER_PATH}/config.json" ]] || die "missing ${BF16_MASTER_PATH}/config.json"
  IFS=, read -r -a train_files <<<"${TRAIN_FILES}"
  IFS=, read -r -a val_files <<<"${VAL_FILES}"
  for file in "${train_files[@]}"; do
    [[ -f "${file}" ]] || die "missing train file: ${file}"
  done
  for file in "${val_files[@]}"; do
    [[ -f "${file}" ]] || die "missing validation file: ${file}"
  done
fi

# --- Hydra overrides: only non-default behavior ---
HYDRA_ARGS=(
  # Algorithm and data.
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.norm_adv_by_std_in_grpo=False
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.seed="${SEED}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  +data.apply_chat_template_kwargs.enable_thinking=True

  # Model and shared actor settings.
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0

  # vLLM rollout.
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_WORKERS}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="${VLLM_WORKER_EXTENSION}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8_e4m3
  +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend="${ROLLOUT_MOE_BACKEND}"
  # Reward and trainer.
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=False
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${RUN_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.max_actor_ckpt_to_keep=2
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
  trainer.use_v1=False
)

# --- Launch ---
COMMAND=(
  python3 -m verl.trainer.main_ppo
  "${ENGINE_ARGS[@]}"
  "${HYDRA_ARGS[@]}"
  "${ACTOR_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${MODE_ARGS[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "${HYDRA_OVERRIDES[@]}"
)

print_command() {
  local arg
  for arg in "${COMMAND[@]}"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

printf 'MODE=%s HARDWARE=%s TOPOLOGY=%sx%s OPTIMIZER=%s\n' \
  "${MODE}" "${HARDWARE}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_OPTIMIZER}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command
  exit 0
fi

if [[ "${COMPOSE_ONLY:-0}" == 1 ]]; then
  "${COMMAND[@]}" --cfg job --resolve
  exit 0
fi

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
run_rc="${PIPESTATUS[0]}"
set -e
exit "${run_rc}"
