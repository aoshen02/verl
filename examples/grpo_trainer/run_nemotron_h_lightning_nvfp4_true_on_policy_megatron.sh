#!/usr/bin/env bash
# DAPO | Nemotron-H Lightning NVFP4 | vLLM rollout | Megatron Lite training
# True on-policy: vLLM batch-invariant rollout and the Megatron Lite NVFP4
# actor produce bitwise-equal response logprobs for the same policy version.
#
# Runs inside the image from docker/Dockerfile.nemotron_h_true_on_policy, which
# installs pinned vLLM, Megatron-LM (megatron.lite + verl_mlite) and verl.
# Nothing is mounted or patched at run time; verl_mlite is selected through
# Hydra (`pkg://verl_mlite.config` and `model_engine=mlite`, as the DeepSeek-V4
# recipe does; `engine.impl=vllm` is the vLLM-aligned Nemotron-H implementation)
# and its engine registers itself via `engine.custom_backend_module`.
#
# Topology (GB200, 4 GPUs/node; colocated hybrid engine on one node):
#   actor:   Megatron Lite PP4, TP1/EP1/CP1, dense DP1, dist_opt
#   rollout: vLLM TP1 DP4 EP4, Humming W4A16 MoE, FlashInfer one-sided all2all
# The Megatron Lite NVFP4 actor requires world size == PP == 4.
# Containers need the IMEX channel (/dev/nvidia-caps-imex-channels) for the
# FlashInfer one-sided all2all.
#
# Two-update acceptance: ACCEPTANCE_STEPS=2 stops after two optimizer updates
# without changing data, lengths, batch, rollout or the LR schedule.
set -euo pipefail

usage() {
  echo "usage: $0 [Hydra overrides...]"
  echo "required env: MODEL_PATH TRAIN_FILES"
  echo "optional env: VAL_FILES OUTPUT_ROOT ACCEPTANCE_STEPS DRY_RUN COMPOSE_ONLY ..."
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

# --- Inputs ---
: "${MODEL_PATH:?set MODEL_PATH to the Nemotron-H Lightning NVFP4 checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to the DAPO-Math-17k parquet/jsonl}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
MODEL_REVISION="${MODEL_REVISION:-bee7596271d1495f6992ae224aefde4410e816b8}"
TRAIN_FILES_SHA256="${TRAIN_FILES_SHA256:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"

# --- Algorithm and data ---
# Batch, lengths and optimizer follow the DeepSeek-V4 true on-policy recipe
# (examples/grpo_trainer/run_deepseek_v4_true_on_policy_preview_megatron.sh,
# aligned mode).
SEED="${SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-14000}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-False}"
FILTER_GROUPS="${FILTER_GROUPS:-False}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# --- Optimizer and LR schedule ---
ACTOR_LR="${ACTOR_LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-null}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETAS="${BETAS:-[0.9,0.95]}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ACCEPTANCE_STEPS="${ACCEPTANCE_STEPS:-}"

# --- Topology ---
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
ACTOR_PP="${ACTOR_PP:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_DP="${ROLLOUT_DP:-4}"
ROLLOUT_EP="${ROLLOUT_EP:-4}"

# --- Megatron Lite NVFP4 actor contract ---
MLITE_SURROGATE_CONTRACT="${MLITE_SURROGATE_CONTRACT:-moe-fixedscale-grouped-tf32rz-bf16edges-v3}"
MLITE_ROUTED_VJP_BACKEND="${MLITE_ROUTED_VJP_BACKEND:-compact-f32-tma-nosplit}"
MLITE_ROUTED_VJP_TOKEN_LIMIT="${MLITE_ROUTED_VJP_TOKEN_LIMIT:-16384}"
# Dynamic micro-batches pack whole sequences; without context parallelism one
# micro-batch must hold the longest sequence and stay within the VJP bound.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
MLITE_ROUTED_FORWARD_REDUCTION="${MLITE_ROUTED_FORWARD_REDUCTION:-ep4-fi-onesided-fp32-top6-first-rank-v1}"

# --- vLLM rollout (historical serving configuration) ---
ROLLOUT_MOE_BACKEND="${ROLLOUT_MOE_BACKEND:-humming}"
ROLLOUT_ALL2ALL_BACKEND="${ROLLOUT_ALL2ALL_BACKEND:-flashinfer_nvlink_one_sided}"
ROLLOUT_KV_CACHE_DTYPE="${ROLLOUT_KV_CACHE_DTYPE:-fp8_e4m3}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
ROLLOUT_WORKER_EXTENSION_CLS="${ROLLOUT_WORKER_EXTENSION_CLS-verl_mlite.rollout.layerwise_reload.LayerwiseReloadWorkerExtension}"
ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-8}"

# --- Outputs ---
PROJECT_NAME="${PROJECT_NAME:-verl-nemotron-h-true-on-policy}"
RUN_NAME="${RUN_NAME:-nemotron_h_lightning_nvfp4_dapo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/nemotron_h_true_on_policy}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
if [[ -z "${TRAINER_LOGGERS:-}" ]]; then
  TRAINER_LOGGERS='[console,file]'
  [[ -v WANDB_API_KEY && "${WANDB_MODE:-}" != disabled ]] && TRAINER_LOGGERS='[console,file,wandb]'
fi

# --- Validation ---
(( NNODES * NGPUS_PER_NODE == ACTOR_PP )) ||
  die "the NVFP4 actor requires world size == ACTOR_PP (got $((NNODES * NGPUS_PER_NODE)) vs ${ACTOR_PP})"
(( ROLLOUT_TP * ROLLOUT_DP == NGPUS_PER_NODE )) ||
  die "rollout TP*DP must equal NGPUS_PER_NODE"
(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH <= ROLLOUT_MAX_MODEL_LEN )) ||
  die "prompt+response exceeds ROLLOUT_MAX_MODEL_LEN"
(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH <= PPO_MAX_TOKEN_LEN_PER_GPU )) ||
  die "prompt+response exceeds PPO_MAX_TOKEN_LEN_PER_GPU"
(( PPO_MAX_TOKEN_LEN_PER_GPU <= MLITE_ROUTED_VJP_TOKEN_LIMIT )) ||
  die "PPO_MAX_TOKEN_LEN_PER_GPU exceeds MLITE_ROUTED_VJP_TOKEN_LIMIT"
TOTAL_TRAINING_STEPS=null
if [[ -n "${ACCEPTANCE_STEPS}" ]]; then
  # Capping the run must not rescale the schedule: warmup is in absolute
  # steps, and any decay must be anchored to the formal run with LR_DECAY_STEPS.
  [[ "${LR_DECAY_STYLE}" == constant || "${LR_DECAY_STEPS}" != null ]] ||
    die "ACCEPTANCE_STEPS with LR_DECAY_STYLE=${LR_DECAY_STYLE} needs an explicit LR_DECAY_STEPS"
  TOTAL_TRAINING_STEPS="${ACCEPTANCE_STEPS}"
fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"
  IFS=, read -r -a train_files <<<"${TRAIN_FILES}"
  IFS=, read -r -a val_files <<<"${VAL_FILES}"
  for file in "${train_files[@]}" "${val_files[@]}"; do
    [[ -f "${file}" ]] || die "missing data file: ${file}"
  done
  if [[ -n "${TRAIN_FILES_SHA256}" ]]; then
    [[ "$(sha256sum "${train_files[0]}" | cut -d' ' -f1)" == "${TRAIN_FILES_SHA256}" ]] ||
      die "TRAIN_FILES sha256 mismatch"
  fi
  if [[ "${NNODES}" -gt 1 ]]; then
    : "${RAY_ADDRESS:?multi-node runs require an existing Ray cluster}"
  fi
  mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${JSONL_FILE}")"
fi

# --- Process environment (propagated to every Ray worker below) ---
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER=1
# Batch-invariant Humming serving needs the indexed GEMM; do not inherit an
# image default.
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
export PYTHONHASHSEED="${SEED}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"
RAY_ENV_NAMES=(
  VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER VLLM_HUMMING_MOE_GEMM_TYPE
  PYTHONHASHSEED PYTHONNOUSERSITE VERL_FILE_LOGGER_PATH
)
for name in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL HF_HUB_OFFLINE NCCL_MNNVL_ENABLE; do
  [[ -v "${name}" ]] && RAY_ENV_NAMES+=("${name}")
done
RAY_RUNTIME_ENV=()
for name in "${RAY_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\"")
done

# model_engine=mlite selects mlite_actor/mlite_ref from verl_mlite; the critic
# group needs a disabled mlite entry of its own.
runtime_config_root="$(mktemp -d "${TMPDIR:-/tmp}/nemotron-h-config.XXXXXX")"
trap 'rm -rf "${runtime_config_root}"' EXIT
mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
printf '%s\n' '# @package _global_' 'model_engine: mlite' \
  >"${runtime_config_root}/model_engine/mlite.yaml"
printf '%s\n' '_target_: verl.workers.config.CriticConfig' 'enable: false' 'strategy: mlite' \
  >"${runtime_config_root}/critic/mlite_critic.yaml"

ROLLOUT_EXTRA=()
if [[ -n "${ROLLOUT_WORKER_EXTENSION_CLS}" ]]; then
  ROLLOUT_EXTRA+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=${ROLLOUT_WORKER_EXTENSION_CLS}")
fi

# --- Hydra overrides ---
ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}"
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.filter_groups.enable="${FILTER_GROUPS}"
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}"
  algorithm.rollout_correction.bypass_mode=False
)

DATA=(
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.dataloader_num_workers=0
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.filter_overlong_prompts=False
  data.truncation=error
  data.seed="${SEED}"
  +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}"
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code="${TRUST_REMOTE_CODE}"
  actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}"
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.optim.lr_warmup_init=0.0
  actor_rollout_ref.actor.optim.lr_decay_style="${LR_DECAY_STYLE}"
  actor_rollout_ref.actor.optim.lr_decay_steps="${LR_DECAY_STEPS}"
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
  actor_rollout_ref.actor.optim.betas="${BETAS}"
  actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}"
  actor_rollout_ref.actor.engine.impl=vllm
  '~actor_rollout_ref.actor.engine.grad_offload'
  '~actor_rollout_ref.ref.engine.grad_offload'
  actor_rollout_ref.actor.engine.tp=1
  actor_rollout_ref.actor.engine.etp=1
  actor_rollout_ref.actor.engine.ep=1
  actor_rollout_ref.actor.engine.cp=1
  actor_rollout_ref.actor.engine.vpp=1
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
  actor_rollout_ref.actor.engine.param_offload=False
  actor_rollout_ref.actor.engine.optimizer_offload=True
  actor_rollout_ref.actor.engine.load_hf_weights=True
  actor_rollout_ref.actor.engine.export_dtype=null
  actor_rollout_ref.actor.engine.attention_backend_override=null
  +actor_rollout_ref.actor.engine.full_determinism=True
  +actor_rollout_ref.actor.engine.seed="${SEED}"
  +actor_rollout_ref.actor.engine.impl_cfg.optimizer=dist_opt
  +actor_rollout_ref.actor.engine.impl_cfg.diagnostic_full_training=True
  +actor_rollout_ref.actor.engine.impl_cfg.surrogate_contract="${MLITE_SURROGATE_CONTRACT}"
  +actor_rollout_ref.actor.engine.impl_cfg.routed_vjp_backend="${MLITE_ROUTED_VJP_BACKEND}"
  +actor_rollout_ref.actor.engine.impl_cfg.routed_vjp_token_limit="${MLITE_ROUTED_VJP_TOKEN_LIMIT}"
  +actor_rollout_ref.actor.engine.impl_cfg.routed_forward_reduction="${MLITE_ROUTED_FORWARD_REDUCTION}"
  +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_WORKERS}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.full_determinism=False
  actor_rollout_ref.rollout.seed="${SEED}"
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}"
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend="${ROLLOUT_MOE_BACKEND}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend="${ROLLOUT_ALL2ALL_BACKEND}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype="${ROLLOUT_KV_CACHE_DTYPE}"
)

REWARD=(
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}"
  +reward.reward_kwargs.overlong_buffer_cfg.log=False
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
)

TRAINER=(
  critic.enable=False
  trainer.use_v1=False
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${RUN_NAME}"
  trainer.nnodes="${NNODES}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.val_before_train=False
  trainer.resume_mode=disable
  trainer.default_local_dir="${CKPT_DIR}"
)

COMMAND=(
  python3 -m verl.trainer.main_ppo
  "hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]"
  model_engine=mlite
  "${ALGORITHM[@]}"
  "${DATA[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${ROLLOUT_EXTRA[@]}"
  "${REWARD[@]}"
  "${TRAINER[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "$@"
)

printf 'MODEL_REVISION=%s TOPOLOGY=%sx%s ACTOR_PP=%s ROLLOUT=TP%s/DP%s/EP%s STEPS=%s\n' \
  "${MODEL_REVISION}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_PP}" \
  "${ROLLOUT_TP}" "${ROLLOUT_DP}" "${ROLLOUT_EP}" "${TOTAL_TRAINING_STEPS}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
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
