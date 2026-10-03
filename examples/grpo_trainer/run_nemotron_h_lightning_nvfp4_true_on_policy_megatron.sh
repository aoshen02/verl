#!/usr/bin/env bash
# DAPO | Nemotron-H Lightning NVFP4 | vLLM rollout | Megatron Lite training
# True on-policy: vLLM batch-invariant rollout and the Megatron Lite NVFP4
# actor produce bitwise-equal response logprobs for the same policy version.
#
# Runs inside the image from docker/Dockerfile.nemotron_h_true_on_policy, which
# installs pinned vLLM, Megatron-LM (megatron.lite + verl_mlite) and verl, and
# ships this script with nemotron_h_true_on_policy/ (Hydra groups, model
# manifests) in /opt/nemotron/recipe; nothing is mounted. verl_mlite is selected
# through Hydra (`pkg://verl_mlite.config` and `model_engine=mlite`, as the
# DeepSeek-V4 recipe does; `engine.impl=vllm` is the vLLM-aligned Nemotron-H
# implementation) and its engine registers itself via
# `engine.custom_backend_module`. The one runtime patch is the DeepSeek-V4
# recipe's: importing verl_mlite.engine replaces verl's bucketed weight sender
# (verl_mlite.compat._patch_bucketed_weight_sender).
#
# Topology (GB200, 4 GPUs/node; colocated hybrid engine on one node):
#   actor:   Megatron Lite PP4, TP1/EP1/CP1, dense DP1, dist_opt
#   rollout: vLLM TP1 DP4 EP4, Humming W4A16 MoE, FlashInfer one-sided all2all
# The Megatron Lite NVFP4 actor requires world size == PP == 4.
# Containers need the IMEX channel (/dev/nvidia-caps-imex-channels) for the
# FlashInfer one-sided all2all.
#
# Inputs are verified before launch: both checkpoints against the per-file
# sha256 manifests of their pinned revisions (nemotron_h_true_on_policy/
# manifests), every data file against its sha256.
#
# ACCEPTANCE_STEPS=N stops after N trainer iterations without changing data,
# lengths, batch, rollout or the LR schedule. Each iteration makes
# TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE optimizer updates (4 by default).
#
# Deviations from the DeepSeek-V4 aligned recipe:
# - rollout.full_determinism=False: True gives every sample of a request the
#   same vLLM seed, so the n GRPO samples of a prompt coincide and the
#   advantages vanish. Bitwise rollout/training agreement comes from
#   VLLM_BATCH_INVARIANT=1 (set below), not from the VERL_FULL_DETERMINISM,
#   CUBLAS_WORKSPACE_CONFIG and NCCL_ALGO settings that True would add.
# - PP4/TP1/EP1/CP1 actor with dist_opt and a 16384-token micro-batch budget:
#   the NVFP4 actor needs world size == PP; without CP one micro-batch holds
#   the longest sequence.
# - Humming W4A16 MoE (indexed GEMM) and FlashInfer one-sided all2all instead
#   of DeepGEMM/DeepEP; FP8 KV cache, raw logprobs and no prefix caching are
#   the serving contract the actor replays.
# - Rollout batched tokens 16384 and memory fraction 0.7; no full recompute or
#   cross-entropy fusion; use_fused_kernels=False and trust_remote_code=False
#   (Nemotron-H is native in transformers).
# Settings that equal current defaults (clipping, KL, entropy, epochs, LR
# schedule, parallel sizes) are spelled out to pin the contract.
set -euo pipefail

usage() {
  echo "usage: $0 [Hydra overrides...]"
  echo "required env: MODEL_PATH BF16_MASTER_PATH TRAIN_FILES TRAIN_FILES_SHA256"
  echo "optional env: VAL_FILES VAL_FILES_SHA256 OUTPUT_ROOT ACCEPTANCE_STEPS DRY_RUN COMPOSE_ONLY ..."
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nemotron_h_true_on_policy"

# --- Inputs ---
: "${MODEL_PATH:?set MODEL_PATH to the Nemotron-H Lightning NVFP4 checkpoint}"
# The BF16 release the NVFP4 checkpoint was quantized from
# (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d24): the actor's initial
# BF16 masters. The actor refuses a release that is not the checkpoint's source.
: "${BF16_MASTER_PATH:?set BF16_MASTER_PATH to the Nemotron-H Lightning BF16 release}"
: "${TRAIN_FILES:?set TRAIN_FILES to the DAPO-Math-17k parquet/jsonl}"
MODEL_REVISION="${MODEL_REVISION:-bee7596271d1495f6992ae224aefde4410e816b8}"
BF16_MASTER_REVISION="${BF16_MASTER_REVISION:-a9904d24bcc1d289a1950fa9d2b978c47cf903b9}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${recipe_dir}/manifests/nvfp4@${MODEL_REVISION}.sha256}"
BF16_MASTER_MANIFEST="${BF16_MASTER_MANIFEST:-${recipe_dir}/manifests/bf16@${BF16_MASTER_REVISION}.sha256}"
# Comma-separated, one sha256 per file in TRAIN_FILES / VAL_FILES.
TRAIN_FILES_SHA256="${TRAIN_FILES_SHA256:-}"
VAL_FILES_SHA256="${VAL_FILES_SHA256:-}"
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

# --- Megatron Lite NVFP4 actor ---
# Dynamic micro-batches pack whole sequences; without context parallelism one
# micro-batch must hold the longest sequence.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
# Host RAM the run adds on a 4-GPU node, measured on GB200 (full model, BF16
# masters, CPU-offloaded optimizer): 734 GiB at the end of the first update.
# 0 disables the check.
HOST_MEM_MIN_GIB="${HOST_MEM_MIN_GIB:-760}"
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
# verl's default extension would treat the ModelOpt checkpoint as unquantized
# and rerun its non-idempotent post-load processing on every weight update.
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
if (( TEST_FREQ > 0 )); then
  : "${VAL_FILES:?TEST_FREQ>0 needs an explicit held-out VAL_FILES}"
  [[ "${VAL_FILES}" != "${TRAIN_FILES}" ]] || die "VAL_FILES must not be the training files"
fi
# Without validation verl still builds the val dataset; it is never evaluated.
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
[[ "${VAL_FILES}" != "${TRAIN_FILES}" || -n "${VAL_FILES_SHA256}" ]] || VAL_FILES_SHA256="${TRAIN_FILES_SHA256}"
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
[[ -n "${ROLLOUT_WORKER_EXTENSION_CLS}" ]] ||
  die "ROLLOUT_WORKER_EXTENSION_CLS must name the quantized-reload worker extension"
TOTAL_TRAINING_STEPS=null
if [[ -n "${ACCEPTANCE_STEPS}" ]]; then
  # Capping the run must not rescale the schedule: warmup is in absolute
  # steps, and any decay must be anchored to the formal run with LR_DECAY_STEPS.
  [[ "${LR_DECAY_STYLE}" == constant || "${LR_DECAY_STEPS}" != null ]] ||
    die "ACCEPTANCE_STEPS with LR_DECAY_STYLE=${LR_DECAY_STYLE} needs an explicit LR_DECAY_STEPS"
  TOTAL_TRAINING_STEPS="${ACCEPTANCE_STEPS}"
fi

# verify_model <dir> <manifest>: every listed file matches, and no weight or
# config file is unlisted.
verify_model() {
  local dir="$1" manifest unlisted
  [[ -s "$2" ]] || die "no manifest $2"
  manifest="$(realpath "$2")"
  [[ -d "${dir}" ]] || die "missing model directory ${dir}"
  unlisted="$(cd "${dir}" && find . -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.json' \) -printf '%f\n' |
    sort | comm -23 - <(awk '{print $2}' "${manifest}" | sort))"
  [[ -z "${unlisted}" ]] || die "${dir} has files outside ${manifest}: ${unlisted//$'\n'/ }"
  awk '{print $2}' "${manifest}" | (cd "${dir}" && xargs -P 16 -n 1 sha256sum) |
    sort -k2 | diff -q - <(sort -k2 "${manifest}") >/dev/null ||
    die "${dir} does not match ${manifest}"
}

# verify_files <comma-separated files> <comma-separated sha256s> <name>
verify_files() {
  local -a files sums
  local i
  IFS=, read -r -a files <<<"$1"
  IFS=, read -r -a sums <<<"$2"
  (( ${#files[@]} == ${#sums[@]} )) || die "set $3 to one sha256 per file"
  for i in "${!files[@]}"; do
    [[ -f "${files[i]}" ]] || die "missing data file: ${files[i]}"
    [[ "$(sha256sum "${files[i]}" | cut -d' ' -f1)" == "${sums[i]}" ]] ||
      die "sha256 mismatch: ${files[i]}"
  done
}

if [[ "${DRY_RUN:-0}" != 1 && "${COMPOSE_ONLY:-0}" != 1 ]]; then
  verify_model "${MODEL_PATH}" "${MODEL_MANIFEST}"
  verify_model "${BF16_MASTER_PATH}" "${BF16_MASTER_MANIFEST}"
  verify_files "${TRAIN_FILES}" "${TRAIN_FILES_SHA256}" TRAIN_FILES_SHA256
  verify_files "${VAL_FILES}" "${VAL_FILES_SHA256}" VAL_FILES_SHA256
  if [[ "${NNODES}" -gt 1 ]]; then
    : "${RAY_ADDRESS:?multi-node runs require an existing Ray cluster}"
  elif (( HOST_MEM_MIN_GIB > 0 )); then
    available_gib=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
    (( available_gib >= HOST_MEM_MIN_GIB )) ||
      die "host RAM available ${available_gib} GiB < ${HOST_MEM_MIN_GIB} GiB the CPU-offloaded optimizer needs; free the node or set HOST_MEM_MIN_GIB"
  fi
  mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${JSONL_FILE}")"
fi

# --- Process environment (propagated to every Ray worker below) ---
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER=1
# Batch-invariant Humming serving needs the indexed GEMM; do not inherit an
# image default.
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
# Cold starts (no compile caches) take longer than vLLM's 600 s default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export PYTHONHASHSEED="${SEED}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"
RAY_ENV_NAMES=(
  VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER VLLM_HUMMING_MOE_GEMM_TYPE
  VLLM_ENGINE_READY_TIMEOUT_S PYTHONHASHSEED PYTHONNOUSERSITE VERL_FILE_LOGGER_PATH
)
for name in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL HF_HUB_OFFLINE NCCL_MNNVL_ENABLE; do
  [[ -v "${name}" ]] && RAY_ENV_NAMES+=("${name}")
done
RAY_RUNTIME_ENV=()
for name in "${RAY_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\"")
done


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
  actor_rollout_ref.actor.engine.grad_offload=True
  actor_rollout_ref.actor.engine.tp=1
  actor_rollout_ref.actor.engine.etp=1
  actor_rollout_ref.actor.engine.ep=1
  actor_rollout_ref.actor.engine.cp=1
  actor_rollout_ref.actor.engine.vpp=1
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
  # Parameters stay resident (as in the DeepSeek-V4 recipe); the FP32 gradient
  # buffers are released while the rollout owns the GPU.
  actor_rollout_ref.actor.engine.param_offload=False
  actor_rollout_ref.actor.engine.optimizer_offload=True
  actor_rollout_ref.actor.engine.load_hf_weights=True
  actor_rollout_ref.actor.engine.export_dtype=null
  actor_rollout_ref.actor.engine.attention_backend_override=null
  +actor_rollout_ref.actor.engine.full_determinism=True
  +actor_rollout_ref.actor.engine.seed="${SEED}"
  +actor_rollout_ref.actor.engine.impl_cfg.routed_forward_reduction="${MLITE_ROUTED_FORWARD_REDUCTION}"
  +actor_rollout_ref.actor.engine.impl_cfg.bf16_master_path="${BF16_MASTER_PATH}"
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
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="${ROLLOUT_WORKER_EXTENSION_CLS}"
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
  "hydra.searchpath=[file://${recipe_dir}/config,pkg://verl_mlite.config]"
  model_engine=mlite
  "${ALGORITHM[@]}"
  "${DATA[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${REWARD[@]}"
  "${TRAINER[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "$@"
)

printf 'MODEL_REVISION=%s BF16_MASTER_REVISION=%s TOPOLOGY=%sx%s ACTOR_PP=%s ROLLOUT=TP%s/DP%s/EP%s STEPS=%s\n' \
  "${MODEL_REVISION}" "${BF16_MASTER_REVISION}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_PP}" \
  "${ROLLOUT_TP}" "${ROLLOUT_DP}" "${ROLLOUT_EP}" "${TOTAL_TRAINING_STEPS}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

# COMPOSE_ONLY=1: what main_ppo does before starting Ray (compose, validate,
# resolve, build the engine and rollout configs); prints the resolved config.
if [[ "${COMPOSE_ONLY:-0}" == 1 ]]; then
  python3 - "${COMMAND[@]:3}" <<'PY'
import sys

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config

with initialize_config_module("verl.trainer.config", version_base=None):
    cfg = compose("ppo_trainer", overrides=sys.argv[1:])
validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))
OmegaConf.resolve(cfg)
omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
print(OmegaConf.to_yaml(cfg))
PY
  exit 0
fi

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
run_rc="${PIPESTATUS[0]}"
set -e
exit "${run_rc}"
