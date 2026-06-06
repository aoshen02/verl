#!/usr/bin/env bash
# Qwen3-235B-A22B PRODUCTION GRPO training (SWE-bench, 6T+6R topology).
#
# Purpose: production-scale RL training on SWE-bench with uni-agent rollout.
# Originally a diagnostic fork (2026-05-18) for the 35B 1e8 grad_norm
# investigation; repurposed for prod after 30B + 235B both confirmed healthy
# grad_norm at production parallelism.
#
# Topology (per agent_run/results/qwen3_235b/6node_topology_design/TOPOLOGY.md):
#   ACTOR (train, 6 nodes × 4 GPU = 24 GPU, DP=1):
#     TP=2 CP=2 PP=6 VPP=null EP=4 ETP=1
#     num_layers_in_first_pipeline_stage = num_layers_in_last_pipeline_stage = 15
#     (94 layers = 15 + 4×16 + 15, must be odd per Megatron PP constraint)
#     COMMON_OFFLOAD=False, optimizer_offload_fraction=1.0 (match reference)
#     recompute_granularity=selective, recompute_method=uniform (match reference)
#   ROLLOUT (inference, 6 nodes × 4 GPU = 6 vLLM replicas):
#     INFER_TP=4, gpu_memory_utilization=0.93 (qwen3_235b_bench N=32 sweep)
#     performance_mode=interactivity (sweep winner)
#     prefix-caching ON, max_num_batched_tokens=8192 (defaults)
#
# Model: Qwen3-235B-A22B-Instruct-2507 (NOT the base Qwen3-235B-A22B).
#   The Instruct-2507 variant has max_position_embeddings=262144 (256K
#   NATIVE), so the reference's 128K context just works without YaRN.
#   The base Qwen3-235B-A22B had only 40K native and required YaRN wiring
#   on both vLLM and Megatron sides; deleted from lustre to free space.
#
# Stack: SWE-bench + uni-agent (NOT DAPO Math):
#   - data: swe_rebench_filtered_modal.parquet / swe_bench_verified_modal.parquet
#   - agent: uni_agent.agent_loop.UniAgentLoop (Modal swe-rex sandboxes)
#   - reward: vanilla GRPO (no DAPO overlong buffer)
#   - MTP off (235B has no native MTP head)
#   - max_prompt 4K, max_response 128K (Qwen3.5-35B round-11 实测 mean ≈ 70K)
#
# Rollout config from agent_run/results/qwen3_235b_bench/SWEEP_SUMMARY_N32_N40.md:
#   - performance_mode=interactivity (sweep N=32 winner, +1.5% req/s, TPOT p50 20.4ms)
#   - VLLM_USE_DEEP_GEMM=0 (vllm 0.21 EP/CUTLASS init workaround)
#   - INFER_MEM_UTILIZATION=0.93 (sweep verified)
#   - DO NOT enable -O3 (only marginal at N=40, neutral at N=32)
#   - DO NOT enable VLLM_USE_FLASHINFER_MOE_FP16 + CUTLASS (-20% throughput on B200)
#
# Agent concurrency target = 48 × NNODES_ROLLOUT (= 288 trajectories):
#   per-replica effective vllm in-flight ≈ 32 after Modal tool-call gaps,
#   matching sweep N=32 sweet spot. See configs/agent_config_train_235b.yaml.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
# qwen3_235b_bench N=32 sweep workaround for vllm 0.21 EP/CUTLASS init
# (see agent_run/results/qwen3_235b_bench/SWEEP_SUMMARY_N32_N40.md)
export VLLM_USE_DEEP_GEMM=0

# ================= ray + experiment naming =================
# ================= Verda GB300 — head=tray02, 8T+7R = 15 trays ============
# Diff from GB200 6T+6R variant:
#   - RAY_ADDRESS: 10.0.2.108 (node-08) → 10.0.1.166 (tray02 bond0)
#   - RAY_DATA_HOME: /mnt/shared → /mnt/shared/user
#   - MODEL_PATH: HF cache snapshot glob → direct path (hf download --local-dir)
#   - NNODES_TRAIN 6→8, NNODES_ROLLOUT 6→7
#   - CKPTS_DIR + PROMETHEUS path under /mnt/shared/user/
#   - AGENT_CONFIG + RUNTIME_ENV use *_verda variants
# Topology (ACTOR_TP/CP/PP/EP) NOT yet adjusted for 8 train nodes — user will
# re-derive PP layer split + DP grouping for 32 GPU before launch.
RAY_ADDRESS=${RAY_ADDRESS:-http://10.0.0.13:8265}

# ================= data / model =================
RAY_DATA_HOME=${RAY_DATA_HOME:-/mnt/shared/user}
# SOLO_ROOT used to locate configs/. On Verda the project files are flat under
# /mnt/shared/user/{scripts,configs,repos}/ — no /workspace/vllm/... mount.
SOLO_ROOT=${SOLO_ROOT:-/mnt/shared/user}
# Qwen3-235B-A22B-Instruct-2507 (256K native context, no YaRN needed).
# Downloaded via `hf download --local-dir` → direct dir, NOT HF cache snapshot format.
MODEL_PATH=${MODEL_PATH:-${RAY_DATA_HOME}/models/Qwen3-235B-A22B-Instruct-2507}
TRAIN_FILE=${TRAIN_FILE:-${RAY_DATA_HOME}/data/swe_agent/swe_rebench_filtered_modal.parquet}
TEST_FILE=${TEST_FILE:-${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_modal.parquet}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-${SOLO_ROOT}/configs/agent_config_train_235b_verda.yaml}
RUNTIME_ENV=${RUNTIME_ENV:-${SOLO_ROOT}/configs/runtime_env.verda.yaml}

project_name=${PROJECT_NAME:-'Qwen3-235B-A22B-Instruct-2507-grpo-prod'}
exp_name=${EXP_NAME:-"qwen3_235b_instruct_verda_$(date +%H%M)_8T8R"}
CKPTS_DIR=${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}}
mkdir -p "${CKPTS_DIR}"

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=low_var_kl
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 4))
# SWE-bench trajectory shape: Qwen3.5-35B round-11 实测 mean response ≈ 70K, turns ≈ 92,
# clip 1-8%; 32K 一半 trajectory 直接撞 cap. 必须 128K 起.
max_response_length=$((1024 * 128))
loss_mode="vanilla"
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
val_top_p=0.95

# ================= performance — 6-node training topology =================
# Per TOPOLOGY.md (agent_run/results/qwen3_235b/6node_topology_design/):
# 6 train nodes × 4 GPU = 24 GPU. ACTOR TP=2 × CP=2 × PP=6 = 24, DP=1.
# ETP×EP×PP = 1×4×6 = 24 ✓. 6-node math drops optimizer.step GPU peak
# from 180 GB (4-node) → ~85 GB.
INFER_TP=${INFER_TP:-4}
INFER_EP=${INFER_EP:-1}
INFER_MEM_UTILIZATION=${INFER_MEM_UTILIZATION:-0.93}  # PROD (N=32 sweep on Verda GB300)
update_weights_bucket_megabytes=2048

ACTOR_TP=${ACTOR_TP:-4}      # PROD: TP=4 intra-tray NVLink
ACTOR_CP=${ACTOR_CP:-4}      # PROD: CP=4 for 128K response
ACTOR_PP=${ACTOR_PP:-3}      # PROD: PP=3 across train trays
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-16}     # PROD: EP=16
ACTOR_ETP=${ACTOR_ETP:-1}

# Offload — match reference (8-node DAPO Math): COMMON_OFFLOAD=False.
# At 6 nodes the per-rank weight footprint (11+11 GB) is still small enough
# to keep on GPU; rely on verl-side precision-aware optimizer_cpu_offload only.
COMMON_OFFLOAD=${COMMON_OFFLOAD:-False}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-$COMMON_OFFLOAD}
GRAD_OFFLOAD=${GRAD_OFFLOAD:-$COMMON_OFFLOAD}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-$COMMON_OFFLOAD}
optimizer_cpu_offload=True
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / ACTOR_CP))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / ACTOR_CP))
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2

# ================= async policy =================
rollout_name="vllm"
rollout_mode="async"

# Production cluster: 6 train (Megatron TP=2 CP=2 PP=6 EP=4) + 6 rollout (vllm INFER_TP=4 × 6 replicas) = 12 GB200 nodes.
NNODES_ROLLOUT=${NNODES_ROLLOUT:-6}   # PROD
NNODES_TRAIN=${NNODES_TRAIN:-12}      # PROD
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

train_batch_size=0
gen_prompt_bsz=1
n_resp_per_prompt=8                     # PROD: GRPO group size
ppo_mini_batch_size=64                  # PROD
total_rollout_steps=200000              # PROD: reference 512*400
test_freq=${TEST_FREQ:-20}              # match reference
save_freq=${SAVE_FREQ:-1}               # PROD: every step, MAX_CKPT_TO_KEEP=2 caps storage
staleness_threshold=1.0                 # user-confirmed (reference 0.5)
trigger_parameter_sync_step=2           # PROD
require_batches=1
partial_rollout=True
val_before_train=${VAL_BEFORE_TRAIN:-False}

# Rollout Importance Sampling (from reference)
rollout_is=null
rollout_rs=seq_mean_k1
rollout_rs_threshold="0.999_1.001"

USE_MBRIDGE=True
VANILLA_MBRIDGE=True
USE_DIST_CKPT=False

# ================= Prometheus monitoring =================
ENABLE_PROMETHEUS_MONITORING=${ENABLE_PROMETHEUS_MONITORING:-true}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
PROMETHEUS_CONFIG_FILE=${PROMETHEUS_CONFIG_FILE:-/mnt/shared/user/monitoring/config/prometheus.yml}
PROMETHEUS_SERVED_MODEL_NAME=${PROMETHEUS_SERVED_MODEL_NAME:-qwen3_235b_a22b_prod}

prometheus_params=()
if [[ "$ENABLE_PROMETHEUS_MONITORING" == "true" ]]; then
    prometheus_params=(
        actor_rollout_ref.rollout.prometheus.enable=True
        actor_rollout_ref.rollout.prometheus.port=${PROMETHEUS_PORT}
        actor_rollout_ref.rollout.prometheus.file=${PROMETHEUS_CONFIG_FILE}
        actor_rollout_ref.rollout.prometheus.served_model_name=${PROMETHEUS_SERVED_MODEL_NAME}
    )
    echo "[train] Prometheus monitoring ENABLED"
else
    prometheus_params=(actor_rollout_ref.rollout.prometheus.enable=False)
    echo "[train] Prometheus monitoring DISABLED"
fi

# ================= MTP params =================
# DISABLED — 235B has no native MTP head (same as 30B reasoning).
mtp_params=(
  actor_rollout_ref.model.mtp.enable=False
  actor_rollout_ref.model.mtp.enable_train=False
  actor_rollout_ref.model.mtp.enable_rollout=False
)

CHECKPOINT_CONTENTS=['model','hf_model','extra']

ray job submit --no-wait --address=$RAY_ADDRESS --runtime-env $RUNTIME_ENV \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size="${train_batch_size}" \
    data.return_raw_chat=True \
    data.gen_batch_size=${gen_prompt_bsz} \
    +data.apply_chat_template_kwargs.thinking=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${VANILLA_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${optimizer_cpu_offload} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_CONTENTS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${INFER_MEM_UTILIZATION} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.expert_parallel_size=${INFER_EP} \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.performance_mode=interactivity \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.nccl_timeout=600 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-2} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    trainer.total_epochs=10 \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    "${mtp_params[@]}" \
    "${prometheus_params[@]}" \
    "$@"
