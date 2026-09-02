#!/usr/bin/env bash
# Run the L8 RL smoke inside the immutable specialized-image uv profile.

set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=${RUN_DIR:-/mnt/ephemeral/aoshen/qwen38/run/l8-e16-fsdp8-dapo-rl-20260901}
IMAGE=${IMAGE:-/mnt/ephemeral/aoshen/qwen38/images/qwen38-verl-fsdp-vllm-uv-20260901.sqsh}
TRAIN_MODEL=${TRAIN_MODEL:-/mnt/ephemeral/aoshen/qwen38/derived/qwen38-flash-next-bf16-trainer-smoke-l8-e16-mtp-ple2048-p8-v1}
ROLLOUT_MODEL=${ROLLOUT_MODEL:-/mnt/ephemeral/aoshen/qwen38/derived/qwen38-flash-next-fp8-serving-smoke-l8-e16-mtp-ple2048-p8-v1}
DATA=${DATA:-/mnt/ephemeral/aoshen/qwen38/data/dapo-math-17k-smoke-l8-32x8-20260831}
TRITON_CACHE_DIR_HOST=${TRITON_CACHE_DIR_HOST:-/mnt/ephemeral/aoshen/qwen38/cache/triton}

mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR_HOST"
export ENROOT_CACHE_PATH=/mnt/ephemeral/aoshen/enroot/cache
export ENROOT_DATA_PATH=/mnt/ephemeral/aoshen/enroot/data
export ENROOT_RUNTIME_PATH=/mnt/ephemeral/aoshen/enroot/runtime
export ENROOT_TEMP_PATH=/mnt/ephemeral/aoshen/enroot/tmp

exec enroot start --rc "$WORKTREE/scripts/enroot_exec.sh" \
    -e UV_PROJECT_ENVIRONMENT=/opt/verl-uv-final \
    -e UV_CACHE_DIR=/tmp/uv-cache \
    -e XDG_CACHE_HOME=/run/xdg-cache \
    -e FLASHINFER_WORKSPACE_BASE=/run \
    -e TRITON_CACHE_DIR=/var/tmp \
    -e VLLM_DO_NOT_TRACK=1 \
    -e DRY_RUN="${DRY_RUN:-0}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRAIN_MODEL_PATH=/models/q0 \
    -e ROLLOUT_MODEL_PATH=/models/qwen38 \
    -e ENABLE_MTP="${ENABLE_MTP:-false}" \
    -e ENFORCE_EAGER="${ENFORCE_EAGER:-true}" \
    -e TRAIN_FILE=/opt/data/train.parquet \
    -e VAL_FILE=/opt/data/val.parquet \
    -e OUTPUT_DIR=/run/output \
    -e VERL_FILE_LOGGER_ROOT=/run/output \
    -m "$WORKTREE:/workspace:none:bind,ro" \
    -m "$TRAIN_MODEL:/models/q0:none:bind,ro" \
    -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
    -m "$DATA:/opt/data:none:bind,ro" \
    -m "$RUN_DIR:/run:none:bind,rw" \
    -m "$TRITON_CACHE_DIR_HOST:/var/tmp:none:bind,rw" \
    "$IMAGE" /bin/bash -lc \
    'set -eu; source /opt/verl-uv-final/bin/activate; cd /workspace; bash scripts/run_qwen_l8_verl_smoke.sh'
