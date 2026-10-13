#!/usr/bin/env bash
# Synchronous MOPD | Qwen3.6-35B-A3B | Caesar Cipher + Geometry Simple | FSDP2/vLLM
# Run from any directory against an existing 2-node, 16-GPU Ray cluster.
set -xeuo pipefail

export VERL_ROOT=${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$VERL_ROOT"

# ---- user-adjustable ----
# Prepare pinned checkpoints and parquet files using the example's preparation scripts.
export MOPD_ASSETS=${MOPD_ASSETS:-$HOME/data/qwen3_6_mopd/assets}
export MOPD_DATA=${MOPD_DATA:-$HOME/data/qwen3_6_mopd/data}
export STUDENT_MODEL=${STUDENT_MODEL:-$MOPD_ASSETS/student}
export CAESAR_TEACHER_MODEL=${CAESAR_TEACHER_MODEL:-$MOPD_ASSETS/teachers/caesar_cipher}
export GEOMETRY_TEACHER_MODEL=${GEOMETRY_TEACHER_MODEL:-$MOPD_ASSETS/teachers/simple_geometry}
export MOPD_OUTPUT=${MOPD_OUTPUT:-$VERL_ROOT/outputs/qwen3_6_mopd/$(date -u +%Y%m%dT%H%M%S)-$$}
actor_lr=${ACTOR_LR:-1e-6}
total_training_steps=${TOTAL_TRAINING_STEPS:-20}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}
# ---- end user-adjustable ----

# config.yaml holds the complete recipe, including matched avg@4 evaluation.
# Keep the 8 student GPUs + 4 GPUs per teacher layout for audited telemetry.
ACTOR=(actor_rollout_ref.actor.optim.lr="$actor_lr")
TRAINER=(
    trainer.total_training_steps="$total_training_steps"
    trainer.save_freq="$save_freq"
    trainer.test_freq="$test_freq"
)

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [[ "${VERL_USE_UV:-1}" != 0 ]]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp --with reasoning-gym==0.1.25 python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv run --frozen --all-packages --extra vllm --extra fsdp --with reasoning-gym==0.1.25")
fi

mkdir -p "$MOPD_OUTPUT"
"${LAUNCH[@]}" -m examples.on_policy_distillation_trainer.qwen3_6_35b_mopd.metrics.runtime \
    "${ACTOR[@]}" "${TRAINER[@]}" "${RAY[@]}" "$@" \
    --cfg job --resolve > "$MOPD_OUTPUT/resolved-config.yaml"
"${LAUNCH[@]}" -m examples.on_policy_distillation_trainer.qwen3_6_35b_mopd.metrics.runtime \
    "${ACTOR[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@"
