#!/usr/bin/env bash
set -euo pipefail
cd "$VERL_ROOT"
if [[ "${MOPD_BOOTSTRAP:-0}" == 1 ]]; then
    bash "$VERL_ROOT/examples/on_policy_distillation_trainer/qwen3_6_35b_mopd/bootstrap_runtime.sh" \
        > "$MOPD_OUTPUT/bootstrap-$(hostname).log" 2>&1
fi
export PATH="/opt/verl-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/verl-venv
export RAY_ADDRESS="$MOPD_HEAD_IP:$MOPD_RAY_PORT"
export TRITON_CACHE_DIR="$MOPD_OUTPUT/cache/$(hostname)/triton"
export TORCHINDUCTOR_CACHE_DIR="$MOPD_OUTPUT/cache/$(hostname)/inductor"
export HF_HOME="$MOPD_OUTPUT/cache/$(hostname)/huggingface"
MODULE=examples.on_policy_distillation_trainer.qwen3_6_35b_mopd
ray_pid=''
cleanup() {
    if [[ -n "$ray_pid" ]]; then
        kill "$ray_pid" 2>/dev/null || true
        wait "$ray_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
node_ip=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
[[ -n "$node_ip" ]] || exit 1
if [[ "$SLURM_PROCID" == 0 ]]; then
    if [[ "${MOPD_BOOTSTRAP:-0}" == 1 && ! -f "$MOPD_ASSETS/models.json" ]]; then
        python -m "$MODULE.prepare_models" "$MOPD_ASSETS" > "$MOPD_OUTPUT/prepare-models.log" 2>&1
    fi
    python -m "$MODULE.preflight"
    if [[ "${MOPD_EVAL_TEACHERS:-0}" == 1 && ! -f "$MOPD_TEACHER_BASELINES" ]]; then
        python -m "$MODULE.evaluate_teachers" --domain caesar_cipher \
            --model "$CAESAR_TEACHER_MODEL" --data "$MOPD_DATA" --output "$MOPD_OUTPUT/teacher-eval"
        python -m "$MODULE.evaluate_teachers" --domain simple_geometry \
            --model "$GEOMETRY_TEACHER_MODEL" --data "$MOPD_DATA" --output "$MOPD_OUTPUT/teacher-eval"
        python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['MOPD_OUTPUT']) / 'teacher-eval'
domains = {d: json.loads((root / f'{d}.json').read_text()) for d in ('caesar_cipher', 'simple_geometry')}
Path(os.environ['MOPD_TEACHER_BASELINES']).write_text(json.dumps({'domains': domains}, indent=2))
PY
    fi
    if [[ -n "$MOPD_TEACHER_BASELINES" && ! -f "$MOPD_TEACHER_BASELINES" ]]; then
        echo "Supplied teacher baseline file does not exist: $MOPD_TEACHER_BASELINES" >&2
        exit 1
    fi
    echo "Teacher evaluation enabled: ${MOPD_EVAL_TEACHERS:-0}; starting training Ray cluster"
    ray start --head --node-ip-address="$node_ip" --port="$MOPD_RAY_PORT" \
        --num-cpus=64 --num-gpus=8 --include-dashboard=false \
        --temp-dir="/tmp/mopd-ray-$SLURM_JOB_ID-${SLURM_STEP_ID:-0}" --block > "$MOPD_OUTPUT/ray-head.log" 2>&1 &
    ray_pid=$!
    # The worker retries connection until the head is ready. No fixed startup sleep.
    touch "$MOPD_OUTPUT/head-starting"
    python - <<'PY'
import os, time, ray
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        ray.init(address=os.environ['RAY_ADDRESS'], logging_level='ERROR')
        break
    except (ConnectionError, RuntimeError):
        time.sleep(2)
else:
    raise RuntimeError('Ray head did not become ready')
while time.monotonic() < deadline:
    nodes = [n for n in ray.nodes() if n['Alive']]
    if len(nodes) == 2 and all(n['Resources'].get('GPU', 0) == 8 for n in nodes):
        break
    time.sleep(2)
else:
    raise RuntimeError('Expected two Ray nodes with eight GPUs each')
ray.shutdown()
PY
    VERL_USE_UV=0 bash "$VERL_ROOT/examples/on_policy_distillation_trainer/run_qwen3_6_35b_mopd_fsdp.sh" "$@"
    touch "$MOPD_OUTPUT/training-done"
else
    # Baseline eval can take a long time at 30k output tokens; Slurm bounds this wait.
    while [[ ! -f "$MOPD_OUTPUT/head-starting" ]]; do sleep 2; done
    python - <<'PY'
import os, socket, time
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with socket.create_connection((os.environ['MOPD_HEAD_IP'], int(os.environ['MOPD_RAY_PORT'])), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError('Ray head port did not become ready')
PY
    ray start --address="$RAY_ADDRESS" --node-ip-address="$node_ip" \
        --num-cpus=64 --num-gpus=8 --block > "$MOPD_OUTPUT/ray-worker.log" 2>&1 &
    ray_pid=$!
    while [[ ! -f "$MOPD_OUTPUT/training-done" ]]; do
        kill -0 "$ray_pid" || { echo 'Ray worker stopped unexpectedly' >&2; exit 1; }
        sleep 2
    done
fi
