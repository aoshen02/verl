#!/usr/bin/env bash
#SBATCH --job-name=inferencex-mopd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=batch_1
#SBATCH --account=benchmark
#SBATCH --qos=batch_1_qos
#SBATCH --time=04:00:00
#SBATCH --output=mopd-%j.out

set -euo pipefail
: "${VERL_ROOT:?Set to the absolute cloned verl directory}"
: "${VERL_IMAGE:?Set to a prepared absolute .sqsh path or pinned Pyxis registry image}"
: "${MOPD_ASSETS:?Set to the shared model asset directory}"
: "${MOPD_DATA:?Set to the prepared parquet directory}"
: "${MOPD_OUTPUT:?Set to a NEW shared run directory}"
[[ "$SLURM_JOB_NUM_NODES" == 2 ]] || { echo 'Exactly two nodes required' >&2; exit 1; }
if [[ "${MOPD_BOOTSTRAP:-0}" == 1 ]]; then
    [[ "$VERL_IMAGE" =~ ^registry-1.docker.io#verlai/verl:sha256:[a-f0-9]{64}$ ]] || {
        echo 'Bootstrap requires a digest-pinned official verl image' >&2; exit 1;
    }
else
    [[ "$VERL_IMAGE" == /* && -f "$VERL_IMAGE" ]] || {
        echo 'VERL_IMAGE must be an existing absolute squashfs image' >&2; exit 1;
    }
fi
for path in "$VERL_ROOT" "$MOPD_ASSETS" "$MOPD_DATA" "$MOPD_OUTPUT"; do
    [[ "$path" == /* && "$path" != *[[:space:],:]* ]] || { echo "Unsupported path: $path" >&2; exit 1; }
done
mkdir "$MOPD_OUTPUT"
export STUDENT_MODEL="$MOPD_ASSETS/student"
export CAESAR_TEACHER_MODEL="$MOPD_ASSETS/teachers/caesar_cipher"
export GEOMETRY_TEACHER_MODEL="$MOPD_ASSETS/teachers/simple_geometry"
export MOPD_RUN_NAME="inferencex-mopd-$SLURM_JOB_ID-$(basename "$MOPD_OUTPUT")"
export MOPD_EVAL_TEACHERS="${MOPD_EVAL_TEACHERS:-0}"
export MOPD_TEACHER_BASELINES="${MOPD_TEACHER_BASELINES:-}"
if [[ "$MOPD_EVAL_TEACHERS" == 1 && -z "$MOPD_TEACHER_BASELINES" ]]; then
    export MOPD_TEACHER_BASELINES="$MOPD_OUTPUT/teacher-baselines.json"
fi
export VERL_FILE_LOGGER_PATH="$MOPD_OUTPUT/metrics.jsonl"
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MOPD_HEAD_NODE="${nodes[0]}"
export MOPD_HEAD_IP
MOPD_HEAD_IP=$(getent ahostsv4 "$MOPD_HEAD_NODE" | awk 'NR==1 {print $1}')
[[ -n "$MOPD_HEAD_IP" ]] || { echo 'Cannot resolve Ray head IPv4 address' >&2; exit 1; }
export MOPD_RAY_PORT=$((20000 + SLURM_JOB_ID % 10000))
# Set NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME to the site's working fabric if needed.
# Do not guess interface names or disable IB by default.
git -C "$VERL_ROOT" rev-parse HEAD > "$MOPD_OUTPUT/verl-commit.txt"
git -C "$VERL_ROOT" diff > "$MOPD_OUTPUT/working-tree.diff"
cp "$VERL_ROOT/uv.lock" "$MOPD_OUTPUT/uv.lock"
cp -R --no-preserve=mode,ownership,timestamps \
    "$VERL_ROOT/examples/on_policy_distillation_trainer/qwen3_6_35b_mopd" "$MOPD_OUTPUT/example-source"
if [[ "${MOPD_BOOTSTRAP:-0}" == 1 ]]; then
    printf '%s\n' "$VERL_IMAGE" > "$MOPD_OUTPUT/image-reference.txt"
else
    sha256sum "$VERL_IMAGE" > "$MOPD_OUTPUT/image.sha256"
fi
container_options=()
if [[ -n "${MOPD_CONTAINER_NAME:-}" ]]; then
    container_options+=(--container-name="$MOPD_CONTAINER_NAME")
fi
srun --jobid="$SLURM_JOB_ID" --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-node=8 --cpus-per-task=80 "${container_options[@]}" \
    --container-image="$VERL_IMAGE" \
    --container-mounts="$VERL_ROOT:$VERL_ROOT,$MOPD_ASSETS:$MOPD_ASSETS,$MOPD_DATA:$MOPD_DATA,$MOPD_OUTPUT:$MOPD_OUTPUT" \
    --container-workdir="$VERL_ROOT" --container-writable --container-remap-root \
    bash "$VERL_ROOT/examples/on_policy_distillation_trainer/qwen3_6_35b_mopd/run_slurm_node.sh" "$@"
