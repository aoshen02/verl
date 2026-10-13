#!/usr/bin/env bash
# Launch one attempt without creating or releasing a Slurm allocation.
set -euo pipefail
job_id=${1:?Usage: bash run_in_allocation.sh JOB_ID [Hydra overrides...]}
shift
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric allocation ID' >&2; exit 1; }
example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VERL_ROOT=${VERL_ROOT:-$(cd "$example_dir/../../.." && pwd)}
workspace=$(dirname "$VERL_ROOT")
export VERL_IMAGE=${VERL_IMAGE:-registry-1.docker.io#verlai/verl:sha256:53aa78ecb8c86637d41636a21e88bdd8b18bfb9a753e812dbe321fd6f922deb5}
export MOPD_BOOTSTRAP=${MOPD_BOOTSTRAP:-1}
export MOPD_ASSETS=${MOPD_ASSETS:-$workspace/assets}
export MOPD_DATA=${MOPD_DATA:-$workspace/data/reasoning_gym}
allocation_dir="$workspace/allocations/$job_id"
mkdir -p "$allocation_dir" "$workspace/runs" "$MOPD_ASSETS"
# NFS lock prevents overlapping attempts from sharing the same named containers.
exec 9> "$allocation_dir/attempt.lock"
flock -n 9 || { echo 'An attempt is already running in this allocation' >&2; exit 1; }
job=$(scontrol show job "$job_id" --oneliner)
[[ " $job " == *' JobState=RUNNING '* ]] || { echo 'Allocation is not running yet' >&2; exit 1; }
owner=$(sed -n 's/.* UserId=[^(]*(\([0-9]*\)).*/\1/p' <<< "$job")
[[ "$owner" == "$(id -u)" ]] || { echo 'Allocation belongs to another user' >&2; exit 1; }
export SLURM_JOB_ID="$job_id"
export SLURM_JOB_NODELIST
SLURM_JOB_NODELIST=$(tr ' ' '\n' <<< "$job" | sed -n 's/^NodeList=//p')
export SLURM_JOB_NUM_NODES
SLURM_JOB_NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
[[ "$SLURM_JOB_NUM_NODES" == 2 ]] || { echo 'Expected two allocated nodes' >&2; exit 1; }
export MOPD_CONTAINER_NAME="inferencex-mopd-$job_id"
attempt="attempt-$(date -u +%Y%m%dT%H%M%S)-$$"
export MOPD_OUTPUT="$workspace/runs/inferencex-mopd-$job_id-$attempt"
printf '%s\n' "$MOPD_OUTPUT" > "$allocation_dir/latest-output.txt"
# Completed baselines can be reused; the trainer verifies their provenance.
if [[ -f "$allocation_dir/baseline-output.txt" ]]; then
    baseline_output=$(cat "$allocation_dir/baseline-output.txt")
    if [[ -f "$baseline_output/teacher-baselines.json" ]]; then
        # Store the cache under the already-mounted asset directory.
        mkdir -p "$MOPD_ASSETS/evaluation-$job_id"
        cp "$baseline_output/teacher-baselines.json" "$MOPD_ASSETS/evaluation-$job_id/teacher-baselines.json"
        export MOPD_TEACHER_BASELINES="$MOPD_ASSETS/evaluation-$job_id/teacher-baselines.json"
    fi
fi
echo "Starting $attempt on allocation $job_id; output: $MOPD_OUTPUT"
if bash "$example_dir/submit.slurm" "$@" > "$allocation_dir/$attempt.log" 2>&1; then
    result=0
else
    result=$?
fi
printf '%s\n' "$result" > "$allocation_dir/$attempt.exit-code"
if [[ -f "$MOPD_OUTPUT/teacher-baselines.json" ]]; then
    printf '%s\n' "$MOPD_OUTPUT" > "$allocation_dir/baseline-output.txt"
fi
echo "Attempt exited $result. Allocation $job_id remains available until cancelled or its four-hour limit."
echo "Log: $allocation_dir/$attempt.log"
exit "$result"
