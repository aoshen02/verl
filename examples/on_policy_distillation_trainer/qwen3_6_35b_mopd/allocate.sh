#!/usr/bin/env bash
# Run under tmux/nohup while queued. Once granted, salloc --no-shell is independent.
set -euo pipefail
example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$example_dir/../../.." && pwd)
workspace=$(dirname "$root")
mkdir -p "$workspace/allocations"
request_log=$(mktemp "$workspace/allocations/request.XXXXXXXX.log")
echo "Allocation request log: $request_log"
if salloc --no-shell --job-name=inferencex-mopd --nodes=2 --ntasks-per-node=1 \
    --gpus-per-node=8 --cpus-per-task=80 --mem=0 --exclusive \
    --partition=batch_1 --account=benchmark --qos=batch_1_qos --time=04:00:00 \
    > "$request_log" 2>&1; then
    cat "$request_log"
else
    cat "$request_log" >&2
    exit 1
fi
job_id=$(awk '/Granted job allocation/ {print $NF}' "$request_log" | tail -1)
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo 'Could not read granted allocation ID' >&2; exit 1; }
printf '%s\n' "$job_id" > "$workspace/allocations/latest-job-id.txt"
# Failure of this step cannot release the independent no-shell allocation.
bash "$example_dir/run_in_allocation.sh" "$job_id" "$@"
