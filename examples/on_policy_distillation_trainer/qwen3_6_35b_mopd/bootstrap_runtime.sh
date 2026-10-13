#!/usr/bin/env bash
# Materialize the pinned environment in each node's private Pyxis container.
set -euo pipefail
: "${VERL_ROOT:?}"
export UV_PROJECT_ENVIRONMENT=/opt/verl-venv
export UV_PYTHON=3.12
export NO_VCS_VERSION=1
fingerprint=$(cat "$VERL_ROOT/uv.lock" "$0" | sha256sum | awk '{print $1}')
if [[ -x /opt/verl-venv/bin/python && -f /opt/mopd-bootstrap.sha256 && \
      "$(cat /opt/mopd-bootstrap.sha256)" == "$fingerprint" ]]; then
    echo 'Reusing prepared runtime in this allocation'
    exit 0
fi
# Build metadata in a node-local copy to avoid simultaneous writes to NFS.
build_dir=$(mktemp -d /opt/verl-build.XXXXXXXX)
tar -C "$VERL_ROOT" --exclude=.git --exclude=.venv --exclude=__pycache__ -cf - . |
    tar -C "$build_dir" -xf -
cd "$build_dir"
uv sync --frozen --all-packages --extra vllm --extra fsdp --no-dev --no-editable
uv pip install --python /opt/verl-venv/bin/python 'reasoning-gym==0.1.25'
/opt/verl-venv/bin/python - <<'PY'
import importlib.metadata as m
for name, expected in {'vllm': '0.29.0', 'transformers': '5.12.1', 'reasoning-gym': '0.1.25'}.items():
    assert m.version(name) == expected, (name, m.version(name), expected)
import torch
assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__
import flash_attn, fla, transfer_queue
print('Pinned runtime prepared', flush=True)
PY
sha256sum uv.lock > /opt/verl-uv-lock.sha256
printf '%s\n' "$fingerprint" > /opt/mopd-bootstrap.sha256
