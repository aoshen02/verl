#!/usr/bin/env bash
# Bootstrap the prom + grafana stack on node-08 (current ray head).
#
# Idempotent: re-running brings up the stack; existing TSDB data is preserved.
#
# Run on node-08 host (NOT inside verl-bench container):
#   ssh node-08 'bash /mnt/shared/.../scripts/prometheus/up.sh'
# Or from the project dir if you're already on node-08:
#   bash scripts/prometheus/up.sh

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/mnt/shared/prom-user

# ----- 1. Create NFS-shared dirs -----
sudo mkdir -p \
    "$ROOT/config" \
    "$ROOT/data" \
    "$ROOT/grafana-data" \
    "$ROOT/grafana-provisioning"

# ----- 2. Bootstrap config (only if not already present, since verl will rewrite) -----
if [[ ! -f "$ROOT/config/prometheus.yml" ]]; then
    sudo cp "$SCRIPT_DIR/prometheus_bootstrap.yml" "$ROOT/config/prometheus.yml"
    echo "[up.sh] copied bootstrap config to $ROOT/config/prometheus.yml"
else
    echo "[up.sh] $ROOT/config/prometheus.yml already exists, keeping it"
fi

# ----- 3. Permissions -----
# prom container runs as user 'nobody' (uid 65534) — needs write to TSDB
# grafana container runs as user 'grafana' (uid 472) — needs write to grafana-data
sudo chown -R 65534:65534 "$ROOT/data"
sudo chown -R 472:472     "$ROOT/grafana-data" "$ROOT/grafana-provisioning"
# config must be readable by prom user (and writable by verl which runs as root in the verl-bench container)
sudo chmod 755 "$ROOT/config"
sudo chmod 644 "$ROOT/config/prometheus.yml"

# ----- 4. Pull + start -----
cd "$SCRIPT_DIR"
sudo docker compose pull
sudo docker compose up -d                # prometheus + grafana
# To start prometheus only: sudo docker compose up -d prometheus

# ----- 5. Quick health check -----
sleep 3
echo "=== Prometheus health ==="
curl -sf http://localhost:9090/-/healthy && echo "  ✅ /-/healthy OK" || echo "  ❌ /-/healthy failed"
curl -sf http://localhost:9090/-/ready   && echo "  ✅ /-/ready   OK" || echo "  ❌ /-/ready failed"

echo "=== Grafana health ==="
curl -sf http://localhost:3000/api/health && echo "  ✅ /api/health OK" || echo "  ❌ /api/health failed (may need ~10s warmup)"

echo
echo "=== Access (port-forward from your laptop) ==="
echo "  ssh -L 9090:127.0.0.1:9090 -L 3000:127.0.0.1:3000 node-08"
echo "  Prom UI: http://localhost:9090"
echo "  Grafana: http://localhost:3000  (admin / admin)"
