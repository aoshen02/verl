#!/usr/bin/env bash
# Bootstrap prom + grafana stack on tray02 (Verda ray head).
#
# Idempotent: re-running brings up the stack; existing TSDB data preserved.
#
# Run on tray02 host (NOT inside verl-bench container):
#   ssh pod4-gb300-3-tray02-f3 'bash /mnt/shared/user/scripts/prometheus/up.verda.sh'

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/mnt/shared/user/monitoring

# ----- 1. Create WEKA-shared dirs -----
sudo mkdir -p \
    "$ROOT/config" \
    "$ROOT/data" \
    "$ROOT/grafana-data" \
    "$ROOT/grafana-provisioning"

# ----- 2. Bootstrap config (only if not present; verl rewrites at training launch) -----
if [[ ! -f "$ROOT/config/prometheus.yml" ]]; then
    sudo cp "$SCRIPT_DIR/prometheus_bootstrap.yml" "$ROOT/config/prometheus.yml"
    echo "[up.verda.sh] copied bootstrap config to $ROOT/config/prometheus.yml"
else
    echo "[up.verda.sh] $ROOT/config/prometheus.yml already exists, keeping it"
fi

# ----- 3. Permissions -----
sudo chown -R 65534:65534 "$ROOT/data"
sudo chown -R 472:472     "$ROOT/grafana-data" "$ROOT/grafana-provisioning"
sudo chmod 755 "$ROOT/config"
sudo chmod 644 "$ROOT/config/prometheus.yml"

# ----- 4. Pull + start -----
cd "$SCRIPT_DIR"
sudo docker compose -f docker-compose.verda.yml pull
sudo docker compose -f docker-compose.verda.yml up -d

# ----- 5. Quick health check -----
sleep 4
echo "=== Prometheus health ==="
curl -sf http://localhost:9090/-/healthy && echo "  Prom /-/healthy OK" || echo "  Prom /-/healthy FAILED"
curl -sf http://localhost:9090/-/ready   && echo "  Prom /-/ready   OK" || echo "  Prom /-/ready   FAILED"

echo "=== Grafana health ==="
curl -sf http://localhost:3000/api/health && echo "  Grafana /api/health OK" || echo "  Grafana /api/health FAILED (may need ~10s warmup)"

echo
echo "=== Access (port-forward from your laptop) ==="
echo "  ssh -L 9090:127.0.0.1:9090 -L 3000:127.0.0.1:3000 pod4-gb300-3-tray02-f3"
echo "  Prom UI: http://localhost:9090"
echo "  Grafana: http://localhost:3000  (admin / admin)"
