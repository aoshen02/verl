# Prometheus + Grafana for verl 35B RL monitoring

Independent Docker stack, lives on **node-08** (current ray head).
Config + TSDB on NFS `/mnt/shared/prom-user/` so verl can auto-write
scrape targets from any ray node.

## Architecture

```
node-08 host
├── verl-bench-user     (training, ray head)
└── prometheus-user     (this stack, --network=host:9090)
    └── grafana-user    (optional, --network=host:3000)

NFS /mnt/shared/prom-user/
├── config/prometheus.yml      ← verl auto-writes on every train launch
├── data/                      ← TSDB persistent
├── grafana-data/              ← grafana state (dashboards/users)
└── grafana-provisioning/      ← future: dashboard JSON / datasource yml
```

Why node-08 and not node-01:
* verl reloads prom via `curl http://<own_ip>:9090/-/reload` on each ray node.
  node-08 is in the ray cluster, so its reload task hits this stack. node-01
  is not, so prom there wouldn't get auto-reloaded.

## Setup (one-time)

```bash
# On node-08 host (NOT inside verl-bench container):
ssh node-08 'bash /home/user/setup_new_cluster/vllm/projects/semianalysis-swebench-solo/scripts/prometheus/up.sh'
```

`up.sh` is idempotent. Re-run after upgrades / config changes.

## Verify

```bash
curl http://10.0.2.108:9090/-/healthy   # prom alive
curl http://10.0.2.108:3000/api/health  # grafana alive
```

## Enable verl integration

In `scripts/train_qwen3p5_35b.sh`, append these 5 hydra keys to the ray job
submit command (or set them via env if the script gets the `ENABLE_PROMETHEUS_MONITORING`
gate, like `reference.sh`):

```bash
actor_rollout_ref.rollout.disable_log_stats=False \
+actor_rollout_ref.rollout.prometheus.enable=True \
+actor_rollout_ref.rollout.prometheus.port=9090 \
+actor_rollout_ref.rollout.prometheus.file=/mnt/shared/prom-user/config/prometheus.yml \
+actor_rollout_ref.rollout.prometheus.served_model_name=qwen3p5_35b_a3b \
```

Notes:
* `disable_log_stats=False` is required (verl asserts this). We already have it.
* `prometheus.file` MUST be a host-accessible path that this prom container
  reads. `/mnt/shared/prom-user/config/prometheus.yml` matches the mount in
  `docker-compose.yml`.
* `served_model_name` is just a label shown in Grafana — keeps long model
  paths out of dashboards.

What happens at training launch:
1. verl's `AgentLoopManager` starts N vLLM HTTP servers across rollout nodes,
   collects their `ip:port` addresses.
2. verl writes `prometheus.yml` with `{"job_name": "rollout", "static_configs":
   [{"targets": [...]}]}` to the NFS path.
3. verl schedules a `reload_prometheus` Ray task on every alive ray node.
   Each task curls `http://<own_node_ip>:9090/-/reload`. Only the one on
   node-08 hits this stack — others silently fail (per verl's intended
   design, see `prometheus_utils.py:60-78` comment).
4. Prom hot-reloads, starts scraping all vLLM `/metrics` endpoints.

## Access

```bash
# Port-forward from your laptop:
ssh -L 9090:127.0.0.1:9090 -L 3000:127.0.0.1:3000 node-08

# Then in browser:
#   http://localhost:9090     Prom UI (query, target health, etc.)
#   http://localhost:3000     Grafana (default admin/admin, change it)
```

## Useful Prom queries

After enabling, in Prom UI → Graph tab try:

```
# vLLM request rate per server
rate(vllm:request_success_total[1m])

# KV cache utilization
vllm:gpu_cache_usage_perc

# Time-to-first-token p99
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# Inter-token latency (decode)
rate(vllm:time_per_output_token_seconds_sum[1m]) / rate(vllm:time_per_output_token_seconds_count[1m])

# Number of running + waiting sequences
vllm:num_requests_running
vllm:num_requests_waiting
```

Full vLLM metric list: `curl http://<vllm-ip>:<port>/metrics | grep "^# HELP"`

## Known limitations

* **Ray native metrics not scraped**. verl writes a `ray` job entry with
  `file_sd_configs: [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}]`.
  That path is inside each verl-bench container, not visible to this prom.
  Result: ray scrape fails, vLLM metrics work fine. Fix later if needed
  (mount /tmp/ray from each verl-bench → prom, or write a sidecar to
  aggregate Ray's metrics-export-port across nodes).
* **TSDB on NFS lustre**. Performance is fine for our scrape volume (10s
  interval × ~10 targets × ~100 metrics = <100KB/s). Move to local disk on
  node-08 if it ever becomes a bottleneck.
* **prom container ties to node-08 host**. If the ray head changes, move
  this stack or update verl's reload mechanism. Current head has been
  node-08 for the entire 35B work.

## Stop / clean

```bash
# Stop containers (TSDB preserved):
cd scripts/prometheus && sudo docker compose down

# Wipe everything (TSDB lost):
sudo docker compose down -v
sudo rm -rf /mnt/shared/prom-user/data /mnt/shared/prom-user/grafana-data
```
