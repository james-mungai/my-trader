# Cloud Deployment Notes

The research container can run on any Docker host. For now, keep this as paper/recon only: no live
orders, no real keys, and no discretionary execution.

## Recommended First AWS Path

Use **AWS Lightsail Containers in Tokyo** first:

- Region: `ap-northeast-1`
- Service size: `medium`
- Scale: `1`
- Expected cost: about `$40/month`

Lightsail Containers are enough for the API/dashboard and live paper recon loop. The current local
container has been running around one vCPU burst and well under 512 MB RAM, so medium gives us a
reasonable buffer without jumping straight to EC2.

Important caveat: Lightsail Containers do not behave like a normal Docker host with a durable mounted
volume. Treat local `/app/data` inside the service as operational scratch. For serious multi-day
retention, use one of these instead:

- a Lightsail/EC2 VM with `docker-compose.aws.yml` and a persistent disk/volume
- an S3 export/sync task added later
- a database/object-store pipeline once the logging format stabilizes

## Region Shortlist

Good first regions to test:

- AWS Lightsail Tokyo: `ap-northeast-1`
- AWS Lightsail Seoul: `ap-northeast-2`
- AWS Lightsail Singapore: `ap-southeast-1`
- AWS EC2 Osaka, if we later need it: `ap-northeast-3`

Do not assume the closest city wins. Run the container in each candidate region for 15-30 minutes and
compare:

- `avg_event_lag_30s_ms`
- `max_event_lag_30s_ms`
- reconnect/handshake error counts
- stale/data-blocked periods
- decision latency

## Files Added

```text
Dockerfile.aws                                  production image, no test/dev deps
docker-compose.aws.yml                          Docker host/VM deployment with durable named volume
deployments/aws/lightsail.env.example           AWS/Lightsail environment template
deployments/aws/lightsail-containers.template.json
deployments/aws/lightsail-public-endpoint.json
scripts/aws/build-lightsail-image.ps1
scripts/aws/push-lightsail-image.ps1
scripts/aws/deploy-lightsail-container.ps1
scripts/aws/build-lightsail-image.sh
scripts/aws/push-lightsail-image.sh
scripts/aws/deploy-lightsail-container.sh
```

## Lightsail Container Workflow

Prereqs:

```bash
aws configure
aws sts get-caller-identity
```

Create the container service once. The deploy script can do this automatically, but explicit creation
is fine too:

```bash
aws lightsail create-container-service \
  --region ap-northeast-1 \
  --service-name futures-lab \
  --power medium \
  --scale 1
```

Build the AWS image:

```powershell
.\scripts\aws\build-lightsail-image.ps1
```

Or from Linux/macOS:

```bash
./scripts/aws/build-lightsail-image.sh
```

Push it to Lightsail:

```powershell
.\scripts\aws\push-lightsail-image.ps1 `
  -ServiceName futures-lab `
  -Region ap-northeast-1
```

The AWS CLI returns an image identifier that looks like:

```text
:futures-lab.futures-lab.1
```

Deploy it:

```powershell
.\scripts\aws\deploy-lightsail-container.ps1 `
  -Image ":futures-lab.futures-lab.1" `
  -ServiceName futures-lab `
  -Region ap-northeast-1 `
  -Power medium `
  -Scale 1 `
  -ApiToken "replace-with-a-long-random-token"
```

Check status:

```bash
aws lightsail get-container-services \
  --region ap-northeast-1 \
  --service-name futures-lab
```

The public endpoint will expose:

```text
/health
/
/runtime/start
/runtime/stop
/latest
/market
/decision
/paper
```

Set `API_TOKEN` before exposing the service. The dashboard and `/health` remain public, while runtime
and data endpoints require `X-API-Key`.

## Running Recon On Lightsail Containers

For the API container, start the live paper loop through the API:

```bash
curl -X POST \
  -H "X-API-Key: <token>" \
  https://<lightsail-service-url>/runtime/start
```

Stop it:

```bash
curl -X POST \
  -H "X-API-Key: <token>" \
  https://<lightsail-service-url>/runtime/stop
```

This is suitable for live paper monitoring. For one-off 3h/8h recon jobs with durable raw files,
prefer the VM workflow below until we add S3 export.

## Lightsail VM / Docker Host Workflow

Use this if we want normal Docker volumes and easier file retrieval.

```bash
git clone <repo-url> my-trader
cd my-trader
cp deployments/aws/lightsail.env.example deployments/aws/lightsail.env
docker compose -f docker-compose.aws.yml up --build -d
```

Run a one-off recon:

```bash
RUN_ID="recon_cloud_$(date -u +%Y%m%d_%H%M%S)"
docker compose -f docker-compose.aws.yml run --rm \
  -e DATA_DIR="/app/data/runs/$RUN_ID" \
  -e STRATEGY_VARIANT=stateful_momentum \
  api futures-lab record --seconds 10800 --quiet
```

Analyze:

```bash
docker compose -f docker-compose.aws.yml run --rm \
  -e DATA_DIR="/app/data/runs/$RUN_ID" \
  -e STRATEGY_VARIANT=stateful_momentum \
  api futures-lab replay --flatten-at-end

docker compose -f docker-compose.aws.yml run --rm \
  -e DATA_DIR="/app/data/runs/$RUN_ID" \
  api futures-lab data-summary --largest 8

docker compose -f docker-compose.aws.yml run --rm \
  -e DATA_DIR="/app/data/runs/$RUN_ID" \
  api futures-lab compress-raw --all
```

## Latency Checks

During a run, inspect recent compact features/decisions:

```bash
docker compose -f docker-compose.aws.yml run --rm api sh -c \
  "tail -n 5 /app/data/features/*_features_*.jsonl"
```

Useful fields:

- `exchange_event_lag_ms`
- `avg_event_lag_30s_ms`
- `max_event_lag_30s_ms`
- `data_age_seconds`
- `decision_latency_ms`
- `stateful_momentum_filter`

## Safety

Keep cloud deployment as paper/demo until replay and demo results justify the next step. Live
execution must stay behind explicit user permission, deterministic strategy rules, risk checks, and
careful API key handling.
