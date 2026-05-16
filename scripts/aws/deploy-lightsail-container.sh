#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <lightsail-image-id> [api-token]" >&2
  exit 2
fi

IMAGE="$1"
API_TOKEN="${2:-change-me-before-deploy}"
SERVICE_NAME="${SERVICE_NAME:-futures-lab}"
REGION="${AWS_REGION:-ap-northeast-1}"
POWER="${LIGHTSAIL_POWER:-medium}"
SCALE="${LIGHTSAIL_SCALE:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RENDERED="$(mktemp)"

if ! aws lightsail get-container-services \
  --region "$REGION" \
  --service-name "$SERVICE_NAME" \
  --query "containerServices[0].serviceName" \
  --output text >/dev/null 2>&1; then
  aws lightsail create-container-service \
    --region "$REGION" \
    --service-name "$SERVICE_NAME" \
    --power "$POWER" \
    --scale "$SCALE" >/dev/null
  echo "Created Lightsail container service $SERVICE_NAME ($POWER x $SCALE)."
fi

sed \
  -e "s#__LIGHTSAIL_IMAGE__#$IMAGE#g" \
  -e "s#__API_TOKEN__#$API_TOKEN#g" \
  "$REPO_ROOT/deployments/aws/lightsail-containers.template.json" > "$RENDERED"

aws lightsail create-container-service-deployment \
  --region "$REGION" \
  --service-name "$SERVICE_NAME" \
  --containers "file://$RENDERED" \
  --public-endpoint "file://$REPO_ROOT/deployments/aws/lightsail-public-endpoint.json"

echo
echo "Deployment requested. Check status with:"
echo "aws lightsail get-container-services --region $REGION --service-name $SERVICE_NAME"
