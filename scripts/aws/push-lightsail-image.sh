#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-futures-lab}"
REGION="${AWS_REGION:-ap-northeast-1}"
IMAGE_NAME="${IMAGE_NAME:-futures-lab:aws}"
LABEL="${LIGHTSAIL_IMAGE_LABEL:-futures-lab}"

aws lightsail push-container-image \
  --region "$REGION" \
  --service-name "$SERVICE_NAME" \
  --label "$LABEL" \
  --image "$IMAGE_NAME"

echo
echo "Copy the returned image identifier, for example :$SERVICE_NAME.$LABEL.1,"
echo "then pass it to deploy-lightsail-container.sh <identifier>."
