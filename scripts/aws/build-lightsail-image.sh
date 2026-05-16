#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-futures-lab:aws}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build -f "$REPO_ROOT/Dockerfile.aws" -t "$IMAGE_NAME" "$REPO_ROOT"
echo "Built $IMAGE_NAME"
