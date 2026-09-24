#!/bin/bash
# Build the host agent image. Run from this directory.
set -euo pipefail
cd "$(dirname "$0")"
TAG="${TAG:-spark-agent:v6}"
sudo docker build -t "$TAG" -f Dockerfile .
echo "BUILD_EXIT=$? TAG=$TAG"
