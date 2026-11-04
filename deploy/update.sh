#!/usr/bin/env bash
# Rebuild and restart the deployment, keeping only the last two builds:
# the running image (:latest) and the one before it (:prev, for rollback).
#   deploy/update.sh            # from the repo, after `git pull`
# Roll back: docker tag sota-papers-lab-spl:prev sota-papers-lab-spl:latest
#            && docker compose up -d --no-build   (in deploy/)
set -euo pipefail
cd "$(dirname "$0")"

img=sota-papers-lab-spl
# Docker's own builder: another project may have made its docker-container
# builder the current one, and that builder may not resolve Docker Hub.
export BUILDX_BUILDER=default

if docker image inspect "$img:latest" >/dev/null 2>&1; then
  docker tag "$img:latest" "$img:prev"
fi
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose up -d --build

# Keep the last two builds only: other tags of this image, this project's
# untagged leftovers, and build cache unused for three days go.
docker images "$img" --format '{{.Repository}}:{{.Tag}}' \
  | { grep -vE ":(latest|prev)$" || true; } | xargs -r docker rmi
docker image prune -f --filter label=com.docker.compose.project=sota-papers-lab
docker builder prune -af --filter until=72h
