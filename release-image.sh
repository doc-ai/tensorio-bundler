#!/usr/bin/env sh

# Builds docker image and pushes it to Docker Hub
# Assumes that invoker has authenticated local docker to push to docai/tensorio-bundler

set -e

DOCKER_REPO=${DOCKER_REPO:-docai/tensorio-bundler}
PROJECT_ROOT=$(dirname $0)
GIT_SHORT_SHA=$(cd $PROJECT_ROOT && git rev-parse --short HEAD)
DATE_NOW=$(date -u +%Y%m%d-%H%M)

docker build -t ${DOCKER_REPO}:latest ${PROJECT_ROOT}
docker tag ${DOCKER_REPO}:latest ${DOCKER_REPO}:$GIT_SHORT_SHA
docker tag ${DOCKER_REPO}:latest ${DOCKER_REPO}:$DATE_NOW

docker push ${DOCKER_REPO}:latest
docker push ${DOCKER_REPO}:$GIT_SHORT_SHA
docker push ${DOCKER_REPO}:$DATE_NOW
