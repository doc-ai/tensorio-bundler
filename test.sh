#!/usr/bin/env sh

# Assumes:
# You have Docker installed on your machine, as the tests run in a Docker
# container

DOCKER_TAG=${DOCKER_TAG:-docai/tensorio-bundler:test-$(date -u +%Y%m%d-%H%M)}
DOCKER_CONTEXT=$(dirname $0)

set -e

docker build -t $DOCKER_TAG $DOCKER_CONTEXT
docker run -it $DOCKER_TAG python -m unittest discover -v
