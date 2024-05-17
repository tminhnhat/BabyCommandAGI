#!/bin/bash
set -ex

find ./workspace -mindepth 1 -maxdepth 1 -not -name '.*' -exec rm -rf {} \;
docker-compose down

./new_store.sh

echo "Clean completed"
