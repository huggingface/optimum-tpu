#!/bin/bash

# This is required by GKE, see
# https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#privileged-mode
ulimit -l 68719476736

# Hugging Face Hub related
if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${MODEL_ID}"

text-generation-launcher --port 8080 \
  --max-batch-size 4 \
  --model-id ${MODEL_ID}

