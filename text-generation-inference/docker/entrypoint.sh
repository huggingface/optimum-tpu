#!/bin/bash

# Hugging Face Hub related
if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${MODEL_ID}"

text-generation-launcher --port 8080 \
  --max-batch-size 4 \
  --model-id ${MODEL_ID}

