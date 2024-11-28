#!/bin/bash

# This is required by GKE, see
# https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#privileged-mode
ulimit -l 68719476736

# Hugging Face Hub related
if [[ -z "${MAX_BATCH_SIZE}" ]]; then
  MAX_BATCH_SIZE=4
fi
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"

if [[ -z "${JSON_OUTPUT_DISABLE}" ]]; then
  JSON_OUTPUT_DISABLE=--json-output
else
  JSON_OUTPUT_DISABLE=""
fi
export JSON_OUTPUT_DISABLE="${JSON_OUTPUT_DISABLE}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${MODEL_ID}"

if [[ -z "${QUANTIZATION}" ]]; then
  QUANTIZATION=""
else
  QUANTIZATION="jetstream_int8"
fi
export QUANTIZATION="${QUANTIZATION}"



exec text-generation-launcher --port 8080 \
  --max-batch-size ${MAX_BATCH_SIZE} \
  ${JSON_OUTPUT_DISABLE} \
  --model-id ${MODEL_ID}
