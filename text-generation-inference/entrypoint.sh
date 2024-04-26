#!/bin/bash

# Hugging Face Hub related
if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${MODEL_ID}"

# TGI related
if [[ -n "${TGI_MAX_CONCURRENT_REQUESTS}" ]]; then
  export TGI_MAX_CONCURRENT_REQUESTS="${TGI_MAX_CONCURRENT_REQUESTS}"
else
  export TGI_MAX_CONCURRENT_REQUESTS 4
fi

if [[ -n "${TGI_MAX_BATCH_SIZE}" ]]; then
  export TGI_MAX_BATCH_SIZE="${TGI_MAX_BATCH_SIZE}"
else
  export TGI_MAX_BATCH_SIZE 1
fi

if [[ -n "${TGI_MAX_INPUT_TOKENS}" ]]; then
  export TGI_MAX_INPUT_TOKENS="${TGI_MAX_INPUT_TOKENS}"
else
  export TGI_MAX_INPUT_TOKENS 128
fi

if [[ -n "${TGI_MAX_TOTAL_TOKENS}" ]]; then
  export TGI_MAX_TOTAL_TOKENS="${TGI_MAX_TOTAL_TOKENS}"
else
  export TGI_MAX_TOTAL_TOKENS 256
fi

TGI_MAX_BATCH_PREFILL_TOKENS=$(( TGI_MAX_BATCH_SIZE*TGI_MAX_INPUT_TOKENS ))

text-generation-launcher --port 8080 \
  --max-concurrent-requests ${TGI_MAX_CONCURRENT_REQUESTS} \
  --max-batch-size ${TGI_MAX_BATCH_SIZE} \
  --max-batch-prefill-tokens ${TGI_MAX_BATCH_PREFILL_TOKENS} \
  --max-input-tokens ${TGI_MAX_INPUT_TOKENS} \
  --max-total-tokens ${TGI_MAX_TOTAL_TOKENS} \
  --model-id ${MODEL_ID}

