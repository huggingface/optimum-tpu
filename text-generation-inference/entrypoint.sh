#!/bin/bash

# Hugging Face Hub related
if [[ -z "${HF_MODEL_ID}" ]]; then
  echo "HF_MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
  export REVISION="${HF_MODEL_REVISION}"
fi

if [[ -n "${HF_MODEL_TRUST_REMOTE_CODE}" ]]; then
  export TRUST_REMOTE_CODE="${HF_MODEL_TRUST_REMOTE_CODE}"
fi

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
  export TGI_MAX_INPUT_LENGTH 128
fi

if [[ -n "${TGI_MAX_TOTAL_TOKENS}" ]]; then
  export TGI_MAX_TOTAL_TOKENS="${TGI_MAX_TOTAL_TOKENS}"
else
  export TGI_MAX_TOTAL_TOKENS 256
fi

text-generation-launcher --port 8080 \
  --model
  --max-concurrent-requests ${TGI_MAX_CONCURRENT_REQUESTS}
  --max-batch-size ${TGI_MAX_BATCH_SIZE}
  --max-input-tokens ${TGI_MAX_INPUT_TOKENS} \
  --max-total-tokens ${TGI_MAX_TOTAL_TOKENS}

