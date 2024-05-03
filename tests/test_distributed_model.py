import os

import pytest
import torch
from transformers import AutoTokenizer

from optimum.tpu.distributed_model import DistributedModel


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


def _test_distributed_model_prefill(model_id):
    # This test ensures model can be loaded in a parallel way and
    # that the "proxy" distributed model can be used to prefill the model.
    # Disable tokenizers parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = ["Running something in parallel means"]
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    tokens = input_ids.clone()

    model = DistributedModel(model_id, sample_greedy)
    next_tokens = model.prefill(**inputs, position_ids=pos_ids)
    tokens = torch.cat([tokens, next_tokens], dim=-1)

    # Data can be decoded even before leaving
    decoded_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    print()
    print("------------------------------------------")
    print("Decoded texts:")
    print(decoded_texts[0])
    print("------------------------------------------")
    # Even if models are different, for this simple test results are the same.
    expected_text = "Running something in parallel means that"
    assert expected_text == decoded_texts[0]


def test_distributed_model_prefill_gpt2():
    _test_distributed_model_prefill("openai-community/gpt2")


@pytest.mark.slow
def test_distributed_model_prefill_gemma7b():
    _test_distributed_model_prefill("google/gemma-7b")

@pytest.mark.slow
def test_distributed_model_prefill_llama3_8b():
    _test_distributed_model_prefill("meta-llama/Meta-Llama-3-8B")
