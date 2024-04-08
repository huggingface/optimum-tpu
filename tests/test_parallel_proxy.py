import os
from optimum.tpu.distributed_model import DistributedModel
from transformers import AutoTokenizer
import torch


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


def test_distributed_model_prefill():
    # This model will not actually shard gpt2, but it ensures model can be loaded in a parallel way and
    # that the "proxy" distributed model can be used to prefill the model.
    # NOTE: if environment variable DEBUG=1 is set, the test will be much more verbose.
    model_id = "openai-community/gpt2"
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
    decoded_texts = tokenizer.batch_decode(tokens)
    print()
    print("------------------------------------------")
    print("Decoded texts:")
    print(decoded_texts[0])
    print("------------------------------------------")
    expected_text = "Running something in parallel means that"
    assert expected_text == decoded_texts[0]
