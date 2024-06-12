#!/usr/bin/python

import argparse
import datetime
import os
import platform
import time
from typing import List

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, StaticCache

from optimum.tpu.modeling import AutoModelForCausalLM


os.environ["PJRT_DEVICE"] = "TPU"


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        return_dict=False,
        use_cache=True,
        past_key_values=past_key_values,
    )[0]
    new_token = sample_greedy(logits)
    return new_token


def conditional_compile(func):
    if "DBG_COMPILE" in os.environ:
        compiled = torch.compile(func, backend="openxla")
        return compiled
    return func


def summary(values: List[float]):
    values.sort()
    n = len(values)
    if n % 2 == 0:
        median = (values[n // 2 - 1] + values[n // 2]) / 2
    else:
        median = values[n // 2]
    total = sum(values)
    mean = sum(values) / n
    print(f"Decode time: {total}, average: {mean}, median: {median}")


def main():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument("--model_id", type=str,
                        default="google/gemma-2b",
                        help="Model ID (e.g.: google/gemma-2b, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--max_cache_length", type=int, default=256, help="Maximum cache length for the model")
    args = parser.parse_args()

    prg_start = time.time()
    print(f"⏳ Loading model {args.model_id}...")
    model_id = args.model_id
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    device = model.device
    model = model.eval()
    print(f"✅ Model loaded in {time.time() - prg_start} seconds.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set pad token for cases where it is None, e.g. for Mistral
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Here's a funny thing:", "Once upon a time,"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    batch_size, sequence_length = inputs["input_ids"].shape
    max_cache_length = 1024
    max_new_tokens = args.max_new_tokens

    # setup static cache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_length,
        device=model.device,
        dtype=model.dtype,
    )
    start = time.time()
    cache_position = torch.arange(sequence_length, device=device)
    generated_ids = torch.zeros(
        (batch_size, sequence_length + max_new_tokens + 1),
        dtype=torch.int,
        device=device,
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch.int)

    # prefill here
    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    logits = model(
        **inputs,
        cache_position=cache_position,
        return_dict=False,
        use_cache=True,
        position_ids=pos_ids,
        past_key_values=past_key_values,
    )[0]
    next_token = sample_greedy(logits)
    xm.mark_step()
    generated_ids[:, sequence_length] = next_token[:, 0]
    end = time.time()
    print(f"Prefill took {end - start} seconds.")

    pos_ids = pos_ids.max(axis=-1)[0].unsqueeze(1) + 1

    model = conditional_compile(model)
    cache_position = torch.tensor([sequence_length], device=device)
    decode_times = []
    for i in range(max_new_tokens):
        step_start = time.time()
        next_token = decode_one_tokens(model, next_token.clone(), pos_ids, cache_position, past_key_values)
        cache_position += 1
        generated_ids[:, cache_position] = next_token
        pos_ids += 1
        xm.mark_step()
        step_end = time.time()
        step_time = step_end - step_start
        decode_times.append(step_time)
        print(f"Step {i} took {step_time} seconds.")
    summary(decode_times)

    print(f"Decoding start at {datetime.datetime.now()}")

    decoded_texts = tokenizer.batch_decode(generated_ids)
    for i, text in enumerate(decoded_texts):
        print(i, text)

    end = time.time()
    print(f"Program run in {end - prg_start} seconds. Device: {device} System: {platform.system()}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
