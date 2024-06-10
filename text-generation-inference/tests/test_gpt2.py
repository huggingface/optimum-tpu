import os

import pytest
from text_generation_server.generator import TpuGenerator
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)
from tqdm import tqdm

from optimum.tpu.model import fetch_model


MODEL_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024


@pytest.fixture(scope="module")
def model_path():
    # Add variables to environment so they can be used in AutoModelForCausalLM
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    path = fetch_model(MODEL_ID)
    return path


def test_info(model_path):
    generator = TpuGenerator.from_pretrained(model_path, revision="", max_batch_size=1, max_sequence_length=1)
    info = generator.info
    assert info.requires_padding is True
    assert info.device_type == "xla"
    assert info.window_size == 0
    assert info.speculate == 0


def create_request(
    id: int,
    inputs: str,
    max_new_tokens=20,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int = 0,
    repetition_penalty: float = 1.0,
):
    # For these tests we can safely set typical_p to 1.0 (default)
    typical_p = 1.0
    if not do_sample:
        # Drop top_p parameter to avoid warnings
        top_p = 1.0
    parameters = NextTokenChooserParameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
        repetition_penalty=repetition_penalty,
        typical_p=typical_p,
    )
    stopping_parameters = StoppingCriteriaParameters(max_new_tokens=max_new_tokens)
    return Request(id=id, inputs=inputs, parameters=parameters, stopping_parameters=stopping_parameters)


@pytest.mark.parametrize(
    "input_text, token_id, token_text, do_sample",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            383,
            " The",
            False,
        ],
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            775,
            " We",
            True,
        ],
    ],
    ids=["greedy", "sample"],
)
@pytest.mark.parametrize("batch_size", [1, 4], ids=["single", "multiple"])
def test_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path):
    generator = TpuGenerator.from_pretrained(model_path, revision="", max_batch_size=batch_size, max_sequence_length=SEQUENCE_LENGTH)
    requests = []
    max_new_tokens = 20
    for i in range(batch_size):
        requests.append(create_request(id=0, inputs=input_text, do_sample=do_sample, max_new_tokens=max_new_tokens))
    # Let's be pessimistic when estimating max_tokens
    batch_size * (len(input_text) + max_new_tokens)
    batch = Batch(id=0, requests=requests, size=batch_size, max_tokens=batch_size * SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == batch_size
    # Whatever was passed as max_tokens, the server will correct it
    # because of static batching
    assert next_batch.max_tokens == batch_size * SEQUENCE_LENGTH
    assert len(generations) == batch_size
    for g in generations:
        tokens = g.tokens
        assert tokens.ids == [token_id]
        assert tokens.texts == [token_text]


@pytest.mark.parametrize(
    "input_text, max_new_tokens, generated_text, do_sample",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            " The sun was setting, and the wind was blowing. The sun was setting, and the wind was",
            False,
        ],
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            " We sat at the front door and watched the clock on a box of yellow paper and found it almost",
            True,
        ],
    ],
    ids=["greedy", "sample"],
)
def test_decode_single(input_text, max_new_tokens, generated_text, do_sample, model_path):
    generator = TpuGenerator.from_pretrained(model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH)
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=do_sample)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in tqdm(range(max_new_tokens - 1), "Decoding tokens"):
        assert next_batch.size == 1
        assert next_batch.max_tokens == 1024
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    assert output.text == generated_text


def test_decode_multiple(model_path):
    generator = TpuGenerator.from_pretrained(model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH)
    input_text = "Once upon a time"
    max_new_tokens = 20
    # Prefill a single request, remembering the generated token
    tokens = {0: [], 1: []}
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == 1
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == 1
    # Decode a few tokens
    gen_tokens = 4
    for _ in tqdm(range(gen_tokens - 1), "Decoding tokens"):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert next_batch.size == 1
    # Add a second request
    request = create_request(id=1, inputs=input_text, max_new_tokens=max_new_tokens)
    batch = Batch(id=1, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch_1 = generator.prefill(batch)
    assert next_batch_1.size == 1
    # We should have generated only a single token
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert len(tokens[1]) == 1
    # Decode more tokens until we reach the maximum for the first request
    batches = [next_batch, next_batch_1]
    for _ in tqdm(range(max_new_tokens - gen_tokens), "Decoding tokens (2nd batch)"):
        generations, next_batch = generator.decode(batches)
        for g in generations:
            tokens[g.request_id].append(g.tokens.ids[0])
        batches = [next_batch]
    # Verify we now only have one pending request
    assert next_batch.size == 1
    assert len(tokens[0]) == max_new_tokens
    assert len(tokens[1]) == max_new_tokens - gen_tokens + 1
    # Verify we have the output for the first request
    for g in generations:
        if g.request_id == 0:
            output = g.generated_text
            assert output.text != ""
            assert output.generated_tokens == max_new_tokens
            generated_text = output.text
    # Continue decoding until the end of the second request
    for _ in tqdm(range(gen_tokens - 1), "Decoding tokens (finishing)"):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert next_batch is None
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert tokens[0] == tokens[1]
    assert output.text == generated_text
