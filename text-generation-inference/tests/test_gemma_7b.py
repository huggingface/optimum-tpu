import os

import pytest
from text_generation_server.generator import TpuGeneratorSingleThread as TpuGenerator
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)
from tqdm import tqdm

from optimum.tpu.model import fetch_model


MODEL_ID = "google/gemma-7b"
SEQUENCE_LENGTH = 128


@pytest.fixture(scope="module")
def model_path():
    # Add variables to environment so they can be used in AutoModelForCausalLM
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    path = fetch_model(MODEL_ID)
    return path


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


def summary(values):
    values.sort()
    n = len(values)
    if n % 2 == 0:
        median = (values[n // 2 - 1] + values[n // 2]) / 2
    else:
        median = values[n // 2]
    total = sum(values)
    mean = sum(values) / n
    print(f"Decode time: {total}, average: {mean}, median: {median}")


@pytest.mark.slow
def test_decode_single(model_path):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 60
    # generated_text = "\n\nThe time is 1984. The place is Airstrip One, the British"
    import time

    generator = TpuGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH
    )
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    start = time.time()
    generations, next_batch = generator.prefill(batch)
    end = time.time()
    print(f"Prefill took {end - start} seconds.")
    # We already generated one token: call decode max_new_tokens - 1 times
    times = []
    for i in tqdm(range(max_new_tokens - 1)):
        step_start = time.time()
        assert next_batch.size == 1
        assert next_batch.max_tokens == 128
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
        step_end = time.time()
        cur_time = step_end - step_start
        print(f"Token {i} took {cur_time} seconds.")
        times.append(cur_time)
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    # assert output.text == generated_text
    print(output.text)
    summary(times)
