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


MODEL_ID = "google/gemma-2b"
SEQUENCE_LENGTH = 1024


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


def test_decode_single(model_path):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20
    generated_text = "\n\nThe first thing I noticed was the smell of the rain. It was a smell I had never"

    generator = TpuGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH
    )
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in tqdm(range(max_new_tokens - 1)):
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
