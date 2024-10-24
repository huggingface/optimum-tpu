from dataclasses import dataclass

from helpers import create_request, prepare_model
from text_generation_server.auto_generator import AutoGenerator
from text_generation_server.pb.generate_pb2 import Batch
from tqdm import tqdm


@dataclass
class DecodeTestParams:
    model_id: str
    sequence_length: int
    expected_text: str
    do_sample: bool = False
    max_new_tokens: int = 20
    top_k: int = 50


def decode_single_test(params):
    model_path = prepare_model(params.model_id, params.sequence_length)
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = params.max_new_tokens

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=params.sequence_length
    )
    request = create_request(
        id=0,
        inputs=input_text,
        max_new_tokens=max_new_tokens,
        do_sample=params.do_sample,
        top_k=params.top_k,
    )
    batch = Batch(id=0, requests=[request], size=1, max_tokens=params.sequence_length)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in tqdm(range(max_new_tokens - 1)):
        assert next_batch.size == 1
        assert next_batch.max_tokens == params.sequence_length
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    # Destroy generator: this will properly stop threads and prevent them from getting stuck if one of the following
    # assertions fails.
    del generator
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    print(f"Generated text: {output.text}")
    if params.do_sample:
        assert output.text != params.expected_text
    else:
        assert output.text == params.expected_text
