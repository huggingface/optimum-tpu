

from time import time

import pytest
from helpers import create_request, prepare_model
from text_generation_server.auto_generator import AutoGenerator
from text_generation_server.pb.generate_pb2 import Batch

from optimum.tpu.jetstream_pt_support import jetstream_pt_available


def test_warmup_jetstream_pytorch():
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
    model_id = "Maykeye/TinyLLama-v0"
    sequence_length = 256

    model_path = prepare_model(model_id, sequence_length)
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=2, max_sequence_length=sequence_length
    )
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    # The maximum tokens length of the model is intentionally not a power of two, to verify that prefill bucketization
    # works as expected (250 -> 256).
    max_tokens = 250
    batch = Batch(id=0, requests=[request], size=1, max_tokens=max_tokens)
    generator.warmup(batch)

    # Prepare a new request with different settings. Warmup should have triggered compilation so this can be run
    # quickly.
    input_text = "What is Deep Learning?"
    max_new_tokens = 3
    max_tokens = 13
    request1 = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=1, requests=[request1], size=1, max_tokens=max_tokens)

    start = time()
    _generations, new_batch =  generator.prefill(batch)
    _generations, new_batch =  generator.decode([new_batch])
    end = time()

    # Prefill and decode time should be less than 1 second (rather fast)
    assert end - start < 1.0
