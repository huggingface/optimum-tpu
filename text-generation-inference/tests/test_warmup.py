

import pytest
from helpers import create_request, prepare_model
from text_generation_server.auto_generator import AutoGenerator
from text_generation_server.pb.generate_pb2 import Batch

from optimum.tpu.jetstream_pt_support import jetstream_pt_available


def test_warmup_jetstream_pytorch():
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
    model_id = "Maykeye/TinyLLama-v0"

    # The maximum sequence length of the model is set to 1000, but warmup will round that up to the next power of two
    # in prefill (1024).
    sequence_length = 1000

    model_path = prepare_model(model_id, sequence_length)
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=sequence_length
    )
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=sequence_length)
    generator.warmup(batch)

