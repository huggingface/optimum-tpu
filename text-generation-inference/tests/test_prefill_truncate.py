import pytest
from helpers import create_request, prepare_model
from text_generation_server.auto_generator import AutoGenerator
from text_generation_server.pb.generate_pb2 import Batch


@pytest.mark.jetstream
@pytest.mark.torch_xla
def test_prefill_truncate():
    model_id = "Maykeye/TinyLLama-v0"
    sequence_length = 1024

    model_path = prepare_model(model_id, sequence_length)
    max_new_tokens = 20

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=sequence_length
    )
    input_text = "And to finish the story, I will say that"

    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=sequence_length)
    generations, _ = generator.prefill(batch)
    assert len(generations) == 1
    assert generations[0].tokens.ids == [357]
    assert generations[0].tokens.texts == [" it"]
    # Now re-test but with truncate
    generator.clear()

    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    # This will only leave last tokens
    request.truncate = 3
    batch = Batch(id=0, requests=[request], size=1, max_tokens=sequence_length)
    generations, _ = generator.prefill(batch)
    assert len(generations) == 1
    assert generations[0].tokens.ids == [266]
    assert generations[0].tokens.texts == [" the"]
