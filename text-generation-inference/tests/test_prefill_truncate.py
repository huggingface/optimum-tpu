from helpers import create_request, prepare_model
from text_generation_server.generator import TpuGeneratorSingleThread as TpuGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_prefill_truncate():
    model_id="Maykeye/TinyLLama-v0"
    sequence_length=1024

    model_path = prepare_model(model_id, sequence_length)
    max_new_tokens = 20

    generator = TpuGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=sequence_length
    )
    input_text = "This is a secret part. Once upon a time,"

    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=sequence_length)
    generations, _ = generator.prefill(batch)
    assert len(generations) == 1
    assert generations[0].tokens.ids == [635]
    assert generations[0].tokens.texts == [" there"]

    # Now re-test but with truncate
    generator.clear()

    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    # This will only leave 5 tokens
    request.truncate = 5
    batch = Batch(id=0, requests=[request], size=1, max_tokens=sequence_length)
    generations, _ = generator.prefill(batch)
    assert len(generations) == 1
    assert generations[0].tokens.ids == [260]
    assert generations[0].tokens.texts == [" a"]
