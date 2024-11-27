import pytest
from helpers import create_request, prepare_model
from text_generation_server.auto_generator import AutoGenerator
from text_generation_server.pb.generate_pb2 import Batch
from tqdm import tqdm


MODEL_ID = "Maykeye/TinyLLama-v0"
SEQUENCE_LENGTH = 256


@pytest.fixture(scope="module")
def model_path():
    return prepare_model(MODEL_ID, SEQUENCE_LENGTH)


def _test_info(model_path, expected_device_type):
    """Verify the model info is correctly loaded and check expected results."""
    generator = AutoGenerator.from_pretrained(model_path, revision="", max_batch_size=1, max_sequence_length=1)
    info = generator.info
    assert info.requires_padding is True
    assert info.device_type == expected_device_type
    assert info.window_size == 0
    assert info.speculate == 0


@pytest.mark.jetstream
def test_jetstream_info(model_path):
    _test_info(model_path, "meta")


@pytest.mark.torch_xla
def test_info(model_path):
    _test_info(model_path, "xla")


def _test_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path):
    """Verify that prefilling a batch with a single request with different sampling techniques."""
    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=batch_size, max_sequence_length=SEQUENCE_LENGTH
    )
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
        if do_sample:
            assert tokens.ids != [token_id]
            assert tokens.texts != [token_text]
        else:
            assert tokens.ids == [token_id]
            assert tokens.texts == [token_text]


@pytest.mark.jetstream
@pytest.mark.parametrize("do_sample", [False, True], ids=["greedy", "sample"])
@pytest.mark.parametrize("batch_size", [1, 4], ids=["single", "multiple"])
def test_jetstream_prefill(do_sample, batch_size, model_path):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    token_id = 347
    token_text = " The"
    _test_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path)


@pytest.mark.torch_xla
@pytest.mark.parametrize("do_sample", [False, True], ids=["greedy", "sample"])
@pytest.mark.parametrize("batch_size", [1, 4], ids=["single", "multiple"])
def test_prefill(do_sample, batch_size, model_path):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    token_id = 571
    token_text = " It"
    _test_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path)


def _test_prefill_change_sampling(
    model_path,
    greedy_expected_token_id,
):
    """Verify changing the sampling strategy between requests in the same batch works as expected."""
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    batch_size = 1

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=batch_size, max_sequence_length=SEQUENCE_LENGTH
    )
    max_new_tokens = 20

    def check_request(do_sample, expected_token_id):
        requests = [create_request(id=0, inputs=input_text, do_sample=do_sample, max_new_tokens=max_new_tokens)]
        batch = Batch(id=0, requests=requests, size=batch_size, max_tokens=batch_size * SEQUENCE_LENGTH)
        generations, _ = generator.prefill(batch)
        tokens = generations[0].tokens
        if do_sample:
            assert tokens.ids != [expected_token_id]
        else:
            assert tokens.ids == [expected_token_id]
        generator.clear()

    # First request is greedy
    check_request(False, greedy_expected_token_id)
    # Second request is sampling
    check_request(True, greedy_expected_token_id)
    # Third request is greedy again
    check_request(False, greedy_expected_token_id)


@pytest.mark.jetstream
def test_jetstream_prefill_change_sampling(model_path):
    _test_prefill_change_sampling(model_path, 347)


@pytest.mark.torch_xla
def test_prefill_change_sampling(model_path):
    _test_prefill_change_sampling(model_path, 571)


def _test_continuous_batching_two_requests(model_path):
    """Verify that two requests added to the batch at different generation steps
    generate the same outputs (continuous batching).
    """
    generator = AutoGenerator.from_pretrained(
        model_path,
        revision="",
        max_batch_size=2,
        max_sequence_length=SEQUENCE_LENGTH,
    )
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


@pytest.mark.jetstream
def test_jetstream_decode_multiple(model_path):
    _test_continuous_batching_two_requests(model_path)


"""NOTE: This test does not work on PyTorch/XLA, because of the way
calculations are done in torch/xla and the effect of KV cache (they produce
similar outputs, but not identical).
"""
@pytest.mark.skip(reason="Test is not supported on PyTorch/XLA")
@pytest.mark.torch_xla
def test_decode_multiple(model_path):
    _test_continuous_batching_two_requests(model_path)
