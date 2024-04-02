import os

import Levenshtein
import pytest


MODEL_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024


@pytest.fixture(scope="module")
def model_name_or_path():
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    yield MODEL_ID


@pytest.fixture(scope="module")
def tgi_service(launcher, model_name_or_path):
    with launcher(model_name_or_path) as tgi_service:
        yield tgi_service


@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(300)
    return tgi_service.client


@pytest.mark.asyncio
async def test_model_single_request(tgi_client):

    # Greedy bounded without input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert (
        response.generated_text == "\n\nDeep learning is a technique that allows you to learn something from a set of"
    )

    # Greedy bounded with input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        return_full_text=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert (
        response.generated_text
        == "What is Deep Learning?\n\nDeep learning is a technique that allows you to learn something from a set of"
    )

    # Sampling
    response = await tgi_client.generate(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=100,
        seed=42,
        decoder_input_details=True,
    )
    assert (
        'The deep neural networks that we create are essentially "miniature" neural networks that can easily be trained'
        in response.generated_text
    )


@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_client, generate_load):
    num_requests = 4
    responses = await generate_load(
        tgi_client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expected = "\n\nDeep learning is a technique that allows you to learn something from a set of"
    for r in responses:
        assert r.details.generated_tokens == 17
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert Levenshtein.distance(r.generated_text, expected) < 3
