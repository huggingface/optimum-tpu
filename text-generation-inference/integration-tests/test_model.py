import os
import Levenshtein
import pytest

MODELS = [
    {
        "id": "openai-community/gpt2",
        "expected_greedy_output": "\n\nDeep learning is a new field of research that has been around for a while",
        "expected_sampling_output": 'A lot of researchers have tried to make a "deep learning" approach that focuses only on what is being shown',
        "expected_multiple_output": "\n\nDeep learning is a technique that allows you to learn something from a single source"
    },
    {
        "id": "google/gemma-2b-it", 
        "expected_greedy_output": "\n\nDeep learning is a subfield of machine learning that allows computers to learn from data",
        "expected_sampling_output": 'Deep learning is a subfield of machine learning that focuses on mimicking the structure and function of the human brain',
        "expected_multiple_output": "\n\nDeep learning is a subfield of machine learning that uses artificial neural networks to learn"
    }
]

SEQUENCE_LENGTH = 1024
TIMEOUT = 120

@pytest.fixture(scope="module", params=MODELS)
def model_config(request):
    return request.param

@pytest.fixture(scope="module")
def model_name_or_path(model_config):
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    yield model_config["id"]

@pytest.fixture(scope="module")
def tgi_service(launcher, model_name_or_path):
    with launcher(model_name_or_path) as tgi_service:
        yield tgi_service

@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(TIMEOUT)
    return tgi_service.client

@pytest.mark.asyncio
async def test_model_single_request(tgi_client, model_config):
    # Bounded greedy decoding without input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == model_config["expected_greedy_output"]

    # Bounded greedy decoding with input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        return_full_text=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == f"What is Deep Learning?{model_config['expected_greedy_output']}"

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
    print(f"\nGot sampling output with seed=42: {response.generated_text}")
    assert model_config["expected_sampling_output"] in response.generated_text

@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_client, generate_load, model_config):
    num_requests = 4
    responses = await generate_load(
        tgi_client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expected = model_config["expected_multiple_output"]
    for r in responses:
        assert r.details.generated_tokens == 17
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert Levenshtein.distance(r.generated_text, expected) < 3

