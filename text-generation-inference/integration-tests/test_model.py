import os
from typing import Any, Dict

import Levenshtein
import pytest


MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "sequence_length": 1024,
        "expected_greedy_output": "\n\nDeep learning is a new field of research that has been around for a while",
        "expected_sampling_output": 'The fundamental concepts of deep learning are the same as those used to train and understand your first language, or your first set of skills',
        "expected_batch_output": "\n\nDeep learning is a technique that allows you to learn something from a single source",
        "args": [
            "--max-input-length", "512",
            "--max-total-tokens", "1024",
            "--max-batch-prefill-tokens", "512",
            "--max-batch-total-tokens", "1024"
        ],
        "env_config": {
            "MAX_BATCH_SIZE": "4",
            "JETSTREAM_PT_DISABLE": "1",
            "SKIP_WARMUP": "1",
        }
    },
    "gemma": {
        "model_id": "google/gemma-2b-it",
        "sequence_length": 1024,
        "expected_greedy_output": "\n\nDeep learning is a subfield of machine learning that allows computers to learn from data",
        "expected_sampling_output": "\n\n**Deep learning** is a subfield of machine learning that enables computers to learn from data without explicit programming",
        "expected_batch_output": "\n\nDeep learning is a subfield of machine learning that allows computers to learn from data",
        "args": [
            "--max-input-length", "512",
            "--max-total-tokens", "1024",
            "--max-batch-prefill-tokens", "512",
            "--max-batch-total-tokens", "1024"
        ],
        "env_config": {
            "MAX_BATCH_SIZE": "4",
            "SKIP_WARMUP": "1",
        }
    }
}

@pytest.fixture(scope="module", params=MODEL_CONFIGS.keys())
def model_config(request) -> Dict[str, Any]:
    """Fixture that provides model configurations for testing."""
    return MODEL_CONFIGS[request.param]

@pytest.fixture(scope="module")
def model_name_or_path(model_config):
    os.environ["HF_SEQUENCE_LENGTH"] = str(model_config["sequence_length"])
    yield model_config["model_id"]

@pytest.fixture(scope="module")
def tgi_service(launcher, model_name_or_path):
    with launcher(model_name_or_path) as tgi_service:
        yield tgi_service

@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(1000)
    return tgi_service.client

@pytest.fixture(scope="module")
def expected_outputs(model_config):
    return {
        "greedy": model_config["expected_greedy_output"],
        "sampling": model_config["expected_sampling_output"],
        "batch": model_config["expected_batch_output"]
    }

@pytest.mark.asyncio
async def test_model_single_request(tgi_client, expected_outputs):
    # Bounded greedy decoding without input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == expected_outputs["greedy"]

    # Bounded greedy decoding with input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        return_full_text=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == f"What is Deep Learning?{expected_outputs['greedy']}"

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

    assert expected_outputs["sampling"] in response.generated_text

@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_client, generate_load, expected_outputs):
    num_requests = 4
    responses = await generate_load(
        tgi_client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expected = expected_outputs["batch"]
    for r in responses:
        assert r.details.generated_tokens == 17
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert Levenshtein.distance(r.generated_text, expected) < 3
