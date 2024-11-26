# import os
# import Levenshtein
# import pytest

# MODEL_ID = "google/gemma-2b-it"
# # MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# SEQUENCE_LENGTH = 1024

# @pytest.fixture(scope="module")
# def model_name_or_path():
#     os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
#     yield MODEL_ID


# @pytest.fixture(scope="module")
# def tgi_service(launcher, model_name_or_path):
#     with launcher(model_name_or_path) as tgi_service:
#         yield tgi_service


# @pytest.fixture(scope="module")
# async def tgi_client(tgi_service):
#     await tgi_service.health(1000)
#     # time.sleep(120)
#     # raise Exception("Stop here")
#     return tgi_service.client

# @pytest.mark.asyncio
# async def test_model_single_request(tgi_client):

#     # Bounded greedy decoding without input
#     response = await tgi_client.generate(
#         "What is Deep Learning?",
#         max_new_tokens=17,
#         decoder_input_details=True,
#         # best_of=1,
#         # do_sample=False,
#     )
#     assert response.details.generated_tokens == 17
#     # assert (
#     #     response.generated_text == "\n\nDeep learning is a technique that allows you to learn something from a set of"
#     # )
#     assert (
#         response.generated_text == "\n\nDeep learning is a subfield of machine learning that allows computers to learn from data"
#     )

#     # Bounded greedy decoding with input
#     response = await tgi_client.generate(
#         "What is Deep Learning?",
#         max_new_tokens=17,
#         return_full_text=True,
#         decoder_input_details=True,
#         # best_of=1,
#         # do_sample=False,
#     )
#     assert response.details.generated_tokens == 17
#     assert (
#         response.generated_text
#         == "What is Deep Learning?\n\nDeep learning is a subfield of machine learning that allows computers to learn from data"
#     )

#     # Sampling
#     response = await tgi_client.generate(
#         "What is Deep Learning?",
#         do_sample=True,
#         top_k=50,
#         top_p=0.9,
#         repetition_penalty=1.2,
#         max_new_tokens=100,
#         seed=42,
#         decoder_input_details=True,
#     )
#     print(f"\nGot sampling output with seed=42: {response.generated_text}")

#     assert (
#         'Deep learning is a subfield of machine learning that focuses on mimicking the structure and function of the human brain'
#         in response.generated_text
#     )


# @pytest.mark.asyncio
# async def test_model_multiple_requests(tgi_client, generate_load):
#     num_requests = 4
#     responses = await generate_load(
#         tgi_client,
#         "What is Deep Learning?",
#         max_new_tokens=17,
#         n=num_requests,
#         # do_sample=False,  # Explicitly use greedy decoding
#         # best_of=1,  # Match the single request settings
#     )

#     assert len(responses) == 4
#     expected = "\n\nDeep learning is a subfield of machine learning that uses artificial neural networks to learn"
#     for r in responses:
#         assert r.details.generated_tokens == 17
#         # Compute the similarity with the expectation using the levenshtein distance
#         # We should not have more than two substitutions or additions
#         assert Levenshtein.distance(r.generated_text, expected) < 3