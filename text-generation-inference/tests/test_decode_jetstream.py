
import pytest
from decode_tests_utils import DecodeTestParams, decode_single_test


# All tests in this file are for jetstream
pytestmark = pytest.mark.jetstream

@pytest.mark.slow
@pytest.mark.parametrize("do_sample", [False, True], ids=["greedy", "sample"])
@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="meta-llama/Llama-2-7b-hf",
            sequence_length=256,
            expected_text="\nWinston Smith, his chin nuzzled into his breast in an effort to escape",
            top_k=100,
        ),
        DecodeTestParams(
            model_id="meta-llama/Meta-Llama-3-8B",
            sequence_length=256,
            expected_text=" Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,",
            top_k=100,
        ),
        DecodeTestParams(
            model_id="google/gemma-7b",
            sequence_length=128,
            expected_text="\n\nThe year was 1984.\n\nThe place was Oceania.\n\nThe man was",
        ),
        DecodeTestParams(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            sequence_length=1024,
            expected_text="\n\nGeorge Orwell, 1984\n\nThe clocks are striking thirteen",
        ),
    ],
    ids=["Llama-2-7b-hf", "Meta-Llama-3-8B", "gemma-7b", "Mixtral-8x7B"],
)
def test_decode_single_jetstream_pytorch_slow(params, do_sample):
    params.do_sample = do_sample
    decode_single_test(params)


@pytest.mark.parametrize("do_sample", [False, True], ids=["greedy", "sample"])
@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="Maykeye/TinyLLama-v0",
            sequence_length=256,
            expected_text=" The sun was shining and the sky was shining.\nSuddenly, a big wind came and blew the wind away.",
            max_new_tokens=25,
        ),
        DecodeTestParams(
            model_id="google/gemma-2b",
            sequence_length=1024,
            expected_text="\n\nThe first thing I noticed was the smell of the rain. It was a smell I had never",
        ),
        DecodeTestParams(
            model_id="dacorvo/Mixtral-tiny", # This is a random tiny model, just to test model can be loaded.
            sequence_length=512,
            expected_text="манaminationVariableßer Rog malesazine longふ Toy Champions enero Facereverse▲verbose prosecut literally disappearedअ",
        ),
        DecodeTestParams(
            # NOTE: this test is interesting because it is a fine-tuned model that requires padding on weights to work.
            model_id="Trendyol/Trendyol-LLM-7b-base-v0.1",
            sequence_length=512,
            expected_text="\nThe clocks were striking thirteen, and the clocks were striking thirteen.",
        ),
        DecodeTestParams(
            model_id="meta-llama/Llama-3.2-1B",
            sequence_length=256,
            expected_text=" Winston Smith, his chin nuzzled into his breast, stretched, and looked out across the city",
            max_new_tokens=20,
        )
    ],
    ids=["TinyLLama-v0", "gemma-2b", "Mixtral-tiny", "Trendyol-LLM-7b-base-v0.1", "Llama-3.2-1B"],
)
def test_decode_single_jetstream_pytorch(params, do_sample):
    params.do_sample = do_sample
    decode_single_test(params)


def test_decode_repetition_penalty_jetstream_pytorch():
    """Test if the repetition penalty generates something without crashing."""
    params = DecodeTestParams(
            model_id="Maykeye/TinyLLama-v0",
            sequence_length=256,
            expected_text=" The sun was shining and it was very hot.\nSuddenly, a big wind came and",
            repetition_penalty=1.2
        )
    decode_single_test(params)
