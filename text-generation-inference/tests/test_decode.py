
import pytest
from decode_tests_utils import DecodeTestParams, decode_single_test


@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="google/gemma-2b",
            sequence_length=1024,
            expected_text="\n\nThe first thing I noticed was the smell of the rain. It was a smell I had never",
        ),
        DecodeTestParams(
            model_id="Maykeye/TinyLLama-v0",
            sequence_length=1024,
            expected_text=" It was a very special day, and it was a very special day.\nThe mommy said",
        ),
    ],
    ids=["gemma-2b", "TinyLLama-v0"],
)
def test_decode_single(params):
    decode_single_test(params)

@pytest.mark.slow
@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="meta-llama/Meta-Llama-3-8B",
            sequence_length=256,
            expected_text=" Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,",
        ),
        DecodeTestParams(
            model_id="google/gemma-7b",
            sequence_length=128,
            expected_text="\n\nThe first line of George Orwell’s <em>1984</em> is a perfect example",
        ),
        DecodeTestParams(
            model_id="mistralai/Mistral-7B-v0.3",
            sequence_length=128,
            expected_text=" Winston Smith, his chin nuzzled into his breast in an effort to escape the v",
        ),
    ],
    ids=["Meta-Llama-3-8B", "gemma-7b", "Mistral-7B-v0.3"],
)
def test_decode_single_slow(params):
    decode_single_test(params)
