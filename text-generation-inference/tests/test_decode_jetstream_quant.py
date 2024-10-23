
import pytest
from decode_tests_utils import DecodeTestParams, decode_single_test

from optimum.tpu.jetstream_pt_support import jetstream_pt_available


@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="google/gemma-2b",
            sequence_length=1024,
            expected_text="\n\nThe first thing I noticed was the smell of the rain. It was a very heavy rain,",
        ),
        DecodeTestParams(
            model_id="Maykeye/TinyLLama-v0",
            sequence_length=256,
            expected_text=" It was a very special day, and it was a very special day.\nThe mommy said to her, \"Let",
            max_new_tokens=25,
        ),
    ],
    ids=["gemma-2b", "TinyLLama-v0"],
)
def test_decode_jetstream_quantization(quantization_jetstream_int8, params):
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
    decode_single_test(params)


@pytest.mark.slow
@pytest.mark.parametrize("params",
    [
        DecodeTestParams(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            sequence_length=1024,
            expected_text="\n\nGeorge Orwell, 1984\n\nThe clocks are striking thirteen",
        ),
        DecodeTestParams(
            model_id="meta-llama/Meta-Llama-3-8B",
            sequence_length=256,
            expected_text=" Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,",
        ),
        DecodeTestParams(
            model_id="meta-llama/Meta-Llama-3-70B",
            sequence_length=512,
            expected_text=" Winston Smith,s,s,s,s,s,s,s,s,s,s",
        ),
    ],
    ids=["Mixtral-8x7B", "Meta-Llama-3-8B" ,"Meta-Llama-3-70B"],
)
def test_decode_jetstream_quantization_slow(quantization_jetstream_int8, params):
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
    decode_single_test(params)
