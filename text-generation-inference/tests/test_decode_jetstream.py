
import pytest
from decode_tests_utils import DecodeTestParams, decode_single_test

from optimum.tpu.jetstream_pt_support import jetstream_pt_available


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
            expected_text="\n\nThe time is 1984. The place is Airstrip One, the British",
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
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
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
    ],
    ids=["TinyLLama-v0", "gemma-2b", "Mixtral-tiny"],
)
def test_decode_single_jetstream_pytorch(params, do_sample):
    if not jetstream_pt_available():
        pytest.skip("Jetstream PyTorch is not available")
    params.do_sample = do_sample
    decode_single_test(params)
