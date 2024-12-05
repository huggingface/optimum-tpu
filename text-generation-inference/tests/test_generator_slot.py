import numpy as np
import pytest
from text_generation_server.pb.generate_pb2 import Request
from transformers import AutoTokenizer, GenerationConfig


TOKENIZERS = ["NousResearch/Llama-2-7b-hf", "openai-community/gpt2"]

# Defining this global variable will parametrize all tests in this file
pytestmark = pytest.mark.parametrize(
    "input_text, generated_text",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            " Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,"
            " slipped quickly through the glass doors of Victory Mansions, though not quickly enough"
            " to prevent a swirl of gritty dust from entering along with him.",
        ],
        ["This sentence is written in chinese:", "我很感谢你的热情"],
        ["Some text might contain a lot of emojis like 😃", "😍💪 👉 👀"],
    ],
    ids=["spaces", "chinese-utf8", "emojis"],
)


@pytest.fixture(params=TOKENIZERS)
def tokenizer(request):
    t = AutoTokenizer.from_pretrained(request.param)
    t.padding_side = "left"
    t.pad_token_id = t.eos_token_id
    return t


def _test_decode_streaming(slot, return_tensors, tokenizer, input_text, generated_text):
    request = Request(id=0, inputs=input_text)
    slot.assign(0, request, GenerationConfig())

    inputs = tokenizer(input_text, padding="max_length", max_length=len(input_text) + 1, return_tensors=return_tensors)
    input_ids = inputs["input_ids"][0]
    generated_tokens = tokenizer(generated_text, add_special_tokens=False)["input_ids"]

    # We need to regenerate the full text as the tokenizer might change it (extra spaces might be added)
    all_input_ids = np.concatenate([input_ids, generated_tokens])
    full_text = tokenizer.decode(all_input_ids, skip_special_tokens=True)
    regenerated_text = full_text[len(input_text) :]

    # Initialize the slot with the inputs
    slot.reset(input_ids, selector=None)

    assert slot.generated_tokens == 0

    # Simulate an iterative generation (i.e. don't call select and use known tokens instead)
    decoded_text = ""
    for i in range(len(generated_tokens)):
        text = slot.append(generated_tokens[i])
        assert slot.generated_tokens == i + 1
        decoded_text += text

    assert decoded_text == regenerated_text


@pytest.mark.jetstream
def test_decode_streaming_jetstream(tokenizer, input_text, generated_text):
    from text_generation_server.jetstream_pt_support.generator import Slot

    slot = Slot(0, tokenizer)
    _test_decode_streaming(slot, "pt", tokenizer, input_text, generated_text)

@pytest.mark.torch_xla
def test_decode_streaming(tokenizer, input_text, generated_text):
    from text_generation_server.generator import Slot

    # Note: device used is cpu to make it faster
    slot = Slot(0, tokenizer, "cpu")
    _test_decode_streaming(slot, "pt", tokenizer, input_text, generated_text)
