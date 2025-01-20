
import pytest
from decode_tests_utils import *


# All tests in this file are for jetstream
pytestmark = pytest.mark.jetstream

@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_decode_single_jetstream_pytorch():
    params = DecodeTestParams(
            model_id="Qwen/Qwen2.5-0.5B",
            # model_id="Maykeye/TinyLLama-v0",
            # model_id="tengomucho/tiny_qwen2.5",
            sequence_length=256,
            expected_text=" Winston Smith, his chin nuzzled into his breast, stretched, and looked out across the city",
            max_new_tokens=20,
        )


    model_path = prepare_model(params.model_id, params.sequence_length)
    # model_path = paqrams.model_id

    # input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    input_text = "Winston Smith, his chin nuzzled into his"
    max_new_tokens = params.max_new_tokens

    generator = AutoGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=params.sequence_length
    )
    request = create_request(
        id=0,
        inputs=input_text,
        max_new_tokens=max_new_tokens,
        do_sample=params.do_sample,
        top_k=params.top_k,
        seed=1234,
        repetition_penalty=params.repetition_penalty,
    )
    batch = Batch(id=0, requests=[request], size=1, max_tokens=params.sequence_length)
    generations, next_batch = generator.prefill(batch)
    print(f"generations prefill: {generations}")
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in tqdm(range(max_new_tokens - 1)):
        assert next_batch.size == 1
        assert next_batch.max_tokens == params.sequence_length
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])

    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    # print(f"generations: {generations}")
    print(f"Generated text: {output.text}")
    if params.do_sample:
        if output.text == params.expected_text:
            print("❌: Expected text is equal to generated text")
            return
    else:
        if output.text != params.expected_text:
            print("❌: Expected text is not equal to generated text")
            return
    print("✅: Test passed")
