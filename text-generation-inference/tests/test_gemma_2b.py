import os

import pytest
from text_generation_server.generator import TpuGenerator
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)
from tqdm import tqdm

from optimum.tpu.model import fetch_model


MODEL_ID = "google/gemma-2b"
SEQUENCE_LENGTH = 1024


@pytest.fixture(scope="module")
def model_path():
    # Add variables to environment so they can be used in AutoModelForCausalLM
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    path = fetch_model(MODEL_ID)
    return path


def create_request(
    id: int,
    inputs: str,
    max_new_tokens=20,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int = 0,
    repetition_penalty: float = 1.0,
):
    # For these tests we can safely set typical_p to 1.0 (default)
    typical_p = 1.0
    if not do_sample:
        # Drop top_p parameter to avoid warnings
        top_p = 1.0
    parameters = NextTokenChooserParameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
        repetition_penalty=repetition_penalty,
        typical_p=typical_p,
    )
    stopping_parameters = StoppingCriteriaParameters(max_new_tokens=max_new_tokens)
    return Request(id=id, inputs=inputs, parameters=parameters, stopping_parameters=stopping_parameters)


def test_decode_single(model_path):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20
    generated_text = "\n\nThe first thing I noticed was the smell of the rain. It was a smell I had never"

    generator = TpuGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH
    )
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in tqdm(range(max_new_tokens - 1)):
        assert next_batch.size == 1
        assert next_batch.max_tokens == 1024
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    assert output.text == generated_text


def test_decode_multi(model_path):
    generator = TpuGenerator.from_pretrained(
            model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH
    )
    prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
    ]
    for prompt in prompts:
        input_text = prompt # "It was a bright cold day in April, and the clocks were striking thirteen."
        max_new_tokens = 20
        # generated_text = "\n\nThe first thing I noticed was the smell of the rain. It was a smell I had never"
    
        # generator = TpuGenerator.from_pretrained(
        #     model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH
        # )
        request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
        batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
        generations, next_batch = generator.prefill(batch)
        # We already generated one token: call decode max_new_tokens - 1 times
        for _ in tqdm(range(max_new_tokens - 1)):
            assert next_batch.size == 1
            assert next_batch.max_tokens == 1024
            assert len(generations) == 1
            assert len(generations[0].tokens.ids) == 1
            generations, next_batch = generator.decode([next_batch])
        assert next_batch is None
        assert len(generations) == 1
        output = generations[0].generated_text
        print("---- One output text: ", output.text)
    print("---- finish all four inference tests")
    # assert output.generated_tokens == max_new_tokens
    # assert output.finish_reason == 0
    # assert output.text == generated_text

    # sampled_tokens_list.extend(token_ids)

    # print("---- All output tokens.")
    # print(sampled_tokens_list)
    # print("---- All output text.")
    # print(tokenizer.decode(sampled_tokens_list))
