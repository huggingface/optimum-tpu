<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Tuning Llama Model

Training Large Language Models (LLMs) on Google Tensor Processing Units (TPUs) with Single Program Multiple Data (SPMD) offers a multitude of benefits. TPUs provide competitive processing power, enabling good training times and allowing researchers to experiment with larger models and datasets efficiently. SPMD architecture optimizes resource utilization by distributing tasks across multiple TPUs, enhancing parallelism and scalability.
The easiest approach to tune a model with SPMD is using Fully Sharded Data Parallel [(FSDP)](https://engineering.fb.com/2021/07/15/open-source/fsdp/). Pytorch/XLA most recent and performant implementation is [FSDP v2](https://github.com/pytorch/xla/blob/master/docs/fsdpv2.md), that allows to shard weights, activations and outputs.


This example shows how to tune Meta's LLama2 and Llama3 models on single host TPUs. For information on TPUs architecture, you can consult the [documentation](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).


### Prerequisites

We consider you have already created a single-host TPU VM, such as a `v5litepod8` setup, and you have ssh access to the machine.
You need to install few modules:

```shell
pip install datasets evaluate
```

Note that to work with the gated model, you will need to export the `HF_TOKEN` variable, or authenticate using the `huggingface-cli login` command (see [here](https://huggingface.co/settings/tokens) for details).

### Instructions

To use FSDPv2, it needs to be enabled:

```python
from optimum.tpu import fsdp_v2
fsdp_v2.use_fsdp_v2()
```

Then, the tokenizer and model need to be loaded. We will choose [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) for this example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add custom token for padding Llama
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
```

To tune the model with the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset, you can load it and obtain the `quote` column:

```python
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

You then need to specify the FSDP training arguments to enable the sharding feature,the function will deduce the classes that should be sharded:

```python
fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
```

Now training can be done as simply as using the standard `Trainer` class:

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=24,
        num_train_epochs=10,
        max_steps=-1,
        output_dir="/tmp/output",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last=True,  # Required by FSDP v2 and SPMD.
        **fsdp_training_args,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
```
