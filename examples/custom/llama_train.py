#pip install datasets
#pip install accelerate
from optimum.tpu import fsdp_v2
fsdp_v2.use_fsdp_v2()

import torch
from transformers import AutoTokenizer
from optimum.tpu import AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add custom token for padding Llama
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained(model_id)

from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=1,
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