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

# Tuning Llama2 Model with the `wikitext-2-raw-v1` Dataset

Training Large Language Models (LLMs) on Google Tensor Processing Units (TPUs) with Single Program Multiple Data (SPMD) offers a multitude of benefits. TPUs provide competitive processing power, enabling good training times and allowing researchers to experiment with larger models and datasets efficiently. SPMD architecture optimizes resource utilization by distributing tasks across multiple TPUs, enhancing parallelism and scalability. This approach not only accelerates training but also enables seamless scaling to tackle increasingly complex natural language processing tasks. Moreover, the combination of TPUs and SPMD ensures cost-effectiveness by maximizing computational efficiency. For details on using SPMD on Pytorch/XLA you can refer to the [documentation](https://github.com/pytorch/xla/blob/master/docs/spmd.md).

## Prerequisites

You need to install few modules:

```shell
pip install datasets evaluate
```

## Instructions

You can now use a modified version of `run_clm.py` to train your model on the `wikitext-2-raw-v1` dataset:

```bash
python examples/language-modeling/run_clm.py \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --dataset_name wikitext \
   --dataset_config_name wikitext-2-raw-v1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --num_train_epochs 1 \
   --do_train \
   --output_dir /tmp/output \
   --overwrite_output_dir \
   --save_strategy no \
   --logging_strategy no \
   --remove_unused_columns no \
   --optim adafactor \
   --torch_dtype bfloat16 \
   --dataloader_drop_last yes \
   --block_size 1024 \
   --learning_rate 5e-5 \
   --max_steps 10 \
   --logging_steps 10 \
   --spmd_2d_sharding 1 \
   --spmd_grad_chkpt
```

This step will use the `Trainer` class in `optimum-tpu` to mark the model sharding and adapt it for the training on Pytorch/XLA. Training can take around 20 minutes on a TPU `v5e-litepod8`.
