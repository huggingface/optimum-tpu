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

Training Large Language Models (LLMs) on Google Tensor Processing Units (TPUs) with Single Program Multiple Data (SPMD) offers a multitude of benefits. TPUs provide competitive processing power, enabling good training times and allowing researchers to experiment with larger models and datasets efficiently. SPMD architecture optimizes resource utilization by distributing tasks across multiple TPUs, enhancing parallelism and scalability. This approach not only accelerates training but also enables seamless scaling to tackle increasingly complex natural language processing tasks. Moreover, the combination of TPUs and SPMD ensures cost-effectiveness by maximizing computational efficiency. For details on using SPMD on Pytorch/XLA you can refer to the [documentation](https://github.com/pytorch/xla/blob/master/docs/spmd.md).

This example shows how to tune Meta's LLama2 and Llama3 models on single-host and multi-host TPUs. For information on this, you can consult the [architecture documentation](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).

## Tuning LLama2

### Prerequisites

We consider you have already created a single-host TPU VM, such as a `v5litepod8` setup, and you have ssh access to the machine.
You need to install few modules:

```shell
pip install datasets evaluate
```

Note that to work with the gated model, you will need to export the `HF_TOKEN` variable, or authenticate using the `huggingface-cli login` command (see [here](https://huggingface.co/settings/tokens) for details).

### Instructions

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

## Llama3 tuning on multi-host setups

Tuning Llama3 can be done on larger multi-host setups, such as the `v5litepod16`. We assume here you already created the required TPU Pod slices VM. For details on how to do that, please check the related [documentation](https://cloud.google.com/tpu/docs/pytorch-pods).

### Instructions

First, you will need to setup some variable to simplify using the `gcloud` command line to issue the commands on all workers.

```shell
export ZONE=us-central1-a
export TPU_NAME=tpu-name
```

Once that is done, you can setup the environment that will be used for the tuning.

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --worker=all --command="\
    rm -rf hf;\
    virtualenv hf;\
    source hf/bin/activate;\
    pip install -U pip;\
    pip install 'torch~=2.3.0' 'torch_xla[tpu]~=2.3.0' \
        -f https://storage.googleapis.com/libtpu-releases/index.html;\
    pip install numpy accelerate sentencepiece datasets evaluate"
```

To complete this example, we will fine-tune Llama3 on the `wikitext` dataset. To do that, it will be necessary to use the `HF_TOKEN` set as environment variable, to propagate that on all workers, and allow them to access the gated model.
Finally, it will be possible to launch this command to use optimum-tpu's custom `run_clm.py` script.

```shell
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --worker=all \
  --command="\
    source hf/bin/activate;\
    rm -rf optimum-tpu;\
    git clone https://github.com/huggingface/optimum-tpu.git;\
    cd optimum-tpu;\
    git checkout basic-training;
    pip install -e .;\
    HF_TOKEN=$HF_TOKEN python3 examples/language-modeling/run_clm.py \
      --model_name_or_path meta-llama/Meta-Llama-3-8B \
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
      --spmd_grad_chkpt"
```
