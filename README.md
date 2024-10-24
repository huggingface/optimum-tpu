<div align="center">

Optimum-TPU
===========================
<h4>Take the most out of Google Cloud TPUs with the ease of 🤗 transformers</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://huggingface.co/docs/optimum/index)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)
[![Optimum TPU / Test TGI on TPU](https://github.com/huggingface/optimum-tpu/actions/workflows/test-pytorch-xla-tpu-tgi.yml/badge.svg)](https://github.com/huggingface/optimum-tpu/actions/workflows/test-pytorch-xla-tpu-tgi.yml)
</div>

[Tensor Processing Units (TPU)](https://cloud.google.com/tpu) are AI accelerator made by Google to optimize
performance and cost from AI training to inference.

This repository exposes an interface similar to what Hugging Face transformers library provides to interact with
a magnitude of models developed by research labs, institutions and the community.

We aim at providing our user the best possible performances targeting Google Cloud TPUs for both training and inference
working closely with Google and Google Cloud to make this a reality.


## Supported Model and Tasks

We currently support a few LLM models targeting text generation scenarios:
- 💎 Gemma (2b, 7b)
- 🦙 Llama2 (7b) and Llama3 (8b)
- 💨 Mistral (7b)


## Installation

`optimum-tpu` comes with an handy PyPi released package compatible with your classical python dependency management tool.

`pip install optimum-tpu -f https://storage.googleapis.com/libtpu-releases/index.html`

`export PJRT_DEVICE=TPU`


## Inference

`optimum-tpu` provides a set of dedicated tools and integrations in order to leverage Cloud TPUs for inference, especially
on the latest TPU version `v5e`. 

Other TPU versions will be supported along the way.

### Text-Generation-Inference

As part of the integration, we do support a [text-generation-inference (TGI)](https://github.com/huggingface/optimum-tpu/tree/main/text-generation-inference) backend allowing to deploy and serve
incoming HTTP requests and execute them on Cloud TPUs.

Please see the [TGI specific documentation](text-generation-inference) on how to get started.

### JetStream Pytorch Engine

`optimum-tpu` provides an optional support of JetStream Pytorch engine inside of TGI. This support can be installed using the dedicated CLI command:

```shell
optimum-tpu install-jetstream-pytorch
```

To enable the support, export the environment variable `JETSTREAM_PT=1`.

## Training

Fine-tuning is supported and tested on the TPU `v5e`. We have tested so far:

- 🦙 Llama-2 7B and Llama-3 8B
- 💎 Gemma 2B and 7B

You can check the examples:

- [Fine-Tune Gemma on Google TPU](https://github.com/huggingface/optimum-tpu/blob/main/examples/language-modeling/gemma_tuning.ipynb)
- The [Llama fine-tuning script](https://github.com/huggingface/optimum-tpu/blob/main/examples/language-modeling/llama_tuning.md)
