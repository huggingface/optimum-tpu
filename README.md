<div align="center">

# Optimum-TPU
===========================
<h4>Take the most out of Google Cloud TPUs with the ease of 🤗 transformers</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://huggingface.co/docs/optimum/index)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

</div>

[Tensor Processing Units (TPU)](https://cloud.google.com/tpu) are AI accelerator made by Google to optimize
performance and cost from AI training to inference.

This repository exposes an interface similar to what Hugging Face transformers library provides to interact with
a magnitude of models developed by research labs, institutions and the community.

We aim at providing our user the best possible performances targeting Google Cloud TPUs for both training and inference
working closely with Google and Google Cloud to make this a reality.

## Installation

`optimum-tpu` comes with an handy PyPi released package compatible with your classical python dependency management tool.

`pip install optimum-tpu`

## Inference

`optimum-tpu` provides a set of dedicated tools and integrations in order to leverage Cloud TPUs for inference, especially
on the latest TPU version `v5`.


### Text-Generation-Inference

As part of the integration, we do support a [text-generation-inference (TGI)](https://github.com/huggingface/optimum-tpu/tree/main/text-generation-inference) backend allowing to deploy and serve
incoming HTTP requests and execute them on Cloud TPUs.

Please see the [TGI specific documentation]() on how to get started

## Training

Journey just started, we are working hard on this; stay tuned! ⚒️
