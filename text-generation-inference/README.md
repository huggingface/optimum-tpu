# Text-generation-inference docker image for Pytorch/XLA

This docker image integrates into a base image:

- the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) launcher and scheduling front-end,
- an XLA specific inference server for text-generation.

## Features

The basic features of the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) product are supported:

- continuous batching,
- token streaming,
- greedy search and multinomial sampling using [transformers](https://huggingface.co/docs/transformers/generation_strategies#customize-text-generation).

The main differences with the standard service for CUDA and CPU backends are that:

- the service uses a single internal static batch,
- new requests are inserted in the static batch during prefill,
- the static KV cache is rebuilt entirely during prefill.

## License

This docker image is released under [HFOIL 1.0](https://github.com/huggingface/text-generation-inference/blob/bde25e62b33b05113519e5dbf75abda06a03328e/LICENSE).

HFOIL stands for Hugging Face Optimized Inference License, and it has been specifically designed for our optimized inference solutions. While the source code remains accessible, HFOIL is not a true open source license because we added a restriction: to sell a hosted or managed service built on top of TGI, we require a separate agreement.

Please refer to [this reference documentation](https://github.com/huggingface/text-generation-inference/issues/726) to see if the HFOIL 1.0 restrictions apply to your deployment.

## Deploy the service

The service is launched simply by running the tpu-tgi container with two sets of parameters:

```
docker run <system_parameters> ghcr.io/huggingface/tpu-tgi:latest <service_parameters>
```

- system parameters are used to map ports, volumes and devices between the host and the service,
- service parameters are forwarded to the `text-generation-launcher`.

### Common system parameters

Finally, you might want to export the `HF_TOKEN` if you want to access gated repository.

Here is an example of a service instantiation on a single host TPU:

```
docker run -p 8080:80 \
       --net=host --privileged \
       -v $(pwd)/data:/data \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/tpu-tgi:latest \
       <service_parameters>
```



### Using a standard model from the 🤗 [HuggingFace Hub](https://huggingface.co/models)


The snippet below shows how you can deploy a service from a hub standard model:

```
docker run -p 8080:80 \
       --net=host --privileged \
       -v $(pwd)/data:/data \
       -e HF_TOKEN=${HF_TOKEN} \
       -e HF_BATCH_SIZE=1 \
       -e HF_SEQUENCE_LENGTH=1024 \
       ghcr.io/huggingface/tpu-tgi:latest \
       --model-id mistralai/Mistral-7B-v0.1 \
       --max-concurrent-requests 1 \
       --max-input-length 512 \
       --max-total-tokens 1024 \
       --max-batch-prefill-tokens 512 \
       --max-batch-total-tokens 1024
```


### Choosing service parameters

Use the following command to list the available service parameters:

```
docker run ghcr.io/huggingface/tpu-tgi --help
```

The configuration of an inference endpoint is always a compromise between throughput and latency: serving more requests in parallel will allow a higher throughput, but it will increase the latency.

The models for now work with static input dimensions `[batch_size, max_length]`.

It leads to a maximum number of tokens of `max_tokens = batch_size * max_length`.

This adds several restrictions to the following parameters:

- `--max-concurrent-requests` must be set to `batch size`,
- `--max-input-length` must be lower than `max_length`,
- `--max-total-tokens` must be set to `max_length` (it is per-request),
- `--max-batch-prefill-tokens` must be set to `batch_size * max_input_length`,
- `--max-batch-total-tokens` must be set to `max_tokens`.

### Choosing the correct batch size

As seen in the previous paragraph, model static batch size has a direct influence on the endpoint latency and throughput.

Please refer to [text-generation-inference](https://github.com/huggingface/text-generation-inference) for optimization hints.

Note that the main constraint is to be able to fit the model for the specified `batch_size` within the total device memory available
on your instance.

## Query the service

You can query the model using either the `/generate` or `/generate_stream` routes:

```
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

## Build your own image

The image must be built from the top directory

```
make tpu-tgi
```
