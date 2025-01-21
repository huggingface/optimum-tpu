# Installing Optimum-TPU inside a Docker Container

This guide explains how to run Optimum-TPU within a Docker container using the official PyTorch/XLA image.

## Prerequisites

Before starting, ensure you have:
- Docker installed on your system
- Access to a TPU instance
- Sufficient permissions to run privileged containers

## Using the PyTorch/XLA Base Image

### 1. Pull the Docker Image

First, set the environment variables for the image URL and version:

```bash
export TPUVM_IMAGE_URL=us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla
export TPUVM_IMAGE_VERSION=r2.5.0_3.10_tpuvm

# Pull the image
docker pull ${TPUVM_IMAGE_URL}:${TPUVM_IMAGE_VERSION}
```

### 2. Run the Container

Launch the container with the necessary flags for TPU access:

```bash
docker run -ti \
    --rm \
    --shm-size 16GB
    --privileged \
    --net=host \
    ${TPUVM_IMAGE_URL}@sha256:${TPUVM_IMAGE_VERSION} \
    bash
```
`--shm-size 16GB --privileged --net=host` is required for docker to access the TPU

### 3. Install Optimum-TPU

Once inside the container, install Optimum-TPU:

```bash
pip install optimum-tpu -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Verification

To verify your setup, run this simple test:

```bash
python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

You should see output indicating the XLA device is available (e.g., `xla:0`).

## Next Steps
After setting up your container, you can:
- Start training models using Optimum-TPU. Refer to our [training example section](../howto/more_examples).
- Run inference workloads. Check out our [serving guide](../howto/serving).
