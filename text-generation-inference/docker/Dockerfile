# Fetch and extract the TGI sources (TGI_VERSION is mandatory)
FROM alpine AS tgi
ARG TGI_VERSION
RUN test -n ${TGI_VERSION:?}
RUN mkdir -p /tgi
ADD https://github.com/huggingface/text-generation-inference/archive/${TGI_VERSION}.tar.gz /tgi/sources.tar.gz
RUN tar -C /tgi -xf /tgi/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TGI original Dockerfile)
# Note that the build image is aligned on the same Linux version as the base image (Debian bookworm/ Ubuntu 22.04)
FROM lukemathwalker/cargo-chef:latest-rust-1.79-bookworm AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef as planner
COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.11-dev
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --profile release-opt --recipe-path recipe.json

COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
RUN cargo build --profile release-opt

# Python base image
FROM ubuntu:22.04 AS base

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip3 --no-cache-dir install --upgrade pip

# VERSION is a mandatory parameter
ARG VERSION
RUN test -n ${VERSION:?}

# Python server build image
FROM base AS pyserver

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    make \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN install -d /pyserver
WORKDIR /pyserver
COPY text-generation-inference/server server
COPY --from=tgi /tgi/proto proto
RUN pip3 install -r server/build-requirements.txt
RUN VERBOSE=1 BUILDDIR=/pyserver/build PROTODIR=/pyserver/proto VERSION=${VERSION} make -C server gen-server

# TPU base image (used for deployment)
FROM base AS tpu_base

ARG VERSION=${VERSION}

# Install system prerequisites
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    libpython3.10 \
    libpython3.11 \
    python3.11 \
    git \
    gnupg2 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update pip
RUN pip install --upgrade pip

# Install HuggingFace packages
ARG TRANSFORMERS_VERSION='4.41.1'
ARG ACCELERATE_VERSION='0.27.2'
ARG SAFETENSORS_VERSION='0.4.2'

# TGI base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80 \
    VERSION=${VERSION}

COPY . /opt/optimum-tpu

# Install requirements for TGI, that uses python3.11
RUN python3.11 -m pip install transformers==${TRANSFORMERS_VERSION}

# Install requirements for optimum-tpu, then for TGI then optimum-tpu
RUN python3 -m pip install hf_transfer safetensors==${SAFETENSORS_VERSION} typer
RUN python3 /opt/optimum-tpu/optimum/tpu/cli.py install-jetstream-pytorch --yes
RUN python3 -m pip install -e /opt/optimum-tpu \
        -f https://storage.googleapis.com/libtpu-releases/index.html

# Install benchmarker
COPY --from=builder /usr/src/target/release-opt/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release-opt/text-generation-router-v2 /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release-opt/text-generation-launcher /usr/local/bin/text-generation-launcher
# Install python server
COPY --from=pyserver /pyserver/build/dist dist
RUN pip install dist/text_generation_server*.tar.gz


# TPU compatible image for Inference Endpoints
FROM tpu_base as inference-endpoint

COPY text-generation-inference/docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# TPU compatible image
FROM tpu_base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]
