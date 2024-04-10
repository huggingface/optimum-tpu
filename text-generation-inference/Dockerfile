# Fetch and extract the TGI sources (TGI_VERSION is mandatory)
FROM alpine AS tgi
ARG TGI_VERSION
RUN test -n ${TGI_VERSION:?}
RUN mkdir -p /tgi
ADD https://github.com/huggingface/text-generation-inference/archive/refs/tags/v${TGI_VERSION}.tar.gz /tgi/sources.tar.gz
RUN tar -C /tgi -xf /tgi/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TGI original Dockerfile)
# Note that the build image is aligned on the same Linux version as the base image (Debian bookworm/ Ubuntu 22.04)
FROM lukemathwalker/cargo-chef:latest-rust-1.76-bookworm AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef as planner
COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/launcher launcher
RUN cargo build --release --workspace --exclude benchmark

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

# TGI base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80 \
    VERSION=${VERSION}

# Install system prerequisites
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    libpython3.10 \
    git \
    gnupg2 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . /opt/optimum-tpu

# Update pip
RUN pip install --upgrade pip


# Install HuggingFace packages
ARG TRANSFORMERS_VERSION='4.39.3'
ARG ACCELERATE_VERSION='0.27.2'
ARG SAFETENSORS_VERSION='0.4.2'

# Install Torch and XLA packages for TPU
RUN python3 -m pip install "torch~=2.2.0" "torch_xla[tpu]~=2.2.0" -f https://storage.googleapis.com/libtpu-releases/index.html && \
    python3 -m pip install hf_transfer safetensors==${SAFETENSORS_VERSION} accelerate==${ACCELERATE_VERSION} && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPTIMUM_TPU=${VERSION} python3 -m pip install -e /opt/optimum-tpu

# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher
# Install python server
COPY --from=pyserver /pyserver/build/dist dist
RUN pip install dist/text-generation-server*.tar.gz

# TPU compatible image
FROM tpu_base as tpu_entrypoint

COPY text-generation-inference/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM tpu_base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]
