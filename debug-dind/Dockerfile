FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

# Set up shared memory, IPC and privileges
ENV SHM_SIZE=16gb
ENV IPC=host
ENV PRIVILEGED=true

# Mount cache volume
VOLUME /mnt/hf_cache

# Copy test files
COPY xla_add.py /workspace/xla_add.py

# Install PyTorch XLA dependencies
RUN python -m pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Set working directory
WORKDIR /workspace

# Run XLA test by default
CMD ["python", "xla_add.py"]
