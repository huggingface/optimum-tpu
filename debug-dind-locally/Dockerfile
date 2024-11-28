FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

WORKDIR /workspace

COPY xla_add.py xla_add.py

# RUN python -m pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html

CMD ["python", "xla_add.py"]
