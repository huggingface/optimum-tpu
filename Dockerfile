FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

WORKDIR /app
RUN pip install fastapi uvicorn

COPY server.py .

EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
