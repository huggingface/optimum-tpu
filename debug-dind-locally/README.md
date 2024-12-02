# Run xla inside a container

docker build -t xla_add_container -f Dockerfile .

docker run \
    --shm-size=16G \
    --privileged \
    --network host \
    --ipc host \
    run_xla_add_inside_a_container

# Run xla with docker in docker

docker build -t xla_add_dind -f Dockerfile.dind .

docker run \
  --privileged \
  --network host \
  --ipc host \
  -it \
  xla_add_dind \
  sh -c "docker build -t debug-dind . && docker run --shm-size=16G --privileged --network host --ipc host debug-dind"

<!-- docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm /bin/bash -->

docker run --privileged --net host --shm-size=16G \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --hostname outer-container \
  -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm /bin/bash

apt-get install -y docker.io
sudo dockerd > /var/log/dockerd.log 2>&1 &


docker run --privileged --net host --shm-size=16G \
 -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm \
 PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print('Supported devices:', xm.get_xla_supported_devices())"




python3 -c "import torch_xla.core.xla_model as xm; print(f'Current device: {xm.xla_device()}')"
PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; import torch_xla.debug.metrics as met; print(met.metrics_report())"


PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print('Supported devices:', xm.get_xla_supported_devices())"
-> Supported devices: ['xla:0', 'xla:1', 'xla:2', 'xla:3', 'xla:4', 'xla:5', 'xla:6', 'xla:7']

PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print(f'Available devices: {xm.xrt_world_size()}')"