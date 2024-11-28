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
