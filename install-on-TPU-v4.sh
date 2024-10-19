sudo apt remove unattended-upgrades
sudo apt update
export PJRT_DEVICE=TPU
export PATH="/home/artuskg/.local/bin:$PATH"
pip install build
pip install --upgrade setuptools
sudo apt install python3.10-venv

git clone https://github.com/huggingface/optimum-tpu.git

cd optimum-tpu
make
make build_dist_install_tools
make build_dist

python -m venv optimum_tpu_env
source optimum_tpu_env/bin/activate

pip install torch==2.4.0 torch_xla[tpu]==2.4.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
pip uninstall torchvision # it might insist von 2.4.1
pip install -e .

huggingface-cli login
gsutil cp -r gs://entropix/huggingface_hub ~/.cache/huggingface/hub
