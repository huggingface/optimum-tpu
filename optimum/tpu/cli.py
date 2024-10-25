import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import typer


TORCH_VER = "2.4.0"
JETSTREAM_PT_VER = "02927c9f563082421abe8eedceabe8aedd7ec2f9"
DEFAULT_DEPS_PATH = os.path.join(Path.home(), ".jetstream-deps")

app = typer.Typer()


def _check_module(module_name: str):
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def _run(cmd: str):
    split_cmd = cmd.split()
    subprocess.check_call(split_cmd)


def _install_torch_cpu():
    # install torch CPU version to avoid installing CUDA dependencies
    _run(sys.executable + f" -m pip install torch=={TORCH_VER} --index-url https://download.pytorch.org/whl/cpu")


@app.command()
def install_pytorch_xla(
    force: bool = False,
):
    """
    Installs PyTorch XLA with TPU support.

    Args:
        force (bool): When set, force reinstalling even if Pytorch XLA is already installed.
    """
    if not force and _check_module("torch") and _check_module("torch_xla"):
        typer.confirm(
            "PyTorch XLA is already installed. Do you want to reinstall it?",
            default=False,
            abort=True,
        )
    _install_torch_cpu()
    _run(
        sys.executable
        + f" -m pip install torch-xla[tpu]=={TORCH_VER} -f https://storage.googleapis.com/libtpu-releases/index.html"
    )
    click.echo()
    click.echo(click.style("PyTorch XLA has been installed.", bold=True))


@app.command()
def install_jetstream_pytorch(
    deps_path: str = DEFAULT_DEPS_PATH,
    yes: bool = False,
):
    """
    Installs Jetstream Pytorch with TPU support.

    Args:
        deps_path (str): Path where Jetstream Pytorch dependencies will be installed.
        yes (bool): When set, proceed installing without asking questions.
    """
    if not _check_module("torch"):
        _install_torch_cpu()
    if not yes and _check_module("jetstream_pt") and _check_module("torch_xla2"):
        typer.confirm(
            "Jetstream Pytorch is already installed. Do you want to reinstall it?",
            default=False,
            abort=True,
        )

    jetstream_repo_dir = os.path.join(deps_path, "jetstream-pytorch")
    if not yes and os.path.exists(jetstream_repo_dir):
        typer.confirm(
            f"Directory {jetstream_repo_dir} already exists. Do you want to delete it and reinstall Jetstream Pytorch?",
            default=False,
            abort=True,
        )
    shutil.rmtree(jetstream_repo_dir, ignore_errors=True)
    # Create the directory if it does not exist
    os.makedirs(deps_path, exist_ok=True)
    # Clone and install Jetstream Pytorch
    os.chdir(deps_path)
    _run("git clone https://github.com/google/jetstream-pytorch.git")
    os.chdir("jetstream-pytorch")
    _run(f"git checkout {JETSTREAM_PT_VER}")
    _run("git submodule update --init --recursive")
    # We cannot install in a temporary directory because the directory should not be deleted after the script finishes,
    # because it will install its dependendencies from that directory.
    _run(sys.executable + " -m pip install -e .")

    _run(
        sys.executable
        + f" -m pip install torch_xla[pallas]=={TORCH_VER} "
        + " -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html"
        + " -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        + " -f https://storage.googleapis.com/libtpu-releases/index.html"
    )
    # Install PyTorch XLA pallas
    click.echo()
    click.echo(click.style("Jetstream Pytorch has been installed.", bold=True))


if __name__ == "__main__":
    sys.exit(app())
