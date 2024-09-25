#!/bin/bash
THIS_DIR=$(dirname "$0")

deps_dir=deps
rm -rf $deps_dir
mkdir -p $deps_dir


# install torch cpu to avoid GPU requirements
pip install -r $THIS_DIR/requirements.txt
cd $deps_dir
git clone https://github.com/google/jetstream-pytorch.git
cd jetstream-pytorch
git checkout ec4ac8f6b180ade059a2284b8b7d843b3cab0921
git submodule update --init --recursive
# We cannot install in a temporary directory because the directory should not be deleted after the script finishes,
# because it will install its dependendencies from that directory.
pip install -e .
