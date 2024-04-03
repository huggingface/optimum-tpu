#  Copyright 2024 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-neuron.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

.PHONY:	build_dist style style_check clean

TGI_VERSION ?= 1.4.2

rwildcard=$(wildcard $1) $(foreach d,$1,$(call rwildcard,$(addsuffix /$(notdir $d),$(wildcard $(dir $d)*))))

VERSION := $(shell awk '/__version__ = "(.*)"/{print $$3}' optimum/tpu/version.py | sed 's/"//g')

PACKAGE_DIST = dist/optimum-tpu-$(VERSION).tar.gz
PACKAGE_WHEEL = dist/optimum_tpu-$(VERSION)-py3-none-any.whl
PACKAGE_PYTHON_FILES = $(call rwildcard, optimum/*.py)
PACKAGE_FILES = $(PACKAGE_PYTHON_FILES)  \
				setup.py \
				setup.cfg \
				pyproject.toml \
				README.md \
				MANIFEST.in

# Package build recipe
$(PACKAGE_DIST) $(PACKAGE_WHEEL): $(PACKAGE_FILES)
	python -m build

clean:
	rm -rf dist

tpu-tgi:
	docker build --rm -f text-generation-inference/Dockerfile \
	             --build-arg VERSION=$(VERSION) \
	             --build-arg TGI_VERSION=$(TGI_VERSION) \
				 -t tpu-tgi:$(VERSION) .
	docker tag tpu-tgi:$(VERSION) tpu-tgi:latest

# Run code quality checks
style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix

# Utilities to release to PyPi
build_dist_install_tools:
	python -m pip install build
	python -m pip install twine

build_dist: ${PACKAGE_DIST} ${PACKAGE_WHEEL}

pypi_upload: ${PACKAGE_DIST} ${PACKAGE_WHEEL}
	python -m twine upload ${PACKAGE_DIST} ${PACKAGE_WHEEL}

# Tests

test_installs:
	python -m pip install .[tests]
	python -m pip install git+https://github.com/huggingface/transformers.git

# Stand-alone TGI server for unit tests outside of TGI container
tgi_server:
	python -m pip install -r text-generation-inference/server/build-requirements.txt
	make -C text-generation-inference/server clean
	VERSION=${VERSION} TGI_VERSION=${TGI_VERSION} make -C text-generation-inference/server gen-server

tgi_test: tgi_server
	python -m pip install .[tpu,tests]
	find text-generation-inference -name "text_generation_server-$(VERSION)-py3-none-any.whl" \
	                               -exec python -m pip install --force-reinstall {} \;
	python -m pytest -sv text-generation-inference/tests

tgi_docker_test: tpu-tgi
	python -m pip install -r text-generation-inference/integration-tests/requirements.txt
	python -m pytest -sv text-generation-inference/integration-tests
