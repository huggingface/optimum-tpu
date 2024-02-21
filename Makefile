
VERSION := "0.0.1"

TGI_VERSION ?= 1.4.0

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


test_installs:
	python -m pip install pytest safetensors
	python -m pip install git+https://github.com/huggingface/transformers.git

# Stand-alone TGI server for unit tests outside of TGI container
tgi_server:
	python -m pip install -r text-generation-inference/server/build-requirements.txt
	make -C text-generation-inference/server clean
	VERSION=${VERSION} TGI_VERSION=${TGI_VERSION} make -C text-generation-inference/server gen-server

tgi_test: tgi_server
	python -m pip install pytest
	find text-generation-inference -name "text_generation_server-$(VERSION)-py3-none-any.whl" \
	                               -exec python -m pip install --force-reinstall {} \;
	python -m pytest -s text-generation-inference/tests
