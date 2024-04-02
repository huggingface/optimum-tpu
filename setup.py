import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/tpu/version.py
filepath = "optimum/tpu/version.py"
try:
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


INSTALL_REQUIRES = [
    "transformers == 4.38.1",
]

TESTS_REQUIRE = [
    "pytest",
    "safetensors",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "isort",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRES,
    "tpu": [
        "wheel",
        "torch-xla>=2.2.0",
        "torch>=2.2.0",
    ],
}

setup(
    name="optimum-tpu",
    version=__version__,
    description=(
        "Optimum TPU is the interface between the Hugging Face Transformers library and Google Cloud TPU devices. "
        "It provides a set of tools enabling easy model loading and inference on single and multiple TPU devices for "
        "different downstream tasks."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, fine-tuning, inference, tpu, cloud-tpu, gcp",
    url="https://huggingface.co/hardware/aws",
    author="HuggingFace Inc. Machine Learning Optimization Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)
