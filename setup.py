from setuptools import setup, find_packages
import pathlib
import os.path
import codecs

BASE_DIR = pathlib.Path(__file__).parent.resolve()


def readme():
    with open(f"{BASE_DIR}/README.md") as f:
        return f.read()


def requirements(_requirements: str):
    with open(f"{BASE_DIR}/{_requirements}.txt") as f:
        return f.read().splitlines()


def get_package_version(rel_path):
    def read(_rel_path):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, _rel_path), "r") as fp:
            return fp.read()

    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="gpt-sw3-tokenizer",
    version=get_package_version("src/__about__.py"),
    author="Felix Stollenwerk",
    author_email="felix.stollenwerk@ai.se",
    description="BPE tokenizer with HuggingFace / SentencePiece",
    long_description=readme(),
    keywords=[
        "NLP",
        "transformer",
        "tokenizer",
        "BPE",
        "SentencePiece",
        "HuggingFace",
    ],
    license="Apache 2.0",
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=requirements("requirements"),
    extras_require={
        "dev": requirements("requirements_dev"),
    },
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
)
