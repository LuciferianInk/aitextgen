import os

from pkg_resources import parse_requirements
from setuptools import setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name="aigen",
    packages=["aigen"],  # this must be the same as the name above
    version="0.6.0",
    description="A robust Python tool for text-based AI training and generation using GPT-2.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/aigen",
    keywords=["gpt-2", "gpt2", "text generation", "ai"],
    classifiers=[],
    license="MIT",
    entry_points={"console_scripts": ["aigen=aigen.cli:aigen_cli"]},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=_load_requirements(_PATH_ROOT),
)
