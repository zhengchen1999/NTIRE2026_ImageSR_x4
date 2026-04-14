# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages
from pathlib import Path


def normalize_requirement(line: str) -> str:
    if not line.startswith("git+"):
        return line

    repo = line.rsplit("/", 1)[-1]
    name = repo.split(".git", 1)[0].split("@", 1)[0]
    return f"{name} @ {line}"


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path) as f:
        return [
            normalize_requirement(line.strip())
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="fastgen",
    version="0.1.0",
    description="FastGen is a PyTorch-based framework for building fast generative models using various distillation and acceleration techniques.",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
