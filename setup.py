#!/usr/bin/env python3

from setuptools import find_packages, setup
from pathlib import Path

project_dir = Path(__file__).parent
install_requires = (project_dir / "requirements.txt").read_text().splitlines()

setup(
    name="divide_lm",
    version="0.1",
    python_requires=">=3.6.0",
    description="Utility for deviding a language model by another one",
    author="Liyong Guo",
    author_email='guonwpu@qq.com',
    license="Apache-2.0 License",
    url="https://github.com/k2-fsa/divide_lm",
    packages=find_packages(),
    package_data={
        "":["requirements.txt"]
    },
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Language Model",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
