#!/usr/bin/env python3
"""
Setup script for Flow Block Diffusion Model.
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from file."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def read_readme():
    """Read README file."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="flow-block-diffusion",
    version="0.1.0",
    author="Flow BD3-LM Implementation",
    description="Block Diffusion Language Models for Network Flow Statistics Generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "cpu": ["torch>=2.0.0+cpu"],
        "full": [
            "omegaconf>=2.1.0",
            "hydra-core>=1.1.0", 
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flow-generate=generate_flows:main",
            "flow-demo=demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
    ],
)