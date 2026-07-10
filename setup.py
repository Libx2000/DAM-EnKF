#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="nuwa-assimilation",
    version="1.0.0",
    description="NuWa-Assimilation: AI-based Data Assimilation and Weather Forecasting Framework",
    author="Wang Wuxin",
    author_email="wuxinwang1997@gmail.com",
    url="https://github.com/wuxinwang1997/NuWa-Assimilation",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=0.11.0",
        "torchdata>=0.6.0",
        "hydra-core>=1.3.0",
        "hydra-colorlog>=1.2.0",
        "hydra-optuna-sweeper>=1.3.0",
        "pyrootutils>=1.0.0",
        "h5py>=3.8.0",
        "numpy>=1.24.0",
        "xarray>=2023.0.0",
        "timm>=0.9.0",
        "einops>=0.6.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "click>=8.0.0",
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)