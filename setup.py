#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'flax>=0.6.8',
    'jax>=0.4.8',
    'jaxlib>=0.4.7',
    'matplotlib>=3.5.1',
    'numpy>=1.22.2',
    'optax>=0.1.4',
    'scipy>=1.8.0',
    'wandb>=0.12.11',
    'termcolor>=1.1.0',
    'torch>=1.6.0',
    'distrax~=0.1.2',
    'gym>=0.26.0',
    'argparse-dataclass>=0.2.1',
    'tqdm',
    'seaborn',
    'cloudpickle',
    'pandas',
    'dm_control',
    'trajax @ git+ssh://git@github.com/lenarttreven/trajax.git'
]

extras = {'dev': ['seaborn', 'control>=0.9.2']}
setup(
    name='mbse',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.8',
    include_package_data=True,
    install_requires=required,
    extras_require=extras
    )
