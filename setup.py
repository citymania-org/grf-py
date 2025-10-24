#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path

root_path = Path(__file__).parent

install_requires = [
    'nml==0.8.0',
    'numpy==2.3.4',
    'Pillow==12.0.0',
    'typeguard==4.4.4',
]

setup(
    name='grf',
    setuptools_git_versioning={
        "enabled": True,
        "version_file": root_path / "VERSION",
    },
    description='Framework for making OpenTTD NewGRF files',
    author='dP',
    packages=find_packages(include=['grf', 'grf.*']),
    entry_points={
        'console_scripts': [
            'grftopy = grf.decompile:main',
        ]
    },
    install_requires=install_requires,
    python_requires=">=3.12.10",
    setup_requires=["setuptools-git-versioning>=2"],
)
