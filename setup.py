#!/usr/bin/env python

from setuptools import setup, find_packages

install_requires = [
    'nml==0.6.1',
    'numpy==1.19.5',
    'Pillow==8.4.0',
    'spectra==0.0.11',
]

setup(
    name='grf',
    version='0.2.0',
    description='Framework for making OpenTTD NewGRF files',
    author='dP',
    packages=find_packages(include=['grf']),
    scripts=['bin/grftopy'],
    install_requires=install_requires,
    python_requires=">=3.6.9",
)
