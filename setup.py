#!/usr/bin/env python

from distutils.core import setup


install_requires = [
    'nml==0.5.3',
    'numpy==1.22.0',
    'Pillow==8.4.0',
    'spectra==0.0.11',
]

setup(
    name='grf',
    version='0.1',
    description='Framework for making OpenTTD NewGRF files',
    author='dP',
    packages=['grf'],
    scripts=['bin/grftopy'],
    install_requires=install_requires,
    python_requires=">=3.8",
)
