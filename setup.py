#!/usr/bin/env python

from distutils.core import setup

setup(
    name='grf',
    version='0.1',
    description='Framework for making OpenTTD NewGRF files',
    author='dP',
    packages=['grf'],
    scripts=['bin/grftopy']
)
