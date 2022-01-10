# grf-py
A Python framework for making NewGRFs for OpenTTD

Goal is to combine the convenience and power of Python with control and performance of nfo.

At this point it's mostly just a proof of concept, API, syntax and goals can be changed at any moment.


# grf.py - Idea is to provide two levels of api:

1) Low level (Aciton*) - directly mimic the grf capabilites with python syntax.
2) High level - Something conveniet to use but still interoperable with low-level api.

# grftopy - GRF decompiler into a low-level grf.py api

A tool to inspect the grf files. It's main goal is to provite a readable representation, not to be a fully functional decompiler. I.e. don't expect the generated py files to work and generate the grf back, even though it's not that hard to achieve it's not a priority atm.