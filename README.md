# grf-py

A Python framework for making NewGRF mod files for the OpenTTD game.

It aims to cover the whole cycle of NewGRF development, from resource generation and low-level GRF actions to convenient templating and compilation. Having at all in the one place, powered by one of the best programming languages helps to keep it simple but flexible, minimize repeating code and achieve great readability and maintanability of the code. As grf-py covers pretty much the whole range of GRF format capabilites its primary application are the most complex projects that benefit the most from the ease of development and maintenance while keeping full control over GRF performance. But in future, as grf-py gains more high-level classes it should make it pretty approachable for beginners as well, better than nml at least. Also, grf-py includes `grftopy` decompiler that can produce low-level grf-py code for a .grf file.It is a Python framework that aims to cover the whole cycle of NewGRF development, from resource generation and low-level GRF actions to convenient templating and compilation. Having at all in the one place, powered by one of the best programming languages helps to keep it simple but flexible, minimize repeating code and achieve great readability and maintanability of the code. As grf-py covers pretty much the whole range of GRF format capabilites its primary application are the most complex projects that benefit the most from the ease of development and maintenance while keeping full control over GRF performance. But in future, as grf-py gains more high-level classes it should make it pretty approachable for beginners as well, better than nml at least. Also, grf-py includes `grftopy` decompiler that can produce low-level grf-py code for a .grf file.

Some examples of GRFs with grf-py:
Simple examples: https://github.com/citymania-org/grf-py/tree/main/examples
Road vehicles: https://github.com/citymania-org/uk-dk-busses
Trains: https://github.com/citymania-org/robs-trains
Trees and procedural sprites: https://github.com/citymania-org/debug-trees-grf
Objects and low-level API: https://github.com/citymania-org/cmclient/alpine

## grf.py

Idea is to provide two levels of api:

1) Low level (Action*) - directly mimic the grf capabilites with python syntax.
[Some documentation of the low-level API classes](docs/low_level.md).
2) High level - Something convenient to use but still interoperable with low-level api.

## grftopy 

GRF decompiler into a low-level grf.py api.

A tool to inspect the grf files. It's main goal is to provite a readable representation for studying the inner workings or a grf and decompile resources. Produced code should be mostly functional but expect some bugs as it's not a primary goal. Also support for older grfs (container version 1 and grf format version < 8) is limited.

## Installation

### Installation with pip
`pip install grf`

### Installation from github on linux into a new virtualenv

```
git clone https://github.com/citymania-org/grf-py
cd grf-py
python3 -m venv grfenv
source grfenv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
python setup.py install
```

# Licensing

Currently grf-py is licensed under GPL v3 which basically means any GRF you make with it should be licensed under GPL as well. In part this is due to it importing nml library which is also licensed under GPL. But I do plan to change the lincense to more permissive in future and allow importing grf-py in closed source projects as well. So if you need a more permissive license or have some other suggestions feel free to contact me (dP) via issues or discord.

This also means that if you plan to make any contributions to grf-py please explicitly license them under some non-restrictive license so I won't have to undo them later when I decide to change the license. Preferably use public domain or CC-0, but BSD or MIT should also be fine.
