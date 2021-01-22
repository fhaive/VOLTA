# README

## Current standards

- most code is adapted to work with edge-lists or adjacency matrices
- if graph object is used, it is build on top of Networkx (igraph options/ conversions need to be implemented)
- main functions provide option to run async (on one core) and/or as multiprocesses or as a single thread on one core
- currently we are just writing functions - classes can be added later if desired
- see excel for details

### What is this repository for?

- Quick summary
- Version
- [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

## How do I get set up?

### Prerequisites

- Install https://github.com/Jacobe2169/GMatch4py as instructed
- Install python-igraph . Based on your system there may be specific Prerequisites to fulfill. Refer to the igraph instructions for these.
- For windows users please install the optional requirements as instructed here: https://github.com/GiulioRossetti/cdlib , else some of the community algorithms will not be available

### Install

- download repository (wget ... when public repo)
- cd graphalgorithms
- pip3 install .

Depending on your system you may receive an "invalid command bdist_wheel" error. You can fix this by installing the wheel package.
pip3 install wheel

## Usage

- import graphAlgorithms as ga 

### Contribution guidelines

- Writing tests
- Code review
- Other guidelines

### Who do I talk to?

- Repo owner or admin
- Other community or team contact
