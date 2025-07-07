# PONDS
**P**erturbed **O**bservables in **N**oisy/**N**onlinear **D**ynamics **S**ynthesiser

___

PONDS is a synthetic data generator build alongside [TOAD](https://toad-docs.pages.dev/). Its purpose is to provide a standardized way in generating and storing synthetic climate data used for benchmarking abrupt shift detection.

___

## Installation

Currently, the easiest option to install PONDS on a local machine is by cloning the repo and using pip

```
$ git clone https://github.com/tipmip-methods/ponds.git
$ cd <path>/<to>/<ponds>
$ pip install .
```

___

## Contribution

To contribute to this project, clone the repository locally (see above @ _Installation_) and install ponds in developer mode using

```
pip install -e .[dev]
```

This makes sure, that additional packages are installed, which are helpfull or necessary when working on the project.