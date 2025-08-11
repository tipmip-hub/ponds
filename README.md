# PONDS
**P**erturbed **O**bservables in **N**oisy/**N**onlinear **D**ynamics **S**ynthesiser
___

**Table of Content**

1. [Installation](#installation)
2. [Purpose of PONDS](#purpose-of-ponds)
3. [Contribution](#contribution)


PONDS is a synthetic data generator build alongside [TOAD](https://toad-docs.pages.dev/). It provides a standardized way in generating and storing synthetic climate data used for benchmarking abrupt shift detection.

___

## Installation

Currently, the easiest option to install PONDS on a local machine is by cloning the repo and using pip

```
$ git clone https://github.com/tipmip-methods/ponds.git
$ cd ponds
$ pip install .
```
___

## Purpose of PONDS

1. Framework to implement non-linear time series generators into a climate data environment.
    - to figure out how certain non-linear events can look like in climate data

2. Phenomenological showcasing of abrupt shift detectors (like TOAD).
    - provide a climate dataset and test which kind of shifts the detector finds

3. Mimicking real world shift events.
    - create surrogates for known events to perform ensemble analysis

___

## Contribution

To contribute to this project, clone the repository locally (see above @ _Installation_) and install ponds in developer mode using

```
pip install -e .[dev]
```

This makes sure, that additional packages are installed, which are helpfull or necessary when working on the project.

