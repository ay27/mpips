fork from <https://github.com/ifreyr/mpips>

## Changed:
1. Change from python2 to python3.
2. Add launcher.py, make it more easy to run the project.

MpiPs
=====

A parameter server framework based on MPI4PY.

## Intro

* mpips: the main package
* alg/lr.py: Logistic Regression

## Dependence

* numpy
* openmpi or mpich
* mpi4py

## Run
Just run the launcher.py.
Note that the mpi processes must > ps num which define in lr.py.