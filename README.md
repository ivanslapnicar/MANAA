# MANAA
Notebooks for the project "Matrix Algorithms in Noncommutative Associatve Algebras"


This repository contains software which accompanies and enhances research and publications within the project "Matrix Algorithms in Noncommutative Associatve Algebras" $\to$ [project's website](http://manaa.fesb.unist.hr/).

Notebooks are written in [Julia](https://julialang.org) using [Pluto.jl](https://github.com/fonsp/Pluto.jl).


## Cloning the repository

You can clone the entire repository using `git` command:
```
git clone https://github.com/ivanslapnicar/MANAA
```
You can also download the repository as a zip file.

The repository is now located in the directory  `MANAA`. The notebooks are located in the directory `MANAA/src`

## Running the notebooks on `binder`

Choose the `.html` version of the notebook. In the botebook, 
press `Edit or run this notebook` button and choose `binder`. This will read all the necessary packages and start the notebook (within several minutes).

## Running notebooks on your computer

Install [Julia](https://julialang.org/downloads/). In Julia terminal run the commands
```
> using Pkg
> Pkg.add("Pluto")
> using Pluto
> Pluto.run()
```
This opens local Pluto server in your browser. Now you can choose the notebook and run it.