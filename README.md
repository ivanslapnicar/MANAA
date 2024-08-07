# Matrix Algorithms in Noncommutative Associatve Algebras (MANAA)

This repository contains software which accompanies and enhances research and publications within the project "Matrix Algorithms in Noncommutative Associatve Algebras" $\to$ [project's website](http://manaa.fesb.unist.hr/).

Notebooks are written in [Julia](https://julialang.org) using [Pluto.jl](https://github.com/fonsp/Pluto.jl). The `*.jl` files are in the directory `/src`. The `*.html` files are in the directory `/docs`.

## Contents

* [Fast.jl](https://github.com/ivanslapnicar/MANAA/) and [Fast.html](https://ivanslapnicar.github.io/MANAA/Fast.html) - files accompanying and enhancing the paper: Nevena Jakovčević Stor and Ivan Slapničar, _Fast determinants and inverses of arrowhead and diagonal-plus-rank-one matrices over associative fields_, Axioms 2024, 13(6), Article ID 409.

* [ED_Arrow.jl](https://github.com/ivanslapnicar/MANAA/), [ED_Arrow.html](https://ivanslapnicar.github.io/MANAA/ED_Arrow.html), [ED_DPRk.jl](https://github.com/ivanslapnicar/MANAA/), [ED_DPRk.html](https://ivanslapnicar.github.io/MANAA/ED_DPRk.html), [Plotting.jl](https://github.com/ivanslapnicar/MANAA/), and [Plotting.html](https://ivanslapnicar.github.io/MANAA/Plotting.html) - files accompanying and enhancing the paper: : Ivan Slapničar, Thaniporn Chaysri and Nevena Jakovčević Stor, _Fast Eigenvalue Decomposition of Arrowhead and Diagonal-Plus-Rank-k Matrices of Quaternions_, Mathematics 2024, 12(9), Article ID 1327.

* [Osijek_2024.jl](https://github.com/ivanslapnicar/MANAA/), [Osijek_2024.html](https://ivanslapnicar.github.io/MANAA/ED_Arrow.html), [Sevilla_2024.jl](https://github.com/ivanslapnicar/MANAA/), and [Sevilla_2024.html](https://ivanslapnicar.github.io/MANAA/ED_DPRk.html), are presentations at the 8th Croatian Mathematical Congress, Osijek, July 2-5, 2024, and 9th European Congress of Mathematics, Sevilla, July 15-19, 2024, respectively.

## Cloning the repository

You can clone the entire repository using `git` command:
```
git clone https://github.com/ivanslapnicar/MANAA
```
You can also download the repository as a zip file.

The repository is now located in the directory  `MANAA`. The notebooks are located in the directory `MANAA/src`

## Running notebooks on your computer

Install [Julia](https://julialang.org/downloads/). In Julia terminal run the commands
```
> using Pkg
> Pkg.add("Pluto")
> using Pluto
> Pluto.run()
```
This opens local Pluto server in your browser. Now you can choose the notebook and run it.
