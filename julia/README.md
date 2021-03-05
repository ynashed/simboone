### Install

Tested on Ubuntu 20.04 + Julia Version 1.5.3

### Dependencies

So far no external dependencies or Julia packages. Julia Pro recommended.

### Usage

In Julia REPL, and from the `julia` subdirectory, run the following:

```
include("SimIonization.jl")
using .SimIonization

SimIonization.simulate(N=20000)
```
This will run the forward model with a `Nx3` randomly initialized matrix.
