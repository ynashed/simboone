### Install

Tested on Ubuntu 20.04 + Julia Version 1.5.3

### Dependencies

- `Zygote` package for automatic differentiation
- `HDF5` for file I/O
- `ArgParse` for parsing command line arguments when running as script

### Usage

```
julia julia/SimIonization.jl \
--data_file path_to_h5_file \
--lr 0.001 \
--sample_id 0
```

  - `--data_file` Path to `.h5` data file
  - `--sample_id` Every `.h5` file has many samples. Pick one of them to test.
  - `--num_step` Number of optimization steps.
  - `--lr` Learning rate.
