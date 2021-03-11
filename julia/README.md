## Install

Tested on Ubuntu 20.04 + Julia Version 1.5.3

## Dependencies

- `Zygote` package for automatic differentiation
- `HDF5` for file I/O
- `ArgParse` for parsing command line arguments when running as script

## Usage

### From the cloned git repo
```
julia julia/SimIonization/src/SimIonization.jl \
--data_file path_to_h5_file \
--lr 0.001 \
--sample_id 0
```

  - `--data_file` Path to `.h5` data file
  - `--sample_id` Every `.h5` file has many samples. Pick one of them to test.
  - `--num_step` Number of optimization steps.
  - `--lr` Learning rate.

### Docker/Singularity

#### Building a Docker Image
If docker is installed and configured properly all you need is to run the following (from the `julia/SimIonization` git cloned directory)

```
sudo scripts/build_docker.sh .
```
This will build a `simboone-<git_branch>:latest` docker image which you can push to dockerhub and/or use with Singularity.
```
sudo docker push <docker_username>/simboone-main:latest
```
There's already a prebuilt docker image under `ynashed/simboone-main:latest`

#### Running a Singularity Image on SDF
First create the `.sif` file by running
```
singularity pull docker://<docker_username>/simboone-main:latest
```
Again, a prebuilt Singularity image can be found on SDF under `/scratch/ynashed/singularity_images/`.
A test reconstruction can be run on SDF [login-node] by
```
singularity exec -B /sdf <path_to_sif_directory>/simboone-main_latest.sif julia <path_to_git_clone_root>/julia/SimIonization/src/SimIonization.jl --data_file <path_to_h5_file> --num_step 10000 --lr 0.1
```
