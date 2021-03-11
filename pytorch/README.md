### Install

Tested on Ubuntu 18.04 + CUDA 10.0


The following commands will create Python virtualenv and install PyTorch 1.4.0 
```
python -m pip install --user virtualenv
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r pytorch/requirements.txt
```


### Usage


__Optimize Model Parameters__

```
python pytorch/invert_model.py \
--data_file path_to_h5_file \
--std 0.1 \
--opt adam \
--lr 0.0001 \
--sample_id 0 \
--num_step 10000 \
--test_case 2
```

  - `--data_file` Path to `.h5` data file
  - `--std` Standard deviation to add noise to ground truth parameters
  - `--sample_id` Every `.h5` file has many samples. Pick one of them to test.
  - `--num_step` Number of optmization steps.
  - `--print_step` Frequency to print loss.
  - `--test_case` We offer three wayss to initalize `_v_alpha` and `_v_beta`. `0`: Non-trainable fixed values. `1`: Uniformed sampled values between [0, 1). `2`: Normal distribution around a handcrafted value.
  - `--use_syn` Use synthetic input data (Uniformed sampled values between [0, 1) for E, x, DE/Dx)
  - `--opt` Choose optimizer (`adam` with `lr = 0.001` seems to work)
  - `--lr` Learning rate.
  - `--lr_schedule` Learning rate decay schedule. `lineardecay` seems to work.



__Optimize Input Data__

```
python pytorch/invert_data.py \
--data_file path_to_h5_file \
--scale 0.5 \
--opt adam \
--lr 0.01 \
--sample_id 0 \
--num_step 10000
```

- `--data_file` Path to `.h5` data file
- `--scale` Scaling factor to inject random noise into the ground truth x (as the initial value for optimization)
- `--sample_id` Every `.h5` file has many samples. Pick one of them to test.
- `--num_step` Number of optmization steps.
- `--print_step` Frequency to print loss.
- `--opt` Choose optimizer (`adam` with `lr = 0.01` seems to work)
- `--lr` Learning rate.
- `--lr_schedule` Learning rate decay schedule. `lineardecay` seems to work.