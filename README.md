# Asynchronous Multi-fidelity Hyperparameter Optimization Of Spiking Neural Networks

Experiments are computed on Grid5000. Scripts are launched using `mpiexec` or `mpirun`.

## Installation

We did not face critical issues running Lava-DL and Bindsnet with `Pytorch v.2.1.0` and `CUDA v.18.1`.

### Lava and Lava-DL
Lava-DL must be installed by hand following instructions found [here](https://lava-nc.org/lava/notebooks/in_depth/tutorial01_installing_lava.html).

Compilation of the custom CUDA code is not thread safe. One can add following lines to their scripts to avoid issues:

```python
import os

path = f"<YOUR_FOLDER>/torch_extension_mnist_{<PROCESS_RANK>}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path
```

#### Locally installing network module
Locally install `Lave` containing networks and other tools concerning the SNN part using Lava-DL with:
```
pip install -e ./Lave
```

### Zellij

The [Zellij](https://github.com/ThomasFirmin/zellij/) version was freezed in this repo.
An OPENMPI distribution is necessaary, parallelization is made using `mpi4py`.

For the version used in these experiments:
```
$ pip install -e ./zellij
```

## Running scripts

There are 4 scripts for the main experiments:

- `experiment_1`:
  - `f_launch_cascbo_mnist_lava.py`
  - Dataset: MNIST
  - Hyperparameters: 22
  - Retrain best :
    - `f_launch_fixed_fidelity_mnist.py`
- `experiment_2`:
  - `f_launch_cascbo_mnist_lava_ext.py`
  - Dataset: MNIST
  - Hyperparameters: 46
  - Retrain best :
    - `f_launch_fixed_fidelity_mnist_ext.py`
- `experiment_3`:
  - `f_launch_cascbo_nmnist_lava.py`
  - Dataset: NMNIST
  - Hyperparameters: 22
  - Retrain best :
    - `f_launch_fixed_fidelity_nmnist.py`
- `experiment_4`:
  - `f_launch_cascbo_mnist_lava_ext.py`
  - Dataset: NMNIST
  - Hyperparameters: 46
  - Retrain best :
    - `f_launch_fixed_fidelity_nmnist_ext.py`
- `experiment_5`:
  - `f_launch_cascbo_shd.py`
  - Dataset: SHD
  - Hyperparameters: 21
  - Retrain best :
    - `f_launch_fixed_fidelity_shd.py`
- `experiment_2`:
  - `f_launch_cascbo_shd_ext.py`
  - Dataset: SHD
  - Hyperparameters: 42
  - Retrain best :
    - `f_launch_fixed_fidelity_shd_ext.py`

### Options
- `--dataset`: name of the dataset, choose between `MNIST, NMNIST, SHD`.
- `--time`: Duration of the experiment in seconds.
- `--mpi`: `{synchronous, asynchronous, flexible}`. Use `flexible` for these experiments.
- `--gpu`: If True use GPU.
- `--record_time`: Record evaluation time for all SNNs.
- `--gpu_per_node`: Deprecated. All GPUs must be isolated within their node. (One process per GPU)

### Search spaces

Search spaces are found in `search_spaces.py` and defined according to `Zellij`.

### Architectures

- The network architectures are in `./Lave/lib/lave/optim_conv.py`
- The objective function is in `./Lave/lib/lave/objective.py`. It defines the training, validation and testing loops.
- The early stopping is in `./Lave/lib/lave/early_stopping.py`.

### Execution with `mpirun`

```
mpiexec -machinefile <HOSTFILE> -rankfile <RANKFILE> -n 16 python3 <SCRIPT_NAME> --dataset MNIST --mpi flexible --gpu --time 50400
```

### Results

Results and plots of all 6 experiments are in the `experiment*` folders.
