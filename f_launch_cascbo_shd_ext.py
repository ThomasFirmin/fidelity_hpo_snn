# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-16T19:08:43+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from lave.optim_conv import FfSHDExtended
from lave.objective import Objective
from lave.early_stoping import NonSpikingPercentageStopping
from load_dataset import DataGetter
from search_spaces import get_shd_lava_cnt_ext

from zellij.core import (
    Loss,
    Experiment,
    Maximizer,
    DoNothing,
    UnitSearchspace,
    Time,
)

from zellij.strategies.continuous import CASCBOI
from zellij.strategies.tools.turbo_state import ICTurboState, SkewedBell

import torch
import numpy as np
import os

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=None)
parser.add_argument("--calls", type=int, default=100000)
parser.add_argument("--dataset", type=str, default="SHD")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="test_casbo_shd_ext")
parser.add_argument("--gpu_per_node", type=int, default=1)

parser.set_defaults(gpu=True, record_time=True)

args = parser.parse_args()
data_size = args.data
dataset_name = args.dataset
calls = args.calls
mpi = args.mpi
gpu = args.gpu
record_time = args.record_time
save_file = args.save
gpu_per_node = args.gpu_per_node

datagetter = DataGetter(dataset_name, frames=25, datasize=data_size)


model = Objective(
    network=FfSHDExtended,
    datagetter=datagetter,
    alpha_test=10,
    beta_test=0.1,
    gpu=gpu,
    early_stopping=NonSpikingPercentageStopping("outpt", 10, 0.1),
    weight_norm=True,
    optimizer="ADAM",
    max_epochs=35,
    start_valid=0.85,
)

loss = Loss(
    objective=Maximizer("test"),
    secondary=[DoNothing("approximate_cost")],
    constraint=["constraint_0"],
    record_time=record_time,
    mpi=mpi,
    kwargs_mode=True,
    threshold=1,
)(model)

# Decision variables
values = get_shd_lava_cnt_ext()
sp = UnitSearchspace(values)

batch_size = 30
time = 50400
tstate = ICTurboState(sp.size, batch_size, torch.ones(1) * torch.inf)
temperature = SkewedBell(1.9, 1000, 12)

if gpu:
    loss.model.device = f"cuda:{loss.rank%gpu_per_node}"  # 2 GPUs per node

path = f"/home/thfirmin/ext_shd/torch_extension_shd_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Time(time)

covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5, ard_num_dims=sp.size, lengthscale_constraint=Interval(0.005, 4.0)
    )
)
noise_constraint = Interval(1e-8, 1e-3)


def ubound_evolv(alpha):
    if alpha < 0.10:
        return 0.05 + alpha * 0.95 / 0.10
    else:
        return 1.0


if loss.rank != 0:
    bo = CASCBOI(
        sp,
        tstate,
        batch_size=batch_size,
        budget=time,
        temperature=temperature,
        initial_size=1005,
        gpu=False,
        covar_module=covar_module,
        noise_constraint=noise_constraint,
        cholesky_size=500,
        beam=10000,
        time_budget=True,
        fixed_ubound=[0],
        ubound_evolv=ubound_evolv,
    )
else:
    bo = CASCBOI(
        sp,
        tstate,
        batch_size=batch_size,
        budget=time,
        temperature=temperature,
        initial_size=1005,
        gpu="cuda:0",
        covar_module=covar_module,
        noise_constraint=noise_constraint,
        cholesky_size=500,
        beam=10000,
        time_budget=True,
        fixed_ubound=[0],
        ubound_evolv=ubound_evolv,
    )

exp = Experiment(bo, loss, stop, save="test_casbo_shd_ext")
exp.run()
