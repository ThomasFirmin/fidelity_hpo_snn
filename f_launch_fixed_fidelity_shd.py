# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:03:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:03:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from lave.optim_conv import FfSHD
from lave.objective import Objective
from load_dataset import DataGetter
from search_spaces import get_shd_lava_cnt

from zellij.core import (
    Loss,
    MockModel,
    Experiment,
    Maximizer,
    MixedSearchspace,
    Threshold,
)

from zellij.strategies.mixed import Default

import torch
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=None)
parser.add_argument("--calls", type=int, default=100000)
parser.add_argument("--dataset", type=str, default="SHD")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="fixed_lava_shd")
parser.add_argument("--gpu_per_node", type=int, default=1)

parser.set_defaults(gpu=True, record_time=True)

args = parser.parse_args()
data_size = args.data
dataset_name = args.dataset
calls = args.calls
mpi = args.mpi
gpu = args.gpu
mock = args.mock
record_time = args.record_time
save_file = args.save
gpu_per_node = args.gpu_per_node

datagetter = DataGetter(dataset_name, frames=262, datasize=data_size)

model = Objective(
    network=FfSHD,
    datagetter=datagetter,
    alpha_test=10,
    beta_test=0.1,
    gpu=gpu,
    weight_norm=True,
    optimizer="ADAM",
    max_epochs=100,
)

if mock:
    loss = Loss(
        objective=Maximizer("test"),
        record_time=record_time,
        mpi=mpi,
        kwargs_mode=True,
    )(
        MockModel(
            {
                "valid": lambda *arg, **kwarg: np.random.random(),
                "train": lambda *arg, **kwarg: np.random.random(),
            }
        )
    )
else:
    loss = Loss(
        objective=Maximizer("test"),
        record_time=record_time,
        mpi=mpi,
        kwargs_mode=True,
        threshold=1,
    )(model)

# Decision variables
values = get_shd_lava_cnt(convert=False)
sp = MixedSearchspace(values)

if gpu:
    loss.model.device = f"cuda:{loss.rank%gpu_per_node}"  # 2 GPUs per node

path = f"/home/thfirmin/ext_shd/torch_extension_shd_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Threshold(loss, "calls", calls)


solutions = [
    [
        0.9547499466,  # train_size
        262,  # frames
        100,  # epochs
        37,  # batch_size
        0.5782001493,  # rate_true
        0.0254826142,  # rate_false
        521,  # f1_n
        768,  # f2_n
        453,  # f3_n
        0.0781437504,  # learning_rate
        0.8974553643,  # beta_1
        0.8161819382,  # beta_2
        0.0616564872,  # tau_grad
        0.5518833782,  # scale_grad
        6.0555873333,  # threshold
        0.4660218715,  # threshold_step
        0.2557697763,  # current_decay
        0.1977035861,  # voltage_decay
        0.3370926349,  # threshold_decay
        0.0850919986,  # refractory_decay
        0.0338640149,  # dropout
    ]
]


df = Default(sp, solutions, batch=32)

exp = Experiment(df, loss, stop, save=save_file)
exp.run()
