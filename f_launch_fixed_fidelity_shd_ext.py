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


from lave.optim_conv import FfSHDExtended
from lave.objective import Objective
from load_dataset import DataGetter
from search_spaces import get_shd_lava_cnt_ext

from zellij.core import (
    Loss,
    Experiment,
    Maximizer,
    MixedSearchspace,
    Time,
)

from zellij.strategies.mixed import Default

import torch
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=None)
parser.add_argument("--time", type=int, default=3600)
parser.add_argument("--dataset", type=str, default="SHD")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="fixed_lava_shd")
parser.add_argument("--gpu_per_node", type=int, default=1)

parser.set_defaults(gpu=True, record_time=True)

args = parser.parse_args()
data_size = args.data
dataset_name = args.dataset
time = args.time
mpi = args.mpi
gpu = args.gpu
mock = args.mock
record_time = args.record_time
save_file = args.save
gpu_per_node = args.gpu_per_node

datagetter = DataGetter(dataset_name, frames=262, datasize=data_size)

model = Objective(
    network=FfSHDExtended,
    datagetter=datagetter,
    alpha_test=10,
    beta_test=0.1,
    gpu=gpu,
    weight_norm=True,
    optimizer="ADAM",
    max_epochs=100,
)

loss = Loss(
    objective=Maximizer("test"),
    record_time=record_time,
    mpi=mpi,
    kwargs_mode=True,
    threshold=1,
)(model)

# Decision variables
values = get_shd_lava_cnt_ext(convert=False)
sp = MixedSearchspace(values)

if gpu:
    loss.model.device = f"cuda:{loss.rank%gpu_per_node}"  # 2 GPUs per node

path = f"/home/thfirmin/ext_shd/torch_extension_shd_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Time(time)


solutions = [
    [
        0.9879427711,  # train_size
        373,  # frames
        100,  # epochs
        28,  # batch_size
        0.8237414656,  # rate_true
        0.0188520025,  # rate_false
        481,  # f1_n
        400,  # f2_n
        500,  # f3_n
        0.0259231432,  # learning_rate
        0.8687233651,  # beta_1
        0.9444588625,  # beta_2
        0.0131974521,  # tau_grad
        0.9436667979,  # scale_grad
        2.4852245539,  # threshold_f1
        0.3792161439,  # threshold_step_f1
        0.3679639858,  # current_decay_f1
        0.1413793084,  # voltage_decay_f1
        0.3894867912,  # threshold_decay_f1
        0.0865880723,  # refractory_decay_f1
        0.0279080650,  # dropout_f1
        5.8593632899,  # threshold_f2
        0.3958481851,  # threshold_step_f2
        0.2801800914,  # current_decay_f2
        0.1922933855,  # voltage_decay_f2
        0.2149475221,  # threshold_decay_f2
        0.4423515499,  # refractory_decay_f2
        0.1565154885,  # dropout_f2
        4.3483422711,  # threshold_f3
        0.2534076046,  # threshold_step_f3
        0.4500735658,  # current_decay_f3
        0.0868223478,  # voltage_decay_f3
        0.2703424189,  # threshold_decay_f3
        0.4997995170,  # refractory_decay_f3
        0.1656216312,  # dropout_f3
        7.0753962269,  # threshold_o
        0.1272141511,  # threshold_step_o
        0.4130635381,  # current_decay_o
        0.0396111344,  # voltage_decay_o
        0.3994000690,  # threshold_decay_o
        0.1230263671,  # refractory_decay_o
    ]
]


df = Default(sp, solutions, batch=32)

exp = Experiment(df, loss, stop, save=save_file)
exp.run()
