# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:03:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin

from lave.optim_conv import ConvMNISTExtended
from lave.objective import Objective
from load_dataset import DataGetter
from search_spaces import get_mnist_rate_lava_cnt_extended

from zellij.core import (
    Loss,
    Experiment,
    Maximizer,
    MixedSearchspace,
    Time,
)

from zellij.strategies.mixed import Default

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=None)
parser.add_argument("--time", type=int, default=3600)
parser.add_argument("--dataset", type=str, default="NMNIST")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="fixed_lava_nmnist_ext")
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

datagetter = DataGetter(dataset_name, frames=25, datasize=data_size)

model = Objective(
    network=ConvMNISTExtended,
    datagetter=datagetter,
    alpha_test=3,
    beta_test=0.03,
    gpu=gpu,
    weight_norm=True,
    max_epochs=24,
)

loss = Loss(
    objective=Maximizer("test"),
    record_time=record_time,
    mpi=mpi,
    kwargs_mode=True,
    threshold=1,
)(model)

# Decision variables
values = get_mnist_rate_lava_cnt_extended(convert=False)
sp = MixedSearchspace(values)

if gpu:
    loss.model.device = f"cuda:{loss.rank%gpu_per_node}"  # 2 GPUs per node

path = f"/home/thfirmin/ext_nmnist/torch_extension_nmnist_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Time(time)

solutions = [
    [
        0.9974494876,  # train_size
        36,  # frames
        24,  # epochs
        120,  # batch_size
        0.7561072398,  # rate_true
        0.0101287715,  # rate_false
        40,  # c1_filters
        34,  # c2_filters
        4,  # c1_k
        5,  # c2_k
        0.0569213220,  # learning_rate
        0.8333406991,  # mu
        0.9869876597,  # gamma
        0.0065710443,  # tau_grad
        0.3181635989,  # scale_grad
        0.6141398946,  # threshold_c1
        0.1612242074,  # threshold_step_c1
        0.1205309654,  # current_decay_c1
        0.0144154015,  # voltage_decay_c1
        0.0101626449,  # threshold_decay_c1
        0.1593337442,  # refractory_decay_c1
        5.0541203552,  # threshold_c2
        0.2588499949,  # threshold_step_c2
        0.1635639064,  # current_decay_c2
        0.1449366593,  # voltage_decay_c2
        0.1738299631,  # threshold_decay_c2
        0.1681087564,  # refractory_decay_c2
        7.8059867138,  # threshold_a1
        0.3625995726,  # threshold_step_a1
        0.3943456429,  # current_decay_a1
        0.0392924613,  # voltage_decay_a1
        0.1786513286,  # threshold_decay_a1
        0.4677194853,  # refractory_decay_a1
        3.1575873755,  # threshold_a2
        0.3676396714,  # threshold_step_a2
        0.1724006533,  # current_decay_a2
        0.0796684456,  # voltage_decay_a2
        0.0223828225,  # threshold_decay_a2
        0.1806588079,  # refractory_decay_a2
        4.3996512242,  # threshold_o
        0.3537850813,  # threshold_step_o
        0.4544485331,  # current_decay_o
        0.1120911911,  # voltage_decay_o
        0.2669884541,  # threshold_decay_o
        0.2498067699,  # refractory_decay_o
        0.0276927324,  # dropout
    ]
]


df = Default(sp, solutions, batch=32)

exp = Experiment(df, loss, stop, save=save_file)
exp.run()
