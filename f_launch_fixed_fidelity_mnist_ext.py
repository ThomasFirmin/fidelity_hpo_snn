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
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="fixed_lava_mnist_ext")
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
values = get_mnist_rate_lava_cnt_extended(convert=False)
sp = MixedSearchspace(values)

if gpu:
    loss.model.device = f"cuda:{loss.rank%gpu_per_node}"  # 2 GPUs per node

path = f"/home/thfirmin/ext_mnist/torch_extension_mnist_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Time(time)

solutions = [
    [
        0.971172773755982,  # train_size
        43,  # frames
        100,  # epochs
        80,  # batch_size
        0.623488464002207,  # rate_true
        0.0366291942562091,  # rate_false
        18,  # c1_filters
        31,  # c2_filters
        5,  # c1_k
        5,  # c2_k
        0.0112743646279059,  # learning_rate
        0.88005572466665,  # mu
        0.964986026334556,  # gamma
        0.00730149775459061,  # tau_grad
        0.846952153942629,  # scale_grad
        0.681577224764828,  # threshold_c1
        0.337248281573376,  # threshold_step_c1
        0.298983907124562,  # current_decay_c1
        0.123770508695436,  # voltage_decay_c1
        0.253628156151655,  # threshold_decay_c1
        0.0898471552384332,  # refractory_decay_c1
        6.84605629153553,  # threshold_c2
        0.246722001726671,  # threshold_step_c2
        0.34217187818028,  # current_decay_c2
        0.0213965649911448,  # voltage_decay_c2
        0.246573855976435,  # threshold_decay_c2
        0.167023496062172,  # refractory_decay_c2
        3.24320311313427,  # threshold_a1
        0.23634293961265,  # threshold_step_a1
        0.133010378638851,  # current_decay_a1
        0.0694482025422336,  # voltage_decay_a1
        0.419602116647094,  # threshold_decay_a1
        0.227511145011068,  # refractory_decay_a1
        6.40172821112883,  # threshold_a2
        0.375006683454248,  # threshold_step_a2
        0.25786126368955,  # current_decay_a2
        0.114645582037045,  # voltage_decay_a2
        0.0546457065742944,  # threshold_decay_a2
        0.207488303280036,  # refractory_decay_a2
        5.91144750908498,  # threshold_o
        0.131812992662951,  # threshold_step_o
        0.175704970251563,  # current_decay_o
        0.146013225909492,  # voltage_decay_o
        0.457669655882147,  # threshold_decay_o
        0.0673303418338418,  # refractory_decay_o
        0.0417266835135585,  # dropout
    ]
]


df = Default(sp, solutions, batch=32)

exp = Experiment(df, loss, stop, save=save_file)
exp.run()
