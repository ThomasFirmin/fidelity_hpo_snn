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


from lave.optim_conv import ConvMNIST
from lave.objective import Objective
from load_dataset import DataGetter
from search_spaces import get_mnist_rate_lava_cnt

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
parser.add_argument("--save", type=str, default="fixed_lava_mnist")
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

datagetter = DataGetter(dataset_name, frames=47, datasize=data_size)


model = Objective(
    network=ConvMNIST,
    datagetter=datagetter,
    alpha_test=5,
    beta_test=0.03,
    gpu=gpu,
    weight_norm=True,
    max_epochs=80,
)

loss = Loss(
    objective=Maximizer("test"),
    record_time=record_time,
    mpi=mpi,
    kwargs_mode=True,
    threshold=1,
)(model)

# Decision variables
values = get_mnist_rate_lava_cnt(convert=False)
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
        0.9980304418,  # train_size
        47,  # frames
        80,  # epochs
        31,  # batch_size
        0.7945626945,  # rate_true
        0.0656207166,  # rate_false
        34,  # c1_filters
        19,  # c2_filters
        4,  # c1_k
        5,  # c2_k
        0.0335652889,  # learning_rate
        0.9156600437,  # mu
        0.9408352661,  # gamma
        0.0314037682,  # tau_grad
        0.5481066957,  # scale_grad
        6.0349168157,  # threshold
        0.1616626300,  # threshold_step
        0.1446785643,  # current_decay
        0.0791091826,  # voltage_decay
        0.1057611209,  # threshold_decay
        0.3211535073,  # refractory_decay
        0.1712105956,  # dropout
    ]
]


df = Default(sp, solutions, batch=30)

exp = Experiment(df, loss, stop, save=save_file)
exp.run()
