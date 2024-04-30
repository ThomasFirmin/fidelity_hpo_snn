# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-20T12:35:18+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-12T14:35:42+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import torch
import numpy as np
import tonic
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import torchvision


# From BindsNet
def poisson(
    datum: torch.Tensor,
    time: int,
) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    mask = datum != 0
    rate[mask] = 1 / datum[mask] * 1000

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate, validate_args=False)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, mask] += (intervals[:, mask] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)


class DataGetter:
    def __init__(
        self,
        name: str,
        datasize=None,
        frames: int = 100,
        split_size: float = 0.2,
        random_state: int = 2903,
    ) -> None:
        self.name = name
        self.frames = frames
        self.datasize = datasize

        if self.name == "MNIST":
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Lambda(lambda x: x * 510),
                    v2.Lambda(lambda x: poisson(x, self.frames)),
                ]
            )

            self.train_dataset = torchvision.datasets.MNIST(
                root="data",
                download=True,
                train=True,
                transform=self.transform,
            )

            if self.datasize:
                data_idx = range(self.datasize)
            else:
                data_idx = range(len(self.train_dataset))

            train_idx, test_idx, _, _ = train_test_split(
                data_idx,
                self.train_dataset.targets,
                test_size=split_size,
                shuffle=True,
                random_state=random_state,
                stratify=self.train_dataset.targets,
            )

            self.train = torch.utils.data.Subset(self.train_dataset, train_idx)
            self.test = torch.utils.data.Subset(self.train_dataset, test_idx)

            self.valid = torchvision.datasets.MNIST(
                root="data",
                download=True,
                train=False,
                transform=self.transform,
            )

            self.encode = True
            self.input_features = 784
            self.input_shape = (1, 28, 28)
            self.classes = 10
            self.time_step = self.frames

        elif self.name == "NMNIST":
            self.transform = tonic.transforms.Compose(
                [
                    tonic.transforms.Downsample(spatial_factor=28 / 34),
                    tonic.transforms.ToFrame(
                        sensor_size=(28, 28, 2), n_time_bins=self.frames
                    ),
                    tonic.transforms.NumpyAsType(bool),
                    tonic.transforms.NumpyAsType(int),
                    torch.Tensor,
                ]
            )

            self.train_dataset = tonic.datasets.NMNIST(
                save_to="data",
                train=True,
                transform=self.transform,
            )

            if self.datasize:
                data_idx = range(self.datasize)
            else:
                data_idx = range(len(self.train_dataset))

            train_idx, test_idx, _, _ = train_test_split(
                data_idx,
                self.train_dataset.targets,
                test_size=split_size,
                shuffle=True,
                random_state=random_state,
                stratify=self.train_dataset.targets,
            )

            self.train = torch.utils.data.Subset(self.train_dataset, train_idx)
            self.test = torch.utils.data.Subset(self.train_dataset, test_idx)

            self.valid = tonic.datasets.NMNIST(
                save_to="data",
                train=False,
                transform=self.transform,
            )

            self.encode = False
            self.input_features = 1568
            self.input_shape = (2, 28, 28)
            self.classes = 10
            self.time_step = self.frames

        elif self.name == "GESTURE":
            self.transform = tonic.transforms.Compose(
                [
                    tonic.transforms.Denoise(filter_time=20000),
                    tonic.transforms.CropTime(max=2000000),
                    tonic.transforms.ToFrame(
                        sensor_size=(128, 128, 2), n_time_bins=self.frames
                    ),
                    tonic.transforms.NumpyAsType(bool),
                    tonic.transforms.NumpyAsType(int),
                    torch.Tensor,
                ]
            )

            self.train_dataset = tonic.datasets.DVSGesture(
                save_to="data",
                train=True,
                transform=self.transform,
            )

            if self.datasize:
                data_idx = range(self.datasize)
            else:
                data_idx = range(len(self.train_dataset))

            train_idx, test_idx, _, _ = train_test_split(
                data_idx,
                self.train_dataset.targets,
                test_size=split_size,
                shuffle=True,
                random_state=random_state,
                stratify=self.train_dataset.targets,
            )

            self.train = torch.utils.data.Subset(self.train_dataset, train_idx)
            self.test = torch.utils.data.Subset(self.train_dataset, test_idx)

            self.valid = tonic.datasets.DVSGesture(
                save_to="data",
                train=False,
                transform=self.transform,
            )

            self.encode = False
            self.input_features = 32768
            self.input_shape = (2, 128, 128)
            self.classes = 11
            self.time_step = self.frames

        elif self.name == "SHD":

            self.transform = tonic.transforms.Compose(
                [
                    tonic.transforms.ToFrame(
                        sensor_size=(700, 1, 1),
                        n_time_bins=self.frames,
                    ),
                    tonic.transforms.NumpyAsType(bool),
                    tonic.transforms.NumpyAsType(int),
                    torch.Tensor,
                    lambda x: x.squeeze(),
                ]
            )

            self.train_dataset = tonic.datasets.SHD(
                save_to="data",
                train=True,
                transform=self.transform,
            )

            if self.datasize:
                data_idx = range(self.datasize)
                labels = np.load("shd_labels.npy")[: self.datasize]
            else:
                data_idx = range(len(self.train_dataset))
                labels = np.load("shd_labels.npy")

            train_idx, test_idx, _, _ = train_test_split(
                data_idx,
                labels,
                test_size=split_size,
                shuffle=True,
                random_state=random_state,
                stratify=labels,
            )

            self.train = torch.utils.data.Subset(self.train_dataset, train_idx)
            self.test = torch.utils.data.Subset(self.train_dataset, test_idx)

            self.valid = tonic.datasets.SHD(
                save_to="data",
                train=False,
                transform=self.transform,
            )

            self.encode = False
            self.input_features = 700
            self.input_shape = (1, 700)
            self.classes = 20
            self.time_step = self.frames

        elif self.name == "SHDWRONG":

            self.transform = tonic.transforms.Compose(
                [
                    tonic.transforms.ToFrame(
                        sensor_size=(700, 1, 1),
                        n_time_bins=self.frames,
                    ),
                    tonic.transforms.NumpyAsType(bool),
                    tonic.transforms.NumpyAsType(int),
                    torch.Tensor,
                    lambda x: x.squeeze(),
                ]
            )

            self.train = tonic.datasets.SHD(
                save_to="data",
                train=True,
                transform=self.transform,
            )
            self.test = tonic.datasets.SHD(
                save_to="data",
                train=False,
                transform=self.transform,
            )
            self.valid = None

            self.encode = False
            self.input_features = 700
            self.input_shape = (1, 700)
            self.classes = 20
            self.time_step = self.frames

        elif self.name == "SSC":

            self.transform = tonic.transforms.Compose(
                [
                    tonic.transforms.Downsample(spatial_factor=140 / 700),
                    tonic.transforms.ToFrame(
                        sensor_size=(140, 1, 1),
                        n_time_bins=self.frames,
                    ),
                    tonic.transforms.NumpyAsType(bool),
                    tonic.transforms.NumpyAsType(int),
                    torch.Tensor,
                    lambda x: x.squeeze(),
                ]
            )

            self.train = tonic.datasets.SSC(
                save_to="data",
                split="train",
                transform=self.transform,
            )

            self.test = tonic.datasets.SSC(
                save_to="data",
                split="valid",
                transform=self.transform,
            )

            self.valid = tonic.datasets.SSC(
                save_to="data",
                split="test",
                transform=self.transform,
            )

            self.encode = False
            self.input_features = 140
            self.input_shape = (1, 140)
            self.classes = 35
            self.time_step = self.frames
        else:
            raise ValueError(f"Unknown dataset name, got {self.name}")

    def get_data(self, frames: int):
        self.frames = frames
        self.time_step = frames

        if self.name == "MNIST":
            self.transform.transforms[-1] = v2.Lambda(lambda x: poisson(x, self.frames))
        elif self.name == "NMNIST":
            self.transform.transforms[1] = tonic.transforms.ToFrame(
                sensor_size=(28, 28, 2), n_time_bins=frames
            )
        elif self.name == "GESTURE":
            self.transform.transforms[2] = tonic.transforms.ToFrame(
                sensor_size=(128, 128, 2), n_time_bins=frames
            )
        elif self.name == "SHD" or self.name == "SHDWRONG":
            self.transform.transforms[0] = tonic.transforms.ToFrame(
                sensor_size=(700, 1, 1), n_time_bins=frames
            )
        elif self.name == "SSC":
            self.transform.transforms[1] = tonic.transforms.ToFrame(
                sensor_size=(140, 1, 1), n_time_bins=frames
            )
        else:
            raise ValueError(f"Unknown dataset name, got {self.name}")

        return (
            self.train,
            self.test,
            self.valid,
            self.encode,
            self.input_features,
            self.input_shape,
            self.classes,
            self.time_step,
        )
