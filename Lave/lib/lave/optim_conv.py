import torch

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
from lave.abstract_network import AbstractNetwork
import numpy as np


class ConvMNIST(AbstractNetwork):
    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        threshold=1.25,
        threshold_step=0,
        current_decay=0.25,
        voltage_decay=0.03,
        threshold_decay=0,
        refractory_decay=0,
        dropout=0.05,
        tau_grad=1,
        scale_grad=1,
        requires_grad=True,
        c1_filters=5,
        c1_k=12,
        c1_p=0,
        c1_d=1,
        c1_s=1,
        c2_filters=64,
        c2_k=5,
        c2_p=0,
        c2_d=1,
        c2_s=1,
        a1_k=2,
        a1_p=0,
        a1_s=1,
        a2_k=2,
        a2_p=0,
        a2_s=1,
        weight_norm=True,
    ):
        super(ConvMNIST, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params = {
            "threshold": threshold,
            "threshold_step": threshold_step,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "threshold_decay": threshold_decay,
            "refractory_decay": refractory_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": requires_grad,
        }
        neuron_params_no_grad = {
            "threshold": threshold,
            "threshold_step": threshold_step,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "threshold_decay": threshold_decay,
            "refractory_decay": refractory_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": False,
        }
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }

        c1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1
        )

        a1o_shape = np.floor((c1o_shape + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1)
        c2o_shape = np.floor((a1o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a2o_shape = np.floor((c2o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)

        print(f"SHAPE: ,{c1o_shape}, {a1o_shape}, {c2o_shape}, {a2o_shape}")

        c1 = slayer.block.alif.Conv(
            neuron_params_drop,
            in_features=int(self.inpt_shape[0]),
            out_features=int(c1_filters),
            kernel_size=c1_k,
            stride=c1_s,
            padding=c1_p,
            dilation=c1_d,
            delay=True,
            weight_norm=weight_norm,
        )
        # torch.nn.init.xavier_uniform(c1.synapse.weight)

        a1 = slayer.block.alif.Pool(
            neuron_params_no_grad,
            kernel_size=a1_k,
            stride=a1_s,
            padding=a1_p,
        )

        c2 = slayer.block.alif.Conv(
            neuron_params_drop,
            in_features=int(c1_filters),
            out_features=int(c2_filters),
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            dilation=c2_d,
            delay=True,
            weight_norm=weight_norm,
        )
        # torch.nn.init.xavier_uniform(c2.synapse.weight)

        a2 = slayer.block.alif.Pool(
            neuron_params_no_grad,
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
        )

        outpt = slayer.block.alif.Dense(
            neuron_params,
            (int(a2o_shape), int(a2o_shape), int(c2_filters)),
            self.n_classes,
            weight_norm=weight_norm,
        )
        # torch.nn.init.xavier_uniform(outpt.synapse.weight)

        self.recorders_train = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }

        self.register_layer("c1", c1)
        self.register_layer("a1", a1)
        self.register_layer("c2", c2)
        self.register_layer("a2", a2)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([c1, a1, c2, a2, outpt])

    def forward(self, spikes):
        super().forward(spikes)

        # print("input", spikes.shape)
        spikes = self.layers["c1"](spikes)
        self._update_recorders(spikes, "c1")
        # print("c1", spikes.shape)

        spikes = self.layers["a1"](spikes)
        self._update_recorders(spikes, "a1")
        # print("a1", spikes.shape)

        spikes = self.layers["c2"](spikes)
        self._update_recorders(spikes, "c2")
        # print("c2", spikes.shape)

        spikes = self.layers["a2"](spikes)
        self._update_recorders(spikes, "a2")
        # print("a2", spikes.shape)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(spikes, "outpt")
        # print(f"OUT : {self.out_spike}/{self.computed_image}={self.out_spike/self.computed_image}")

        return spikes


class ConvDVS(AbstractNetwork):
    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        threshold=1.25,
        threshold_step=0,
        current_decay=0.25,
        voltage_decay=0.03,
        threshold_decay=0,
        refractory_decay=0,
        dropout=0.05,
        tau_grad=1,
        scale_grad=1,
        requires_grad=True,
        c1_filters=16,
        c1_k=5,
        c1_p=2,
        c1_d=1,
        c1_s=1,
        c2_filters=32,
        c2_k=3,
        c2_p=1,
        c2_d=1,
        c2_s=1,
        a1_k=4,
        a1_p=0,
        a1_s=1,
        a2_k=2,
        a2_p=0,
        a2_s=1,
        a3_k=2,
        a3_p=0,
        a3_s=1,
        d1_n=512,
        weight_norm=True,
    ):
        super(ConvDVS, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params = {
            "threshold": threshold,
            "threshold_step": threshold_step,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "threshold_decay": threshold_decay,
            "refractory_decay": refractory_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": requires_grad,
        }
        neuron_params_no_grad = {
            "threshold": threshold,
            "threshold_step": threshold_step,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "threshold_decay": threshold_decay,
            "refractory_decay": refractory_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": False,
        }
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }

        a1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1
        )
        c1o_shape = np.floor((a1o_shape + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1)
        a2o_shape = np.floor((c1o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)
        c2o_shape = np.floor((a2o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a3o_shape = np.floor((c2o_shape + 2 * a3_p - (a3_k - 1) - 1) / a3_s + 1)

        a1 = slayer.block.alif.Pool(
            neuron_params_no_grad,
            kernel_size=a1_k,
            stride=a1_s,
            padding=a1_p,
        )

        c1 = slayer.block.alif.Conv(
            neuron_params_drop,
            in_features=int(self.inpt_shape[0]),
            out_features=int(c1_filters),
            kernel_size=c1_k,
            stride=c1_s,
            padding=c1_p,
            dilation=c1_d,
            delay=True,
            weight_norm=weight_norm,
        )

        a2 = slayer.block.alif.Pool(
            neuron_params_no_grad,
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
        )

        c2 = slayer.block.alif.Conv(
            neuron_params_drop,
            in_features=int(c1_filters),
            out_features=int(c2_filters),
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            dilation=c2_d,
            delay=True,
            weight_norm=weight_norm,
        )

        a3 = slayer.block.alif.Pool(
            neuron_params_no_grad,
            kernel_size=a3_k,
            stride=a3_s,
            padding=a3_p,
        )

        d1 = slayer.block.alif.Dense(
            neuron_params_drop,
            (int(a3o_shape), int(a3o_shape), int(c2_filters)),
            d1_n,
            weight_norm=weight_norm,
        )
        outpt = slayer.block.alif.Dense(
            neuron_params,
            d1_n,
            self.n_classes,
            weight_norm=weight_norm,
        )

        self.recorders_train = {
            "inpt": 0,
            "c1": 0,
            "c2": 0,
            "a1": 0,
            "a2": 0,
            "a3": 0,
            "d1": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "c1": 0,
            "c2": 0,
            "a1": 0,
            "a2": 0,
            "a3": 0,
            "d1": 0,
            "outpt": 0,
        }

        self.register_layer("c1", c1)
        self.register_layer("c2", c2)
        self.register_layer("a1", a1)
        self.register_layer("a2", a2)
        self.register_layer("a3", a3)
        self.register_layer("d1", d1)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([a1, c1, a2, c2, a3, d1, outpt])

    def forward(self, spikesIn):
        super().forward(spikesIn)

        # print(spikes.shape)
        spikes = self.layers["a1"](spikesIn)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "a1")
        # print("a1", spikes.shape)

        spikes = self.layers["c1"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "c1")
        # print("c1", spikes.shape)

        spikes = self.layers["a2"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "a2")
        # print("a2", spikes.shape)

        spikes = self.layers["c2"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "c2")
        # print("c2", spikes.shape)

        spikes = self.layers["a3"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "a3")
        # print("a3", spikes.shape)

        spikes = self.layers["d1"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "d1")
        # print("d1", spikes.shape)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(torch.sum(spikes).to(dtype=int, device="cpu"), "outpt")
        # print("outpt", spikes.shape)
        # print("__________________________")

        # print(f"OUT : {self.out_spike}/{self.computed_image}={self.out_spike/self.computed_image}")

        return spikes


class ConvMNISTExtended(AbstractNetwork):

    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        # Convolution 1
        threshold_c1=1.25,
        threshold_step_c1=0,
        current_decay_c1=0.25,
        voltage_decay_c1=0.03,
        threshold_decay_c1=0,
        refractory_decay_c1=0,
        dropout_c1=0.05,
        # Pooling 1
        threshold_a1=1.25,
        threshold_step_a1=0,
        current_decay_a1=0.25,
        voltage_decay_a1=0.03,
        threshold_decay_a1=0,
        refractory_decay_a1=0,
        # Convolution 2
        threshold_c2=1.25,
        threshold_step_c2=0,
        current_decay_c2=0.25,
        voltage_decay_c2=0.03,
        threshold_decay_c2=0,
        refractory_decay_c2=0,
        dropout_c2=0.05,
        # Pooling 2
        threshold_a2=1.25,
        threshold_step_a2=0,
        current_decay_a2=0.25,
        voltage_decay_a2=0.03,
        threshold_decay_a2=0,
        refractory_decay_a2=0,
        # Outputs
        threshold_o=1.25,
        threshold_step_o=0,
        current_decay_o=0.25,
        voltage_decay_o=0.03,
        threshold_decay_o=0,
        refractory_decay_o=0,
        # Others
        c1_filters=5,
        c1_k=12,
        c1_p=0,
        c1_d=1,
        c1_s=1,
        c2_filters=64,
        c2_k=5,
        c2_p=0,
        c2_d=1,
        c2_s=1,
        a1_k=2,
        a1_p=0,
        a1_s=1,
        a2_k=2,
        a2_p=0,
        a2_s=1,
        weight_norm=True,
        tau_grad=1,
        scale_grad=1,
    ):
        super(ConvMNISTExtended, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params_c1 = {
            "threshold": threshold_c1,
            "threshold_step": threshold_step_c1,
            "current_decay": current_decay_c1,
            "voltage_decay": voltage_decay_c1,
            "threshold_decay": threshold_decay_c1,
            "refractory_decay": refractory_decay_c1,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_c1),
            "requires_grad": True,
        }
        neuron_params_c2 = {
            "threshold": threshold_c2,
            "threshold_step": threshold_step_c2,
            "current_decay": current_decay_c2,
            "voltage_decay": voltage_decay_c2,
            "threshold_decay": threshold_decay_c2,
            "refractory_decay": refractory_decay_c2,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_c2),
            "requires_grad": True,
        }
        neuron_params_a1 = {
            "threshold": threshold_a1,
            "threshold_step": threshold_step_a1,
            "current_decay": current_decay_a1,
            "voltage_decay": voltage_decay_a1,
            "threshold_decay": threshold_decay_a1,
            "refractory_decay": refractory_decay_a1,
            "tau_grad": 1.0,
            "scale_grad": 1.0,
            "requires_grad": False,
        }
        neuron_params_a2 = {
            "threshold": threshold_a2,
            "threshold_step": threshold_step_a2,
            "current_decay": current_decay_a2,
            "voltage_decay": voltage_decay_a2,
            "threshold_decay": threshold_decay_a2,
            "refractory_decay": refractory_decay_a2,
            "tau_grad": 1.0,
            "scale_grad": 1.0,
            "requires_grad": False,
        }
        neuron_params_o = {
            "threshold": threshold_o,
            "threshold_step": threshold_step_o,
            "current_decay": current_decay_o,
            "voltage_decay": voltage_decay_o,
            "threshold_decay": threshold_decay_o,
            "refractory_decay": refractory_decay_o,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": True,
        }

        c1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1
        )

        a1o_shape = np.floor((c1o_shape + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1)
        c2o_shape = np.floor((a1o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a2o_shape = np.floor((c2o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)

        print(f"SHAPE: ,{c1o_shape}, {a1o_shape}, {c2o_shape}, {a2o_shape}")

        c1 = slayer.block.alif.Conv(
            neuron_params_c1,
            in_features=int(self.inpt_shape[0]),
            out_features=int(c1_filters),
            kernel_size=c1_k,
            stride=c1_s,
            padding=c1_p,
            dilation=c1_d,
            delay=True,
            weight_norm=weight_norm,
        )

        a1 = slayer.block.alif.Pool(
            neuron_params_a1,
            kernel_size=a1_k,
            stride=a1_s,
            padding=a1_p,
        )

        c2 = slayer.block.alif.Conv(
            neuron_params_c2,
            in_features=int(c1_filters),
            out_features=int(c2_filters),
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            dilation=c2_d,
            delay=True,
            weight_norm=weight_norm,
        )
        a2 = slayer.block.alif.Pool(
            neuron_params_a2,
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
        )

        outpt = slayer.block.alif.Dense(
            neuron_params_o,
            (int(a2o_shape), int(a2o_shape), int(c2_filters)),
            self.n_classes,
            weight_norm=weight_norm,
        )

        self.recorders_train = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }

        self.register_layer("c1", c1)
        self.register_layer("a1", a1)
        self.register_layer("c2", c2)
        self.register_layer("a2", a2)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([c1, a1, c2, a2, outpt])

    def forward(self, spikes):
        super().forward(spikes)

        # print("input", spikes.shape)
        spikes = self.layers["c1"](spikes)
        self._update_recorders(spikes, "c1")
        # print("c1", spikes.shape)

        spikes = self.layers["a1"](spikes)
        self._update_recorders(spikes, "a1")
        # print("a1", spikes.shape)

        spikes = self.layers["c2"](spikes)
        self._update_recorders(spikes, "c2")
        # print("c2", spikes.shape)

        spikes = self.layers["a2"](spikes)
        self._update_recorders(spikes, "a2")
        # print("a2", spikes.shape)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(spikes, "outpt")
        # print(f"OUT : {self.out_spike}/{self.computed_image}={self.out_spike/self.computed_image}")

        return spikes


class FfSHD(AbstractNetwork):
    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        threshold=1.25,
        threshold_step=0,
        current_decay=0.25,
        voltage_decay=0.03,
        threshold_decay=0,
        refractory_decay=0,
        dropout=0.05,
        tau_grad=1,
        scale_grad=1,
        requires_grad=True,
        f1_n=320,
        f2_n=160,
        f3_n=80,
        weight_norm=True,
    ):
        super(FfSHD, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params = {
            "threshold": threshold,
            "threshold_step": threshold_step,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "threshold_decay": threshold_decay,
            "refractory_decay": refractory_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": requires_grad,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }
        neuron_params_drop = {
            **neuron_params,
        }

        f1 = slayer.block.alif.Dense(
            neuron_params_drop,
            self.n_inpt,
            f1_n,
            weight_norm=weight_norm,
            delay=True,
        )

        f2 = slayer.block.alif.Dense(
            neuron_params_drop,
            f1_n,
            f2_n,
            weight_norm=weight_norm,
            delay=True,
        )

        f3 = slayer.block.alif.Dense(
            neuron_params_drop,
            f2_n,
            f3_n,
            weight_norm=weight_norm,
            delay=True,
        )

        outpt = slayer.block.alif.Dense(
            neuron_params,
            f3_n,
            self.n_classes,
            weight_norm=weight_norm,
        )

        self.recorders_train = {
            "inpt": 0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "outpt": 0,
        }

        self.register_layer("f1", f1)
        self.register_layer("f2", f2)
        self.register_layer("f3", f3)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([f1, f2, f3, outpt])

    def forward(self, spikes):
        super().forward(spikes)

        # print("input", spikes.shape)
        spikes = self.layers["f1"](spikes)
        self._update_recorders(spikes, "f1")
        # print("f1", spikes.shape)

        spikes = self.layers["f2"](spikes)
        self._update_recorders(spikes, "f2")
        # print("f2", spikes.shape)

        spikes = self.layers["f3"](spikes)
        self._update_recorders(spikes, "f3")
        # print("f3", spikes.shape)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(spikes, "outpt")
        # print(f"OUT : ", spikes.shape)

        return spikes


class FfSHDExtended(AbstractNetwork):

    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        # Layer 1
        f1_n=320,
        threshold_f1=1.25,
        threshold_step_f1=0,
        current_decay_f1=0.25,
        voltage_decay_f1=0.03,
        threshold_decay_f1=0,
        refractory_decay_f1=0,
        dropout_f1=0.05,
        # Layer 2
        f2_n=160,
        threshold_f2=1.25,
        threshold_step_f2=0,
        current_decay_f2=0.25,
        voltage_decay_f2=0.03,
        threshold_decay_f2=0,
        refractory_decay_f2=0,
        dropout_f2=0.05,
        # Layer 2
        f3_n=80,
        threshold_f3=1.25,
        threshold_step_f3=0,
        current_decay_f3=0.25,
        voltage_decay_f3=0.03,
        threshold_decay_f3=0,
        refractory_decay_f3=0,
        dropout_f3=0.05,
        # Output
        threshold_o=1.25,
        threshold_step_o=0,
        current_decay_o=0.25,
        voltage_decay_o=0.03,
        threshold_decay_o=0,
        refractory_decay_o=0,
        # Other
        weight_norm=True,
        tau_grad=1,
        scale_grad=1,
    ):
        super(FfSHDExtended, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params_f1 = {
            "threshold": threshold_f1,
            "threshold_step": threshold_step_f1,
            "current_decay": current_decay_f1,
            "voltage_decay": voltage_decay_f1,
            "threshold_decay": threshold_decay_f1,
            "refractory_decay": refractory_decay_f1,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_f1),
            "requires_grad": True,
        }
        neuron_params_f2 = {
            "threshold": threshold_f2,
            "threshold_step": threshold_step_f2,
            "current_decay": current_decay_f2,
            "voltage_decay": voltage_decay_f2,
            "threshold_decay": threshold_decay_f2,
            "refractory_decay": refractory_decay_f2,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_f2),
            "requires_grad": True,
        }
        neuron_params_f3 = {
            "threshold": threshold_f3,
            "threshold_step": threshold_step_f3,
            "current_decay": current_decay_f3,
            "voltage_decay": voltage_decay_f3,
            "threshold_decay": threshold_decay_f3,
            "refractory_decay": refractory_decay_f3,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_f3),
            "requires_grad": True,
        }
        neuron_params_o = {
            "threshold": threshold_o,
            "threshold_step": threshold_step_o,
            "current_decay": current_decay_o,
            "voltage_decay": voltage_decay_o,
            "threshold_decay": threshold_decay_o,
            "refractory_decay": refractory_decay_o,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": True,
        }

        f1 = slayer.block.alif.Dense(
            neuron_params_f1,
            self.n_inpt,
            f1_n,
            weight_norm=weight_norm,
            delay=True,
        )

        f2 = slayer.block.alif.Dense(
            neuron_params_f2,
            f1_n,
            f2_n,
            weight_norm=weight_norm,
            delay=True,
        )

        f3 = slayer.block.alif.Dense(
            neuron_params_f3,
            f2_n,
            f3_n,
            weight_norm=weight_norm,
            delay=True,
        )

        outpt = slayer.block.alif.Dense(
            neuron_params_o,
            f3_n,
            self.n_classes,
            weight_norm=weight_norm,
        )

        self.recorders_train = {
            "inpt": 0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "outpt": 0,
        }

        self.register_layer("f1", f1)
        self.register_layer("f2", f2)
        self.register_layer("f3", f3)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([f1, f2, f3, outpt])

    def forward(self, spikes):
        super().forward(spikes)

        # print("input", spikes.shape)
        spikes = self.layers["f1"](spikes)
        self._update_recorders(spikes, "f1")
        # print("f1", spikes.shape)

        spikes = self.layers["f2"](spikes)
        self._update_recorders(spikes, "f2")
        # print("f2", spikes.shape)

        spikes = self.layers["f3"](spikes)
        self._update_recorders(spikes, "f3")
        # print("f2", spikes.shape)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(spikes, "outpt")
        # print(f"OUT : ", spikes.shape)

        return spikes


class ConvSHDExtended(AbstractNetwork):

    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        # Conv 1
        c1_filters=16,
        c1_k=5,
        c1_p=0,
        c1_d=1,
        c1_s=4,
        threshold_c1=1.25,
        threshold_step_c1=0,
        current_decay_c1=0.25,
        voltage_decay_c1=0.03,
        threshold_decay_c1=0,
        refractory_decay_c1=0,
        dropout_c1=0.0,
        # Pool 1
        a1_k=4,
        a1_p=0,
        a1_s=1,
        threshold_a1=1.25,
        threshold_step_a1=0,
        current_decay_a1=0.25,
        voltage_decay_a1=0.03,
        threshold_decay_a1=0,
        refractory_decay_a1=0,
        # Conv 2
        c2_filters=16,
        c2_k=5,
        c2_p=0,
        c2_d=1,
        c2_s=2,
        threshold_c2=1.25,
        threshold_step_c2=0,
        current_decay_c2=0.25,
        voltage_decay_c2=0.03,
        threshold_decay_c2=0,
        refractory_decay_c2=0,
        dropout_c2=0.0,
        # Pool 1
        a2_k=4,
        a2_p=0,
        a2_s=1,
        threshold_a2=1.25,
        threshold_step_a2=0,
        current_decay_a2=0.25,
        voltage_decay_a2=0.03,
        threshold_decay_a2=0,
        refractory_decay_a2=0,
        # Output
        threshold_o=1.25,
        threshold_step_o=0,
        current_decay_o=0.25,
        voltage_decay_o=0.03,
        threshold_decay_o=0,
        refractory_decay_o=0,
        # Other
        weight_norm=True,
        tau_grad=1,
        scale_grad=1,
    ):
        super(ConvSHDExtended, self).__init__(n_inpt, n_classes, inpt_shape)

        neuron_params_c1 = {
            "threshold": threshold_c1,
            "threshold_step": threshold_step_c1,
            "current_decay": current_decay_c1,
            "voltage_decay": voltage_decay_c1,
            "threshold_decay": threshold_decay_c1,
            "refractory_decay": refractory_decay_c1,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_c1),
            "requires_grad": True,
        }
        neuron_params_a1 = {
            "threshold": threshold_a1,
            "threshold_step": threshold_step_a1,
            "current_decay": current_decay_a1,
            "voltage_decay": voltage_decay_a1,
            "threshold_decay": threshold_decay_a1,
            "refractory_decay": refractory_decay_a1,
            "tau_grad": 1.0,
            "scale_grad": 1.0,
            "requires_grad": False,
        }

        neuron_params_c2 = {
            "threshold": threshold_c2,
            "threshold_step": threshold_step_c2,
            "current_decay": current_decay_c2,
            "voltage_decay": voltage_decay_c2,
            "threshold_decay": threshold_decay_c2,
            "refractory_decay": refractory_decay_c2,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "dropout": slayer.neuron.Dropout(p=dropout_c2),
            "requires_grad": True,
        }
        neuron_params_a2 = {
            "threshold": threshold_a2,
            "threshold_step": threshold_step_a2,
            "current_decay": current_decay_a2,
            "voltage_decay": voltage_decay_a2,
            "threshold_decay": threshold_decay_a2,
            "refractory_decay": refractory_decay_a2,
            "tau_grad": 1.0,
            "scale_grad": 1.0,
            "requires_grad": False,
        }

        neuron_params_o = {
            "threshold": threshold_o,
            "threshold_step": threshold_step_o,
            "current_decay": current_decay_o,
            "voltage_decay": voltage_decay_o,
            "threshold_decay": threshold_decay_o,
            "refractory_decay": refractory_decay_o,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": True,
        }

        c1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1
        )
        a1o_shape = np.floor((c1o_shape + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1)
        c2o_shape = np.floor((a1o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a2o_shape = np.floor((c2o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)

        print(f"SHAPE: ,{c1o_shape}, {a1o_shape}")

        c1 = slayer.block.alif.Conv(
            neuron_params_c1,
            in_features=int(self.inpt_shape[0]),
            out_features=int(c1_filters),
            kernel_size=(1, c1_k),
            stride=(1, c1_s),
            padding=(0, c1_p),
            dilation=(1, c1_d),
            delay=True,
            weight_norm=weight_norm,
        )

        a1 = slayer.block.alif.Pool(
            neuron_params_a1,
            kernel_size=(1, a1_k),
            stride=(1, a1_s),
            padding=(0, a1_p),
        )

        c2 = slayer.block.alif.Conv(
            neuron_params_c2,
            in_features=int(c1_filters),
            out_features=int(c2_filters),
            kernel_size=(1, c2_k),
            stride=(1, c2_s),
            padding=(0, c2_p),
            dilation=(1, c2_d),
            delay=True,
            weight_norm=weight_norm,
        )

        a2 = slayer.block.alif.Pool(
            neuron_params_a2,
            kernel_size=(1, a2_k),
            stride=(1, a2_s),
            padding=(0, a2_p),
        )

        flat1 = slayer.block.alif.Flatten()

        outpt = slayer.block.alif.Dense(
            neuron_params_o,
            int(a2o_shape * c2_filters),
            self.n_classes,
            weight_norm=weight_norm,
        )

        self.recorders_train = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }
        self.recorders_test = {
            "inpt": 0,
            "c1": 0,
            "a1": 0,
            "c2": 0,
            "a2": 0,
            "outpt": 0,
        }

        self.register_layer("c1", c1)
        self.register_layer("a1", a1)
        self.register_layer("c2", c2)
        self.register_layer("a2", a2)
        self.register_layer("flat1", flat1)
        self.register_layer("outpt", outpt)

        self.blocks = torch.nn.ModuleList([c1, a1, c2, a2, flat1, outpt])

    def forward(self, spikes):
        spikes = torch.unsqueeze(spikes, 1)
        spikes = torch.unsqueeze(spikes, 1)

        super().forward(spikes)

        # print("input", spikes.shape)
        spikes = self.layers["c1"](spikes)
        self._update_recorders(spikes, "c1")
        # print("c1", spikes.shape)

        spikes = self.layers["a1"](spikes)
        self._update_recorders(spikes, "a1")
        # print("a1", spikes.shape)

        # print("input", spikes.shape)
        spikes = self.layers["c2"](spikes)
        self._update_recorders(spikes, "c2")
        # print("c1", spikes.shape)

        spikes = self.layers["a2"](spikes)
        self._update_recorders(spikes, "a2")
        # print("a1", spikes.shape)

        spikes = self.layers["flat1"](spikes)

        spikes = self.layers["outpt"](spikes)
        self._update_recorders(spikes, "outpt")
        # print("o", spikes.shape)
        # print(f"OUT : {self.out_spike}/{self.computed_image}={self.out_spike/self.computed_image}")

        return spikes
