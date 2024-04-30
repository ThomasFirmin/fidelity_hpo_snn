# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-21T11:19:42+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:00:55+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core import ArrayVar, FloatVar, IntVar, CatVar
from zellij.utils.converters import FloatMinMax, IntMinMax, CatToFloat, ArrayDefaultC
import scipy
import numpy as np


def loguniform(low, high, size=None):
    return scipy.stats.loguniform.rvs(low, high, size=size)


# reversew
def rev_loguniform(low, high, size=None):
    return high - scipy.stats.loguniform.rvs(low, high, size=size) + low


# discrete loguniform
def disc_loguniform(low, high, size=None):
    return np.round(scipy.stats.loguniform.rvs(low, high - 1, size=size)).astype(int)


# reverse
def disc_rev_loguniform(low, high, size=None):
    return np.round(
        high - scipy.stats.loguniform.rvs(low, high - 1, size=size) + low
    ).astype(int)


######################################
################ LAVA ################
######################################


def get_mnist_rate_lava_cnt(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.1, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(IntVar("frames", 10, 50, converter=IntMinMax(), sampler=loguniform))

    values.append(IntVar("epochs", 1, 20, converter=IntMinMax()))

    values.append(
        IntVar(
            "batch_size", 20, 200, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 48, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 48, converter=IntMinMax()))

    values.append(IntVar("c1_k", 4, 12, converter=IntMinMax()))
    values.append(IntVar("c2_k", 4, 12, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("mu", 0.8, 0.999, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar("gamma", 0.9, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar("tau_grad", 0.001, 1, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.1, 1, converter=FloatMinMax(), sampler=loguniform)
    )

    ## Neurons params

    values.append(
        FloatVar(
            "threshold",
            0.5,
            8,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            "threshold_step",
            0.05,
            0.40,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            "current_decay", 0.05, 0.5, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar(
            "voltage_decay", 0.01, 0.2, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.01,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay",
            0.05,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinMax(), sampler=loguniform)
    )

    return values


def get_mnist_rate_lava_cnt_extended(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.1, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(
        IntVar("frames", 10, 50, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(IntVar("epochs", 1, 20, converter=IntMinMax()))

    values.append(
        IntVar(
            "batch_size", 30, 300, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 48, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 48, converter=IntMinMax()))

    values.append(IntVar("c1_k", 4, 12, converter=IntMinMax()))
    values.append(IntVar("c2_k", 4, 12, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("mu", 0.8, 0.999, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar("gamma", 0.9, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            f"tau_grad",
            0.005,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"scale_grad",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    ## Neurons params
    for layer in ["c1", "c2", "a1", "a2", "o"]:
        values.append(
            FloatVar(
                f"threshold_{layer}",
                0.5,
                8,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"threshold_step_{layer}",
                0.05,
                0.40,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"current_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"voltage_decay_{layer}",
                0.01,
                0.2,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"threshold_decay_{layer}",
                0.01,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"refractory_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )

        if layer not in ["a1", "a2", "o"]:
            values.append(
                FloatVar(
                    f"dropout_{layer}",
                    0.01,
                    0.90,
                    converter=FloatMinMax(),
                    sampler=loguniform,
                )
            )
    return values


def get_dvs_rate_lava_cnt(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.20, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(IntVar("frames", 10, 50, converter=IntMinMax(), sampler=loguniform))

    values.append(IntVar("epochs", 1, 15, converter=IntMinMax(), sampler=loguniform))

    values.append(
        IntVar("batch_size", 5, 40, converter=IntMinMax(), sampler=disc_rev_loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 36, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 36, converter=IntMinMax()))

    values.append(IntVar("c1_k", 8, 48, converter=IntMinMax()))
    values.append(IntVar("c2_k", 8, 48, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("tau_grad", 0.1, 1, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.5, 1, converter=FloatMinMax(), sampler=loguniform)
    )
    ## Neurons params

    values.append(
        FloatVar(
            "threshold",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            "threshold_step",
            0.001,
            0.30,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            "current_decay", 0.1, 0.9, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "voltage_decay", 0.1, 0.9, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.05,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay", 0.05, 0.5, converter=FloatMinMax(), sampler=loguniform
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.6, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    return values


def get_shd_lava_cnt_ext(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.2, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(
        IntVar("frames", 200, 400, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar("epochs", 1, 35, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar(
            "batch_size", 20, 300, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(
        IntVar("f1_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f2_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f3_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("beta_1", 0.8, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("beta_2", 0.8, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            f"tau_grad",
            0.005,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"scale_grad",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    for layer in ["f1", "f2", "f3", "o"]:
        ## Neurons params

        values.append(
            FloatVar(
                f"threshold_{layer}",
                0.5,
                10,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"threshold_step_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"current_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"voltage_decay_{layer}",
                0.01,
                0.2,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"threshold_decay_{layer}",
                0.01,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"refractory_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )

        if layer != "o":
            values.append(
                FloatVar(
                    f"dropout_{layer}",
                    0.01,
                    0.90,
                    converter=FloatMinMax(),
                    sampler=loguniform,
                )
            )

    return values


def get_shd_lava_cnt(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.2, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(
        IntVar("frames", 200, 400, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar("epochs", 1, 35, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar(
            "batch_size", 20, 300, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(
        IntVar("f1_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f2_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f3_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("beta_1", 0.8, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("beta_2", 0.8, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            f"tau_grad",
            0.005,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"scale_grad",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    ## Neurons params

    values.append(
        FloatVar(
            f"threshold",
            0.5,
            10,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            f"threshold_step",
            0.05,
            0.5,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar(
            f"current_decay",
            0.05,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"voltage_decay",
            0.01,
            0.2,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"threshold_decay",
            0.01,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"refractory_decay",
            0.05,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"dropout",
            0.01,
            0.90,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    return values


def get_shd_lava_cnt_conv(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(
        FloatVar("train_size", 0.2, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(
        IntVar("frames", 200, 400, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar("epochs", 1, 25, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar(
            "batch_size", 20, 100, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 32, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 32, converter=IntMinMax()))

    values.append(IntVar("c1_k", 20, 200, converter=IntMinMax()))
    values.append(IntVar("a1_k", 4, 40, converter=IntMinMax()))

    values.append(IntVar("c2_k", 4, 40, converter=IntMinMax()))
    values.append(IntVar("a2_k", 2, 15, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("beta_1", 0.8, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("beta_2", 0.8, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            f"tau_grad",
            0.005,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"scale_grad",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    ## Neurons params
    for layer in ["c1", "c2", "a1", "a2", "o"]:
        values.append(
            FloatVar(
                f"threshold_{layer}",
                0.5,
                10,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"threshold_step_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"current_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"voltage_decay_{layer}",
                0.01,
                0.2,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"threshold_decay_{layer}",
                0.01,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"refractory_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        if layer not in ["a1", "a2", "o"]:
            values.append(
                FloatVar(
                    f"dropout_{layer}",
                    0.01,
                    0.90,
                    converter=FloatMinMax(),
                    sampler=loguniform,
                )
            )
    return values


def get_ssc_lava_cnt(convert=True):
    if convert:
        values = ArrayVar(converter=ArrayDefaultC())
    else:
        values = ArrayVar()

    values.append(FloatVar("train_size", 0.05, 1.0, converter=FloatMinMax()))

    values.append(
        IntVar("frames", 150, 300, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar("epochs", 1, 25, converter=IntMinMax(), sampler=disc_loguniform)
    )

    values.append(
        IntVar(
            "batch_size", 100, 800, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(
        IntVar("f1_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f2_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    values.append(
        IntVar("f3_n", 100, 1000, converter=IntMinMax(), sampler=disc_loguniform)
    )
    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("beta_1", 0.8, 1.0, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("beta_2", 0.8, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            f"tau_grad",
            0.005,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            f"scale_grad",
            0.1,
            1,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )

    for layer in ["f1", "f2", "f3", "o"]:
        ## Neurons params

        values.append(
            FloatVar(
                f"threshold_{layer}",
                0.5,
                10,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"threshold_step_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=rev_loguniform,
            )
        )

        values.append(
            FloatVar(
                f"current_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"voltage_decay_{layer}",
                0.01,
                0.2,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"threshold_decay_{layer}",
                0.01,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )
        values.append(
            FloatVar(
                f"refractory_decay_{layer}",
                0.05,
                0.5,
                converter=FloatMinMax(),
                sampler=loguniform,
            )
        )

        if layer != "o":
            values.append(
                FloatVar(
                    f"dropout_{layer}",
                    0.01,
                    0.90,
                    converter=FloatMinMax(),
                    sampler=loguniform,
                )
            )

    return values


########### OLD ###########


def old_get_mnist_lava_cnt():
    values = ArrayVar(converter=ArrayDefaultC())

    values.append(IntVar("epochs", 1, 40, converter=IntMinMax()))

    values.append(
        IntVar(
            "batch_size", 20, 200, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 128, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 128, converter=IntMinMax()))

    values.append(IntVar("c1_k", 4, 12, converter=IntMinMax()))
    values.append(IntVar("c2_k", 4, 12, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("tau_grad", 0.001, 1, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.1, 1, converter=FloatMinMax(), sampler=loguniform)
    )

    ## Neurons params

    values.append(FloatVar("threshold", 0.4, 4, converter=FloatMinMax()))

    values.append(
        FloatVar(
            "threshold_step", 0.001, 0.25, converter=FloatMinMax(), sampler=loguniform
        )
    )

    values.append(FloatVar("current_decay", 0.1, 0.99, converter=FloatMinMax()))
    values.append(
        FloatVar(
            "voltage_decay", 0.01, 0.2, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.01,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay",
            0.1,
            0.99,
            converter=FloatMinMax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinMax(), sampler=loguniform)
    )

    return values


def old_get_dvs_lava_cnt():
    values = ArrayVar(converter=ArrayDefaultC())

    values.append(IntVar("epochs", 1, 15, converter=IntMinMax()))

    values.append(
        IntVar("batch_size", 1, 20, converter=IntMinMax(), sampler=disc_rev_loguniform)
    )

    values.append(
        FloatVar("rate_true", 0.1, 0.9, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("rate_false", 0.01, 0.09, converter=FloatMinMax(), sampler=loguniform)
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 36, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 36, converter=IntMinMax()))

    values.append(IntVar("c1_k", 4, 48, converter=IntMinMax()))
    values.append(IntVar("c2_k", 4, 48, converter=IntMinMax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("tau_grad", 0.001, 1, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.1, 1, converter=FloatMinMax(), sampler=loguniform)
    )

    ## Neurons params
    values.append(
        FloatVar("threshold", 0.1, 1, converter=FloatMinMax(), sampler=loguniform)
    )

    values.append(
        FloatVar(
            "threshold_step", 0.001, 0.40, converter=FloatMinMax(), sampler=loguniform
        )
    )

    values.append(
        FloatVar(
            "current_decay", 0.1, 0.99, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "voltage_decay", 0.01, 0.9, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.01,
            0.5,
            converter=FloatMinMax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay", 0.1, 0.99, converter=FloatMinMax(), sampler=loguniform
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinMax(), sampler=loguniform)
    )
    return values


##########################################
################ BindsNet ################
##########################################


def old_get_gendh_rate_cnt():
    values = ArrayVar(converter=ArrayDefaultC())

    values.append(CatVar("decoder", ["all", "vote", "2gram"], converter=CatToFloat()))
    values.append(IntVar("epochs", 1, 3, converter=IntMinMax()))
    # values.append(
    #    IntVar("batch_size", 1, 50, converter=IntMinmax(), sampler=disc_loguniform)
    # )

    # Diehl and cook
    values.append(IntVar("map_size", 10, 1000, converter=IntMinMax()))

    ## STDP
    values.append(
        FloatVar("nu_pre", 1e-4, 1e-3, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar("nu_post", 1e-3, 1e-2, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar(
            "strength_exc", 50, 500, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "strength_inh", 60, 600, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar("norm", 78.4, 200, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    ## Excitatory layer
    values.append(FloatVar("e_thresh", -70, -50, converter=FloatMinMax()))
    values.append(FloatVar("e_rest", -65, -55, converter=FloatMinMax()))
    values.append(IntVar("e_refrac", 1, 10, converter=IntMinMax()))
    values.append(
        FloatVar("theta_plus", 0.01, 0.1, converter=FloatMinMax(), sampler=loguniform)
    )
    values.append(
        FloatVar(
            "tc_theta_decay", 1e6, 1e7, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "e_tc_decay", 1000, 5000, converter=FloatMinMax(), sampler=rev_loguniform
        )
    )

    ## Inhibitory layer
    values.append(FloatVar("i_thresh", -30, -10, converter=FloatMinMax()))
    values.append(FloatVar("i_rest", -60, -40, converter=FloatMinMax()))
    values.append(IntVar("i_refrac", 15, 35, converter=IntMinMax()))
    values.append(
        FloatVar("i_tc_decay", 1000, 5000, converter=FloatMinMax(), sampler=loguniform)
    )

    return values


###############################################
################ Spiking Jelly ################
###############################################


def old_get_mnist_spikingjelly_cnt():
    values = ArrayVar(converter=ArrayDefaultC())

    values.append(IntVar("epochs", 1, 40, converter=IntMinMax()))

    values.append(
        IntVar(
            "batch_size", 20, 200, converter=IntMinMax(), sampler=disc_rev_loguniform
        )
    )

    # Architecture
    values.append(IntVar("c1_filters", 1, 128, converter=IntMinMax()))
    values.append(IntVar("c2_filters", 1, 128, converter=IntMinMax()))

    values.append(IntVar("c1_k", 4, 12, converter=IntMinMax()))
    values.append(IntVar("c2_k", 4, 12, converter=IntMinMax()))

    # Optimizer
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinMax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("mu", 0.8, 0.999, converter=FloatMinMax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar("gamma", 0.9, 1.0, converter=FloatMinMax(), sampler=rev_loguniform)
    )

    ## Neurons params
    values.append(FloatVar("threshold", 0.2, 4, converter=FloatMinMax()))
    values.append(FloatVar("threshold_decay", 1, 40, converter=FloatMinMax()))
    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinMax(), sampler=loguniform)
    )

    return values
