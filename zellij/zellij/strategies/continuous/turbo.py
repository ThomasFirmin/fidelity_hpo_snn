# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)
from __future__ import annotations
from zellij.core.errors import InputError
from zellij.core.metaheuristic import UnitMetaheuristic
from zellij.strategies.tools.turbo_state import async_update_state

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace
    from zellij.strategies.tools.turbo_state import AsyncTurboState

import torch
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll

import numpy as np
from datetime import datetime

import os
import gc

import logging

logger = logging.getLogger("zellij.scbo")


class TuRBO(UnitMetaheuristic):
    """Scalable Constrained Bayesian Optimization

    Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `TuRBO <https://botorch.org/tutorials/turbo_1>`_.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : ContinuousSearchspace
        Search space object containing bounds of the search space
    turbo_state : AsyncTurboState
        :code:`AsyncTurboState` object.
    verbose : bool
        If False, there will be no print.
    surrogate : botorch.models.model.Model, default=SingleTaskGP
        Gaussian Process Regressor object from 'botorch'.
        Determines the surrogate model that Bayesian optimization will use to
        interpolate the loss function
    mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
            Object from gpytorch.mlls it determines which marginal loglikelihood to use
            when optimizing kernel's hyperparameters
    likelihood : gpytorch.likelihoods, default=GaussianLikelihood
        Object from gpytorch.likelihoods defining the likelihood.
    batch_size : int, default=4
        Number of solutions sampled within the surrogate, to return at each iteration.
    n_canditates : int, default=None
        Number of candidates to sample with the surrogate.
    initial_size : int, default=10
        Size of the initial set of solution to draw randomly.
    cholesky_size : int, default=800
        Maximum size for which Lanczos method is used instead of Cholesky decomposition.
    beam : int, default=2000
        Maximum number of solutions that can be stored and used to compute the Gaussian Process.
    gpu: bool, default=True
        Use GPU if available
    kwargs
        Key word arguments linked to the surrogate, mll or likelihood.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij

    """

    def __init__(
        self,
        search_space: UnitSearchspace,
        turbo_state: AsyncTurboState,
        batch_size: int,
        verbose: bool = True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        n_candidates: Optional[int] = None,
        initial_size: int = 10,
        cholesky_size: int = 800,
        beam: int = 2000,
        gpu: bool = False,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : UnitSearchspace
            UnitSearchspace :ref:`sp`.
        turbo_state : AsyncTurboState
            :code:`AsyncTurboState` object.
        verbose : bool
            If False, there will be no print.
        surrogate : botorch.models.model.Model, default=SingleTaskGP
            Gaussian Process Regressor object from 'botorch'.
            Determines the surrogate model that Bayesian optimization will use to
            interpolate the loss function
        mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
                Object from gpytorch.mlls it determines which marginal loglikelihood to use
                when optimizing kernel's hyperparameters
        likelihood : gpytorch.likelihoods, default=GaussianLikelihood
            Object from gpytorch.likelihoods defining the likelihood.
        batch_size : int, default=4
            Number of solutions sampled within the surrogate, to return at each iteration.
        n_canditates : int, default=None
            Number of candidates to sample with the surrogate.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        cholesky_size : int, default=800
            Maximum size for which Lanczos method is used instead of Cholesky decomposition.
        beam : int, default=2000
            Maximum number of solutions that can be stored and used to compute the Gaussian Process.
        gpu: bool, default=True
            Use GPU if available
        kwargs
            Key word arguments linked to the surrogate, mll or likelihood.
        """

        super().__init__(search_space, verbose)

        ####################
        # INFO TO RETRIEVE # # See metaheuristic
        ####################
        self.info = ["length", "successes", "failures"]

        ##############
        # PARAMETERS #
        ##############
        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood

        self.batch_size = batch_size

        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.beam = beam

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.turbo_state = turbo_state

        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if isinstance(gpu, str):
            self.device = torch.device(gpu)
        elif gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.double

        # Prior points
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        # Prior objective
        self.train_obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        # Count generated models
        self.models_number = 0

        self.cholesky_size = cholesky_size

        self.iterations = 0

    def _build_kwargs(self):
        # Surrogate kwargs
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }

        for m in self.model_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # Likelihood kwargs
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        for m in self.likelihood_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # MLL kwargs
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        for m in self.mll_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        logger.debug(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def _generate_initial_data(self, n) -> List[list]:
        return self.search_space.random_point(n)

    # Initialize a surrogate
    def _initialize_model(
        self,
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        state_dict: Optional[dict] = None,
    ):
        train_x.to(self.device, dtype=self.dtype)
        train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)

        # define models for objective
        model = self.surrogate(
            train_x,
            train_obj,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            **self.model_kwargs,
        )
        model.to(self.device)

        if "num_data" in self.mll.__init__.__code__.co_varnames:
            mll = self.mll(
                model.likelihood,
                model.model,
                num_data=train_x.shape[-2],  # type: ignore
                **self.mll_kwargs,
            )
        else:
            mll = self.mll(
                model.likelihood,
                model,
                **self.mll_kwargs,
            )

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return mll, model

    def generate_batch(
        self,
        state: AsyncTurboState,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = state.best_point.clone()

        # Add weights based trust region
        # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        # weights = weights / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        ################################

        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0).to(
            dtype=self.dtype, device=self.device
        )
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0).to(
            dtype=self.dtype, device=self.device
        )

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        model.to(self.device)
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)
        model.to("cpu")
        gc.collect()
        return X_next.detach()

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        self.obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)

    def prune(self, X, Y):
        # Remove worst solutions from the beam
        sidx = torch.argsort(Y.squeeze(), descending=True)

        X = X[sidx]
        Y = Y[sidx]

        filled_x = X[: self.beam]
        filled_obj = Y[: self.beam]

        return filled_x, filled_obj

    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """forward

        Abstract method describing one step of the :ref:`meta`.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.
        secondary : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.
        constraint : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """
        gc.collect()
        torch.cuda.empty_cache()
        ask_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        if self.turbo_state.restart_triggered:
            # self.initialized = False
            self.turbo_state.reset()

        if not self.initialized:
            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data(self.initial_size)
            self.initialized = True
            rdict = {
                "iteration": self.iterations,
                "algorithm": "InitTuRBO",
                "ask_date": ask_date,
                "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "length": 1.0,
                "successes": 0.0,
                "failures": 0.0,
                "trestart": self.turbo_state.restart_triggered,
                "greedy": self.turbo_state.greedy_move,
                "best_value": self.turbo_state.best_value,
            }
            rdict["model"] = -1
            rdict["beam"] = len(self.train_obj)

            return train_x, rdict

        elif X is None or Y is None or info is None:
            raise InputError(
                "After initialization TuRBO must receive non-empty X, Y and info in forward."
            )
        else:
            self.iterations += 1

            new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
            new_obj = -torch.tensor(Y, dtype=self.dtype, device=self.device).unsqueeze(
                -1
            )

            new_lengths = torch.tensor(
                info[:, 0], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            new_successes = torch.tensor(
                info[:, 1], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            new_failures = torch.tensor(
                info[:, 2], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)

            # update training points
            self.train_x = torch.cat([self.train_x, new_x], dim=0)
            self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)

            # Remove worst solutions from the beam
            if len(self.train_x) > self.beam:
                self.train_x, self.train_obj = self.prune(self.train_x, self.train_obj)

            # If initial size not reached, returns 1 additionnal solution
            if len(self.train_obj) < self.initial_size:
                rdict = {
                    "iteration": self.iterations,
                    "algorithm": "AddInitTuRBO",
                    "ask_date": ask_date,
                    "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                    "length": 1.0,
                    "successes": 0.0,
                    "failures": 0.0,
                    "trestart": self.turbo_state.restart_triggered,
                    "greedy": self.turbo_state.greedy_move,
                    "best_value": self.turbo_state.best_value,
                }
                rdict["model"] = -1
                rdict["beam"] = len(self.train_obj)
                return self._generate_initial_data(1), rdict
            else:
                self.turbo_state = async_update_state(
                    state=self.turbo_state,
                    X_next=new_x,
                    Y_next=new_obj,
                    lengths=new_lengths,
                    successes=new_successes,
                    failures=new_failures,
                )
                with gpytorch.settings.max_cholesky_size(self.cholesky_size):
                    # reinitialize the models so they are ready for fitting on next iteration
                    # use the current state dict to speed up fitting

                    gc.collect()
                    torch.cuda.empty_cache()

                    mll, model = self._initialize_model(
                        self.train_x,
                        self.train_obj,
                        state_dict=None,
                    )

                    try:
                        fit_gpytorch_mll(mll)
                        gc.collect()
                        torch.cuda.empty_cache()
                    except ModelFittingError:
                        rdict = {
                            "iteration": self.iterations,
                            "algorithm": "FailedTuRBO",
                            "ask_date": ask_date,
                            "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                            "length": 1.0,
                            "successes": 0.0,
                            "failures": 0.0,
                            "trestart": self.turbo_state.restart_triggered,
                            "greedy": self.turbo_state.greedy_move,
                            "best_value": self.turbo_state.best_value,
                        }
                        rdict["model"] = -1
                        rdict["beam"] = len(self.train_obj)
                        return self._generate_initial_data(len(Y)), rdict

                    model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # optimize and get new observation
                    new_x = self.generate_batch(
                        state=self.turbo_state,
                        model=model,
                        X=self.train_x,
                        Y=self.train_obj,
                        batch_size=self.batch_size,
                        n_candidates=self.n_candidates,
                    )

                    if self._save:
                        self.save(model)

                    rdict = {
                        "iteration": self.iterations,
                        "algorithm": "TuRBO",
                        "ask_date": ask_date,
                        "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                        "length": self.turbo_state.length,
                        "successes": self.turbo_state.success_counter,
                        "failures": self.turbo_state.failure_counter,
                        "trestart": self.turbo_state.restart_triggered,
                        "greedy": self.turbo_state.greedy_move,
                        "best_value": self.turbo_state.best_value,
                    }
                    rdict["model"] = self.models_number
                    rdict["beam"] = len(self.train_obj)

                    return new_x.cpu().numpy().tolist(), rdict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cmodels_list"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, model):
        path = self._save
        foldername = os.path.join(path, "scbo")
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        std_dict = model.state_dict()
        std_dict["nlengthscale"] = model.covar_module.base_kernel.lengthscale

        torch.save(
            std_dict,
            os.path.join(foldername, f"{self.models_number}_model.pth"),
        )
        self.models_number += 1
