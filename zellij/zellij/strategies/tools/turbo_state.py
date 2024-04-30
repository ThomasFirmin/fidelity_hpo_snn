# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations

from typing import Optional, Union
from abc import ABC, abstractmethod

from zellij.core.errors import InitializationError

import torch
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.generation.sampling import MaxPosteriorSampling
from torch import Tensor


import torch
from dataclasses import dataclass
from torch import Tensor
import numpy as np
import gc

import logging
import math

logger = logging.getLogger("zellij.turbotools")


@dataclass
class AsyncTurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 2
    success_counter: int = 0
    success_tolerance: int = 2
    best_point: Optional[Tensor] = None
    best_value: float = -float("inf")
    restart_triggered: bool = False
    current_lbounds: Optional[Tensor] = None
    current_ubounds: Optional[Tensor] = None
    greedy_move: bool = False

    def __post_init__(self):
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.best_point = torch.zeros(self.dim)

    def reset(self):
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = 2
        self.success_counter = 0
        self.success_tolerance = 2
        self.restart_triggered = False
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.greedy_move = False


def get_best_index_for_batch(Y: Tensor):
    """Return the index for the best point."""
    return Y.argmax()


def async_update_state(state, X_next, Y_next, lengths, successes, failures, eps=1e-5):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_idx = get_best_index_for_batch(Y=Y_next)
    x_next = X_next[best_idx]
    y_next = Y_next[best_idx]
    l_next = lengths[best_idx]
    s_next = successes[best_idx]
    f_next = failures[best_idx]

    current_lbounds = state.current_lbounds.clone().to(x_next.device)
    current_ubounds = state.current_ubounds.clone().to(x_next.device)

    in_tr_x = torch.all((x_next >= current_lbounds) & (x_next <= current_ubounds))

    # Greedy within this context != historic of greedy move in state
    instant_greedy_move = False
    next_length = -1

    # At least one new candidate is feasible
    p_improvement_threshold = state.best_value + eps * math.fabs(state.best_value)
    # In TR
    if y_next > p_improvement_threshold:
        state.best_point = torch.Tensor(x_next)
        state.best_value = y_next.item()
        if in_tr_x:
            state.greedy_move = False
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.greedy_move = True
            instant_greedy_move = True
            state.success_counter = 0
            state.failure_counter = 0
            next_length = l_next.item()
            next_successes = s_next.item()
    else:
        state.success_counter = 0
        state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if instant_greedy_move:
        state.length = next_length
        if next_successes > 0:
            state.success_counter = next_successes + 1
        else:
            state.failure_counter = 0
    elif state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_tr_length(state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


@dataclass
class CTurboState:
    dim: int
    batch_size: int
    best_constraint_values: Tensor
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # type: ignore
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

    def reset(self):
        self.best_constraint_values = (
            torch.ones_like(self.best_constraint_values) * torch.inf
        )
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.success_counter = 0
        self.success_tolerance = 3  # Note: The original paper uses 3
        self.best_value = -float("inf")
        self.restart_triggered = False

        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def c_get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_c_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = c_get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state


@dataclass
class AsyncCTurboState:
    dim: int
    batch_size: int
    best_constraint_values: Tensor
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 2
    success_counter: int = 0
    success_tolerance: int = 2
    best_point: Optional[Tensor] = None
    best_value: float = -float("inf")
    restart_triggered: bool = False
    current_lbounds: Optional[Tensor] = None
    current_ubounds: Optional[Tensor] = None
    greedy_move: bool = False

    def __post_init__(self):
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.best_point = torch.zeros(self.dim)

    def reset(self):
        self.best_constraint_values = (
            torch.ones_like(self.best_constraint_values) * torch.inf
        )
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = 2
        self.success_counter = 0
        self.success_tolerance = 2
        self.restart_triggered = False
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.greedy_move = False


def async_update_c_state(
    state, X_next, Y_next, C_next, lengths, successes, failures, eps=1e-5
):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_idx = c_get_best_index_for_batch(Y=Y_next, C=C_next)
    x_next = X_next[best_idx]
    y_next = Y_next[best_idx]
    c_next = C_next[best_idx]
    l_next = lengths[best_idx]
    s_next = successes[best_idx]
    f_next = failures[best_idx]

    current_lbounds = state.current_lbounds.clone().to(x_next.device)
    current_ubounds = state.current_ubounds.clone().to(x_next.device)

    in_tr_x = torch.all((x_next >= current_lbounds) & (x_next <= current_ubounds))

    # Greedy within this context != historic of greedy move in state
    instant_greedy_move = False
    next_length = -1
    if (c_next <= 0).all():
        # At least one new candidate is feasible
        p_improvement_threshold = state.best_value + eps * math.fabs(state.best_value)
        # In TR
        if y_next > p_improvement_threshold or (state.best_constraint_values > 0).any():
            state.best_point = torch.Tensor(x_next)
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
            if in_tr_x:
                state.greedy_move = False
                state.success_counter += 1
                state.failure_counter = 0
            else:
                state.greedy_move = True
                instant_greedy_move = True
                state.success_counter = 0
                state.failure_counter = 0
                next_length = l_next.item()
                next_successes = s_next.item()
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.best_point = torch.Tensor(x_next)
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
            if in_tr_x:
                state.greedy_move = False
                state.success_counter += 1
                state.failure_counter = 0
            else:
                state.greedy_move = True
                instant_greedy_move = True
                state.success_counter = 0
                state.failure_counter = 0
                next_length = l_next.item()
                next_successes = s_next.item()
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if instant_greedy_move:
        state.length = next_length
        if next_successes > 0:
            state.success_counter = next_successes + 1
        else:
            state.failure_counter = 0
    elif state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


@dataclass
class ICTurboState:
    dim: int
    batch_size: int
    best_constraint_values: Tensor
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 2
    success_counter: int = 0
    success_tolerance: int = 2
    best_value: float = -float("inf")
    best_cost: float = float("inf")
    restart_triggered: bool = False
    best_point: Optional[Tensor] = None
    current_lbounds: Optional[Tensor] = None
    current_ubounds: Optional[Tensor] = None
    greedy_move: bool = False

    def __post_init__(self):
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.best_point = torch.zeros(self.dim)

    def reset(self):
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = 2
        self.success_counter = 0
        self.success_tolerance = 2
        self.restart_triggered = False
        self.current_lbounds = torch.zeros(self.dim)
        self.current_ubounds = torch.ones(self.dim)
        self.greedy_move = False


def iget_best_index_for_batch(Y: Tensor, C: Tensor, Cost: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        costs = Cost.clone()

        score[~is_feas] = -float("inf")
        best_idx = score.argmax()
        bscore = score[best_idx]
        mask = score == bscore
        if len(costs[~mask]) > 1:
            costs[~mask] = float("inf")
            best_idx = costs.argmin()

        return best_idx
    return C.clamp(min=0).sum(dim=-1).argmin()


def iupdate_c_state(
    state, X_next, Y_next, C_next, Cost_next, lengths, successes, failures, eps=1e-5
):

    best_idx = iget_best_index_for_batch(Y_next, C_next, Cost_next)
    x_next = X_next[best_idx]
    y_next = Y_next[best_idx]
    c_next = C_next[best_idx]
    cost_next = Cost_next[best_idx]
    l_next = lengths[best_idx]
    s_next = successes[best_idx]
    f_next = failures[best_idx]

    current_lbounds = state.current_lbounds.clone().to(x_next.device)
    current_ubounds = state.current_ubounds.clone().to(x_next.device)

    in_tr_x = torch.all((x_next >= current_lbounds) & (x_next <= current_ubounds))

    # Greedy within this context != historic of greedy move in state
    instant_greedy_move = False
    next_length = -1
    if (c_next <= 0).all():
        # At least one new candidate is feasible
        p_improvement_threshold = state.best_value + eps * math.fabs(state.best_value)
        # In TR
        if y_next > p_improvement_threshold or (state.best_constraint_values > 0).any():
            state.best_point = torch.Tensor(x_next)
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
            state.best_cost = cost_next.item()
            if in_tr_x:
                state.greedy_move = False
                state.success_counter += 1
                state.failure_counter = 0
            else:
                state.greedy_move = True
                instant_greedy_move = True
                state.success_counter = 0
                state.failure_counter = 0
                next_length = l_next.item()
                next_successes = s_next.item()
        elif (
            y_next <= p_improvement_threshold
            and y_next >= state.best_value
            and cost_next < state.best_cost
        ):
            state.best_point = torch.Tensor(x_next)
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
            state.best_cost = cost_next.item()
            if in_tr_x:
                state.greedy_move = False
                state.success_counter += 1
                state.failure_counter = 0
            else:
                state.greedy_move = True
                instant_greedy_move = True
                state.success_counter = 0
                state.failure_counter = 0
                next_length = l_next.item()
                next_successes = s_next.item()
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.best_point = torch.Tensor(x_next)
            state.best_value = y_next.item()
            state.best_constraint_values = torch.Tensor(c_next)
            state.best_cost = cost_next.item()
            if in_tr_x:
                state.greedy_move = False
                state.success_counter += 1
                state.failure_counter = 0
            else:
                state.greedy_move = True
                instant_greedy_move = True
                state.success_counter = 0
                state.failure_counter = 0
                next_length = l_next.item()
                next_successes = s_next.item()
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if instant_greedy_move:
        state.length = next_length
        if next_successes > 0:
            state.success_counter = next_successes + 1
        else:
            state.failure_counter = 0
    elif state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


class ConstrainedCostAwareMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained Cost Aware max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        cost_model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        temperature: float,
        best_cost: float,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        epsilon: float = 1e-3,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.cost_model = cost_model
        self.constraint_model = constraint_model
        self.temperature = temperature
        print(f"TEMPERATURE : {temperature}")
        self.best_cost = best_cost
        self.epsilon = epsilon

    def _convert_samples_to_scores(
        self, Y_samples, Cost_samples, C_samples, num_samples
    ) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            Cost_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the cost model.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = -total_violation

        # Filter according to cost
        delta_cost = Cost_samples - self.best_cost
        is_costly = delta_cost > 0

        # normalize costly
        if torch.any(is_costly):
            max_costly = delta_cost[is_costly].max()
            norm_delta_costly = delta_cost[is_costly] / max_costly
            non_costly = scores.shape[1] - len(norm_delta_costly)

            # Compute acceptance probability
            probabilities = torch.exp(-norm_delta_costly / self.temperature)
            random_sample = torch.rand(
                len(norm_delta_costly), device=probabilities.device
            )
            deny = random_sample > probabilities
            deny_sum = deny.sum()
            non_deny = len(norm_delta_costly) - deny_sum

            # Keep costly samples according to probability
            # if number kept elements are lower than batch size
            # slowly force the acceptance
            eps = 0
            while non_costly + non_deny < num_samples:
                print("DENYING")
                eps += self.epsilon
                random_sample = (
                    torch.rand(len(norm_delta_costly), device=probabilities.device)
                    - eps
                )
                deny = random_sample > probabilities
                deny_sum = deny.sum()
                non_deny = len(norm_delta_costly) - deny_sum

            scores[:, is_costly][:, deny] = -float("inf")
            print(f"DENIED : {deny_sum} {deny.shape}")
        else:
            print(Cost_samples)
            print("NOTHING COSTLY")
        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        print(Y_samples)
        # Constraints
        c_posterior = self.constraint_model.posterior(
            X=X, observation_noise=observation_noise
        )
        C_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Cost
        Cost_samples = self.cost_model(X=X)

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples,
            C_samples=C_samples,
            Cost_samples=Cost_samples,
            num_samples=num_samples,
        )
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)


class ConstrainedTSPerUnitPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained Cost Aware max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        cost_model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        best_score: float,
        temperature: float,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        device=torch.device("cpu"),
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.cost_model = cost_model
        self.cost_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.constraint_model = constraint_model
        self.constraint_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.temperature = temperature
        self.best_score = best_score
        self.device = device

    def _convert_samples_to_scores(self, Y_samples, C_samples, Cost_samples) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            Cost_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the cost model.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        cost_anneal = (
            (Cost_samples**self.temperature).repeat(Y_samples.shape[0], 1).unsqueeze(-1)
        )

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = (
                -total_violation * cost_anneal[~has_feasible_candidate]
            )
        else:
            scores = scores - self.best_score
            mask = scores > 0
            scores[mask] /= cost_anneal[mask]
            scores[~mask] *= cost_anneal[~mask]

        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        gc.collect()
        torch.cuda.empty_cache()

        # Objective function
        self.model.to(self.device)
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            # Note: `posterior_transform` is only used for the objective
            posterior_transform=self.posterior_transform,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Constraints
        self.constraint_model.to(self.device)
        c_posterior = self.constraint_model.posterior(
            X=X, observation_noise=observation_noise
        )
        C_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))
        self.constraint_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Cost
        self.cost_model.to(self.device)
        Cost_samples = torch.exp(self.cost_model(X).mean)
        self.cost_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples,
            C_samples=C_samples,
            Cost_samples=Cost_samples,
        )
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)


class Temperature(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def temperature(self, elapsed: float) -> float:
        pass


class SkewedBell(Temperature):
    """SkewedBell

    SkewedBell cooling for CASCBO.

    :math:`T = \\gamma\\frac{xe^{\\sqrt{\\alpha x^p}}}{x+e^{-\\alpha x^p}}`

    Attributes
    ----------
    gamma : float
        Influences the value of the peak.
    alpha : float
        Skewness of the curve
    p : float
        Influences the shift of the curve.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, gamma: float, alpha: float, p: float):
        self.gamma = gamma
        self.alpha = alpha
        self.p = p

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value > 0:
            self._gamma = value
        else:
            raise InitializationError(f"gamma must be >0. Got {value}")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value > 0:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be >0. Got {value}")

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        if value > 0:
            self._p = value
        else:
            raise InitializationError(f"p must be >0. Got {value}")

    def temperature(self, elapsed: float) -> float:
        n = elapsed * np.exp(-np.sqrt(self.alpha) * elapsed**self.p)
        d = elapsed + np.exp(-self.alpha * elapsed**self.p)
        return self.gamma * n / d
