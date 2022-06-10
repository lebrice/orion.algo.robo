""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
from __future__ import annotations

from logging import getLogger as get_logger
from typing import Sequence

import torch
from orion.algo.space import Space
from orion.core.worker.trial import Trial

from orion.algo.robo.ablr.ablr_model import ABLR
from orion.algo.robo.base import AcquisitionFnName, MaximizerName, RoBO

logger = get_logger(__file__)


class RoBO_ABLR(RoBO[ABLR]):
    """Implements the ABLR[1] algorithm, using RoBO.

    [1] [Scalable HyperParameter Transfer Learning](
        https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)


    Parameters
    ----------
    space: ``orion.algo.space.Space``
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed to sample initial points and candidates points.
        Default: 0.
    n_initial_points: int
        Number of initial points randomly sampled. If new points
        are requested and less than `n_initial_points` are observed,
        the next points will also be sampled randomly instead of being
        sampled from the parzen estimators.
        Default: ``20``
    maximizer: str
        The optimizer for the acquisition function.
        Can be one of ``{"random", "scipy", "differential_evolution"}``.
        Defaults to 'random'
    acquisition_func: str
        Name of the acquisition function. Can be one of ``['ei', 'log_ei', 'pi', 'lcb']``.
    hparams: dict | ABLR.HParams | None
        Hyperparameters for the ABLR model.
        If None, the default hyperparameters will be used.
    normalize_inputs: bool
        Whether to normalize the inputs.
    """

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = 0,
        n_initial_points: int = 20,
        maximizer: MaximizerName = "random",
        acquisition_func: AcquisitionFnName = "ei",
        hparams: ABLR.HParams | dict | None = None,
        normalize_inputs: bool = True,
    ):
        super().__init__(
            space=space,
            seed=seed,
            n_initial_points=n_initial_points,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
        )
        self.hparams = hparams
        self.normalize_inputs = normalize_inputs

    def build_model(self):
        return ABLR(
            self.space,
            hparams=self.hparams,
            normalize_inputs=self.normalize_inputs,
        )

    def seed_rng(self, seed: int) -> None:
        super().seed_rng(seed)
        torch_seed = self.rng.randint(0, int(1e8))
        torch.random.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    @property
    def state_dict(self) -> dict:
        state = super().state_dict
        state["torch_rng"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_rng"] = torch.cuda.random.get_rng_state_all()
        return state

    def set_state(self, state_dict: dict):
        super().set_state(state_dict)
        torch_rng = state_dict["torch_rng"]
        torch.random.set_rng_state(torch_rng)

        # NOTE: In principle it could be possible to load a state_dict from an env with cuda in an
        # env without cuda and vice-versa.
        if torch.cuda.is_available() and "torch_cuda_rng" in state_dict:
            torch_cuda_rng = state_dict["torch_cuda_rng"]
            torch.cuda.random.set_rng_state_all(torch_cuda_rng)

    def suggest(self, num: int) -> list[Trial] | None:
        return super().suggest(num=num)
