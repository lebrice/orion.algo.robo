""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
from __future__ import annotations

from logging import getLogger as get_logger
from typing import Sequence

import numpy as np
import torch
from orion.algo.space import Space
from orion.core.worker.trial import Trial

from orion.algo.robo.ablr.ablr_model import ABLR
from orion.algo.robo.ablr.normal import Normal
from orion.algo.robo.base import (
    AcquisitionFnName,
    MaximizerName,
    RoBO,
    WrappedRoboModel,
    build_bounds,
)

logger = get_logger(__file__)


class RoBO_ABLR(RoBO["OrionABLRWrapper"]):
    """[WIP]: Wrapper for the ABLR[1] algorithm.

    The algo is implemented in the HPO-Warm-Start repo.

    [1] [Scalable HyperParameter Transfer Learning](
        https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
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
        """

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
        **kwargs:
            Arguments specific to each RoBO algorithms. These will be registered as part of
            the algorithm's configuration.


        BUG: log_ei seems to only work when batch size == 1.
        """
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
        return OrionABLRWrapper(
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


class OrionABLRWrapper(ABLR, WrappedRoboModel):
    """Orion wrapper around the ABLR algorithm."""

    @property
    def lower(self) -> np.ndarray:
        return build_bounds(self.space)[0]

    @property
    def upper(self) -> np.ndarray:
        return build_bounds(self.space)[1]

    def set_state(self, state_dict: dict) -> None:
        """Restore the state of the optimizer"""
        random_state: dict = state_dict.pop("rng")
        torch.random.set_rng_state(random_state["torch"])
        self.load_state_dict(state_dict)

    def state_dict(
        self,
        destination: str | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict:
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict
        state_dict["rng"] = {
            "torch": torch.random.get_rng_state(),
        }
        return state_dict

    def seed(self, seed: int) -> None:
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        # NOTE: No need to create a bunch of seeds, we only need the pytorch seed.
        pytorch_seed = seed

        torch.manual_seed(pytorch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pytorch_seed)
            # Not really necessary, makes code a LOT slower.
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True

    # TODO: Not yet adding warm-start support here.
    # def warm_start(
    #     self, warm_start_trials: Dict[ExperimentInfo, List[Trial]]
    # ) -> None:
    #     """ TODO: Warm-start the ABLR algorithm. """
    #     return super().warm_start(warm_start_trials)


def ablr_main():
    from warmstart.tasks.quadratics import QuadraticsTask

    task = QuadraticsTask()

    model = ABLR(
        task,
        epochs=10,
        batch_size=10_000,
    )

    # TODO: Should we rescale things? The samples wouldn't match their space though.
    print(f"Task: {task}")

    train_dataset = task.make_dataset(10_000)
    X, y = train_dataset.tensors
    # Just to get rid of the negative values in X.

    model.train(X, y)
    test_dataset = task.make_dataset(100)
    test_x, test_y = test_dataset.tensors

    with torch.no_grad():
        y_pred_dist: Normal = model.predict_dist(test_x)
        y_pred = y_pred_dist.sample()

        for x, y_pred_i, y_true in zip(test_x, y_pred, test_y):
            print(f"X: {x}, y_pred: {y_pred_i.item()}, y_true: {y_true.item():.3f}")
        # test_loss = nn.MSELoss()(test_y, y_pred.reshape(test_y.shape))

    # print(f"Test loss: {test_loss}")
    exit()


if __name__ == "__main__":
    ablr_main()
