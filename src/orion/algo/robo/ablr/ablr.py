""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""

from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np
import torch

from orion.algo.robo.ablr.ablr_model import ABLR
from orion.algo.robo.base import RoBO, build_bounds

logger = getLogger(__file__)


class RoBO_ABLR(RoBO):
    """[WIP]: Wrapper for the ABLR[1] algorithm.

    The algo is implemented in the HPO-Warm-Start repo.

    [1] [Scalable HyperParameter Transfer Learning](
        https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
    """

    def __init__(
        self,
        space,
        seed=0,
        n_initial_points=20,
        maximizer="random",
        # acquisition_func="log_ei",# BUG: log_ei seems to only work when batch size == 1.
        acquisition_func="ei",
        # feature_map: Encoder = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 1000,
        epochs: int = 1,
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
        """
        super().__init__(
            space=space,
            seed=seed,
            n_initial_points=n_initial_points,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            # feature_map=None,
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            normalize_inputs=normalize_inputs,
        )

    def _initialize_model(self):
        self.model = OrionABLRWrapper(
            self.space,
            alpha=self.alpha,
            beta=self.beta,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            normalize_inputs=self.normalize_inputs,
        )
        # lower, upper = build_bounds(self.space)
        # n_hypers = infer_n_hypers(build_kernel(lower, upper))

    def set_state(self, state_dict: Dict):
        return super().set_state(state_dict)

    def suggest(self, num: int = None) -> List:
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        # BUG: Getting singular cholesky matrices when suggesting with too few points?
        return super().suggest(num=num)


class OrionABLRWrapper(ABLR):
    @property
    def lower(self) -> np.ndarray:
        return build_bounds(self.space)[0]

    @property
    def upper(self) -> np.ndarray:
        return build_bounds(self.space)[1]

    def set_state(self, state_dict: Dict) -> None:
        """Restore the state of the optimizer"""
        # TODO: Might be bugs with the shape of y_mean and y_var
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> Tuple:
        random_state: Dict = state_dict.pop("rng", {})
        if random_state:
            torch.random.set_rng_state(random_state["torch"])
        return super().load_state_dict(state_dict, strict=strict)

    def state_dict(
        self, destination: str = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict:
        state_dict = super().state_dict(destination, prefix, keep_vars)
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

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(pytorch_seed)
            torch.backends.cudnn.deterministic = True

        torch.manual_seed(pytorch_seed)

    # TODO: Not yet adding warm-start support here.
    # def warm_start(
    #     self, warm_start_trials: Dict[ExperimentInfo, List[Trial]]
    # ) -> None:
    #     """ TODO: Warm-start the ABLR algorithm. """
    #     return super().warm_start(warm_start_trials)


def ablr_main():
    import matplotlib.pyplot as plt
    import numpy as np
    from warmstart.tasks.quadratics import QuadraticsTask, QuadraticsTaskHParams

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
