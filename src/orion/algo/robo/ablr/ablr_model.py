""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from logging import getLogger as get_logger
from typing import Any, Callable, OrderedDict, TypeVar

import numpy as np
import torch
import tqdm
from orion.algo.space import Space
from robo.models.base_model import BaseModel
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset

from orion.algo.robo.ablr.encoders import Encoder, NeuralNetEncoder
from orion.algo.robo.ablr.normal import Normal
from orion.algo.robo.ablr.patched_lbfgs import PatchedLBFGS
from orion.algo.robo.ablr.utils import try_function
from orion.algo.robo.base import Model, build_bounds

T = TypeVar("T", np.ndarray, Tensor, Normal)
logger = get_logger(__name__)


class AblrNetwork(nn.Module):
    """Network used by the ABLR model."""

    def __init__(
        self,
        feature_map: Encoder,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
    ):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.feature_map = feature_map

        self.alpha = Parameter(torch.as_tensor([self.initial_alpha], dtype=torch.float))
        self.beta = Parameter(torch.as_tensor([self.initial_beta], dtype=torch.float))

        self._predict_dist: Callable[[Tensor], Normal] | None = None

        input_dims: int = self.feature_map.in_features
        self.x_mean: Tensor
        self.x_var: Tensor
        self.register_buffer("x_mean", torch.zeros([input_dims]))
        self.register_buffer("x_var", torch.ones([input_dims]))

        self.y_mean: Tensor
        self.y_var: Tensor
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_var", torch.ones(()))

    def forward(
        self, x_train: Tensor, y_train: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Normal]]:
        """Return the training loss and a function that will give the
        predictive distribution over y for a given (un-normalized) input x_test.

        NOTE: Assumes that `x_train`, `y_train` and an eventual `x_test` are
        NOT already rescaled/normalized!
        """
        # Change dtypes and devices if necessary.
        x_train = x_train.type_as(self.x_mean)
        y_train = y_train.type_as(self.y_mean)

        x_train = self._normalize_x(x_train)
        y_train = self._normalize_y(y_train)

        assert x_train.shape[-1] == self.x_mean.shape[-1], (x_train.shape, self.x_mean)
        neg_mll, predictive_distribution_fn = self.predictive_mean_and_variance(
            x_train, y_train
        )

        def predictive_distribution(x_test: Tensor) -> Normal:
            x_test = self._normalize_x(x_test)
            x_test = torch.as_tensor(x_test).type_as(self.x_mean)
            y_pred_dist = predictive_distribution_fn(x_test)  # type: ignore
            y_pred_dist = self._unnormalize_y(y_pred_dist)
            return y_pred_dist

        return neg_mll, predictive_distribution

    def predictive_mean_and_variance(
        self, x_t: Tensor, y_t: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Normal]]:
        """Replicates the first part of section 3.1 of the ABLR paper.

        Returns the 'learning criterion' (i.e. the loss term), and a function
        that gives the predictive distribution (Normal) for a given (normalized)
        test input.
        """
        assert len(x_t.shape) == 2
        n = x_t.shape[0]
        y_t = y_t.reshape([-1, 1])

        # Feature map for the dataset of task t.
        phi_t = self.feature_map(x_t.float())
        d = phi_t.shape[-1]

        r_t = self.beta / self.alpha

        if n > d:
            return self._n_greater_than_d(x_t=x_t, y_t=y_t, r_t=r_t, phi_t=phi_t)
        else:
            return self._n_smaller_than_d(x_t=x_t, y_t=y_t, r_t=r_t, phi_t=phi_t)

    def _n_greater_than_d(
        self,
        x_t: Tensor,
        y_t: Tensor,
        r_t: Tensor,
        phi_t: Tensor,
    ):
        n = x_t.shape[0]

        k_t = r_t * (phi_t.T @ phi_t)
        l_t = try_function(torch.linalg.cholesky, k_t, max_attempts=2)
        k_t_inv = try_function(torch.cholesky_inverse, l_t, max_attempts=1)
        l_t_inv = l_t.T @ k_t_inv
        e_t = torch.linalg.multi_dot([l_t_inv, phi_t.T, y_t])

        negative_log_marginal_likelihood = (
            -n / 2 * torch.log(self.beta)
            + self.beta / 2 * ((y_t**2).sum() - r_t * (e_t**2).sum())
            + l_t.diag().log().sum()
        )

        def predict_mean(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            # NOTE: Adding the .T on phi_t_star below, since there seems to be some bugs in the
            # shapes..
            mean = r_t * torch.linalg.multi_dot([e_t.T, l_t_inv, phi_t_star.T])
            return mean.reshape([new_x.shape[0], 1])

        def predict_variance(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            # BUG: Seem to be getting negative values for the variance?
            variance = 1 / self.alpha * ((l_t_inv @ phi_t_star.T) ** 2).sum()
            if (variance < 0).any():
                min_variance = variance.min()
                logger.critical(
                    RuntimeError(f"Variance has negative values! (min={min_variance})")
                )
                variance = variance + torch.abs(min_variance) + 1e-8
            assert (variance > 0).all()
            return variance

        def predictive_distribution(new_x: Tensor) -> Normal:
            mean = predict_mean(new_x)
            variance = predict_variance(new_x)
            predictive_dist = Normal(mean, variance)
            return predictive_dist

        return negative_log_marginal_likelihood, predictive_distribution

    def _n_smaller_than_d(self, x_t: Tensor, y_t: Tensor, r_t: Tensor, phi_t: Tensor):
        n = x_t.shape[0]
        # N <= D: Fewer points than dimensions.
        # (Following the supplementary material)

        try:
            k_t = torch.eye(n) + r_t * phi_t @ phi_t.T
            E_t = try_function(torch.linalg.cholesky, k_t, max_attempts=2)
            k_t_inv = torch.cholesky_inverse(E_t)
            E_t_inv = E_t.T @ k_t_inv
            negative_log_marginal_likelihood = (
                -n / 2 * torch.log(self.beta)
                + self.beta / 2 * ((E_t_inv @ y_t) ** 2).sum()
                + torch.log(E_t.diag()).sum()
            )
        except RuntimeError as err:
            # Often happens that we get NaNs in the matrices (probably due to low
            # variance in y_t?)
            # FIXME: Trying to return random values at this point:
            logger.critical(
                "Unable to make a prediction: %s. Predictive distribution will be "
                "independent from input.",
                err,
            )
            return (
                torch.zeros(1),  # Loss
                lambda _: Normal(
                    loc=y_t.mean(),
                    scale=torch.max(y_t.var(), torch.ones_like(y_t) * 1e-7),
                ),
            )

        def _predict_mean(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            return r_t * torch.linalg.multi_dot(
                [
                    # Note: (Unpacked the (E_t_inv @ y_t).T term)
                    y_t.T,  # [1, 40]
                    E_t_inv,  # [40, 40]
                    E_t_inv.T,  # [40, 40]
                    phi_t,  # [40, 50]
                    phi_t_star.T,  # [50, N]
                ],
            )

        def _predict_variance(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            y_t_norm = (y_t**2).sum()
            second_term = torch.linalg.multi_dot(
                [
                    E_t_inv,
                    phi_t,
                    phi_t_star.T,
                ]
            )
            variance: Tensor = (1 / self.alpha) * (
                y_t_norm - r_t * (second_term**2).sum(0)
            )
            if (variance < 0).any():
                min_variance = variance.min()
                logger.critical(
                    RuntimeError(f"Variance has negative values! (min={min_variance})")
                )
                variance = variance + torch.abs(min_variance) + 1e-8
            if variance.isnan().any():
                logger.error("Variance is NaN!")
            return variance

        def _predictive_distribution(new_x: Tensor) -> Normal:
            mean = _predict_mean(new_x)
            variance = _predict_variance(new_x)
            # logger.debug(f"Predicted variance: {variance}")
            if mean.isnan().any():
                logger.error("Mean is NaN!")
            if variance.isnan().any() or (variance <= 0).any():
                logger.error("Variance is NaN or negative!")
                variance = torch.max(self.y_var, torch.ones_like(self.y_var) * 1e-3)
                # variance = torch.ones_like(mean) * 1000
            predictive_dist = Normal(mean, variance)
            return predictive_dist

        return negative_log_marginal_likelihood, _predictive_distribution

    def get_loss(self, x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> Tensor:
        """Gets the loss for the given input and target."""
        return self(x, y)[0]

    def _normalize_x(self, x: T) -> T:
        x -= self.x_mean
        x = x / self.x_var
        return x

    def _unnormalize_x(self, x: T):
        x *= self.x_var
        x += self.x_mean
        return x

    def _normalize_y(self, y: T) -> T:
        y -= self.y_mean
        y = y / self.y_var
        return y

    def _unnormalize_y(self, y: T) -> T:
        y = y * self.y_var
        y += self.y_mean
        return y


# pylint: disable=too-many-instance-attributes
class ABLR(BaseModel, Model):
    """Surrogate model for a single task."""

    @dataclass
    class HParams:
        """Hyper-Parameters of the ABLR algorithm."""

        alpha: float = 1.0
        beta: float = 1.0
        learning_rate: float = 0.001
        batch_size: int = 1000
        epochs: int = 1
        feature_space_dims: int = 16

    def __init__(
        self,
        space: Space,
        feature_map: Encoder | type[Encoder] = NeuralNetEncoder,
        hparams: ABLR.HParams | dict | None = None,
    ):
        super().__init__()
        self.space: Space = space
        if isinstance(hparams, dict):
            hparams = self.HParams(**hparams)
        self.hparams = hparams or self.HParams()
        self.normalize_inputs = True

        if not isinstance(feature_map, nn.Module):
            feature_map_type = feature_map
            feature_map = feature_map_type(
                input_space=self.space, out_features=self.hparams.feature_space_dims
            )
        self.network = AblrNetwork(
            feature_map=feature_map,
            initial_alpha=self.hparams.alpha,
            initial_beta=self.hparams.beta,
        )
        # NOTE: The "'LBFGS' object doesn't have a _params attribute" bug is fixed with the
        # PatchedLBFGS class.
        # self.optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        self.optimizer = PatchedLBFGS(
            self.network.parameters(), lr=self.hparams.learning_rate
        )
        self.lower, self.upper = build_bounds(self.space)

        self.x_tensor: Tensor | None = None
        self.y_tensor: Tensor | None = None

    def state_dict(
        self,
    ) -> dict[str, Tensor | Any]:
        state_dict: dict[str, Tensor | Any] = {}
        state_dict["network"] = self.network.state_dict()
        state_dict["rng"] = torch.random.get_rng_state()
        state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Orion hook used to restore the state of the algorithm."""
        state_dict = state_dict.copy()
        super().set_state(state_dict)

        network_state = OrderedDict(state_dict.pop("network"))
        self.network.load_state_dict(network_state, strict=True)

        optim_state: dict = state_dict.pop("optimizer")
        self.optimizer.load_state_dict(optim_state)

        rng: Tensor = state_dict.pop("rng")
        torch.random.set_rng_state(rng)

    def seed(self, seed: int) -> None:
        """no-op. RoBO_ABLR is expected to seed all pytorch RNGs."""
        # No need to do anything here really, since the ROBO_ABLR class already seeds all the
        # pytorch RNG.

    # pylint: disable=unused-argument,arguments-differ
    def train(
        self, X: np.ndarray, y: np.ndarray, do_optimize: bool | None = None
    ) -> None:
        """Training method for the algorithm, as a RoBO Model subclass.

        NOTE: This do_optimize parameter is not inherited from the robo BaseModel class, but is
        passed by the RoBO Bayesian optimization solver.
        """
        # RoBO methods assumes that model.X and model.y are numpy arrays. For CPU tensors it works
        # fine, but for CUDA tensors it doesn't, so it's best to keep the numpy and torch variants
        # separately.
        self.X = X
        self.y = y
        # Save the dataset so we can use it to predict the mean and variance later.
        self.x_tensor = torch.as_tensor(X).type_as(self.network.x_mean)
        self.y_tensor = torch.as_tensor(y).type_as(self.network.y_mean).reshape([-1])

        # Set the means and variances at the start of training.
        self.network.x_mean = self.x_tensor.mean(0)
        self.network.x_var = self.x_tensor.var(0).clamp_min_(1e-6)
        self.network.y_mean = self.y_tensor.mean(0)
        self.network.y_var = self.y_tensor.var(0).clamp_min_(1e-6)

        dataset = TensorDataset(self.x_tensor, self.y_tensor)

        # TODO: No validation dataset for now.
        train_dataset = dataset
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )
        # n = len(dataset)
        # n_train = int(0.8 * n)
        # train_dataset = dataset[:n_train]
        # valid_dataset = dataset[n_train:]
        # valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=True)

        outer_pbar = tqdm.tqdm(range(self.hparams.epochs), desc="Epoch", disable=True)
        for epoch in outer_pbar:
            logger.debug("Start of epoch %s", epoch)
            with tqdm.tqdm(
                train_dataloader, position=1, leave=False, disable=True
            ) as inner_pbar:
                for i, (x_batch, y_batch) in enumerate(inner_pbar):

                    self.optimizer.zero_grad()
                    loss, pred_fn = self.network(x_batch, y_batch)
                    # TODO: This predictive distribution thingy, should it be based on the latest
                    # batch? Or the entire dataset?
                    # NOTE: For now, we update the predictive distribution function after each
                    # successful batch batch, so that if something goes wrong and we return early,
                    # we can still use the predictive distribution function to predict the mean
                    # and variance of samples.
                    # Additionally, after all the training is done, we set the predictive function
                    # to be based on the entire dataset.
                    self._predict_dist = pred_fn

                    if loss.requires_grad:
                        loss.backward()
                    else:
                        logger.warning(
                            RuntimeWarning(
                                f"Couldn't train at step {i}, there must have been an "
                                f"error in the ABLR algebra."
                            )
                        )

                    outer_pbar.set_postfix(
                        {
                            "Loss:": f"{loss.item():.3f}",
                            "alpha": self.network.alpha.item(),
                            "beta": self.network.beta.item(),
                        }
                    )
                    # NOTE: LBFGS optimizer requires a closure that recomputes the loss.
                    closure = partial(
                        torch.no_grad()(self.network.get_loss), x_batch, y_batch
                    )

                    try:
                        self.optimizer.step(closure=closure)  # type: ignore
                    except RuntimeError as err:
                        warnings.warn(
                            RuntimeWarning(
                                f"Ending training early because of error: {err}"
                            )
                        )
                        return

        # Create the predictive distribution function based on the entire dataset.
        with torch.no_grad():
            dataset_loss, self._predict_dist = self.network(
                self.x_tensor, self.y_tensor
            )
            avg_sample_loss = dataset_loss / len(self.x_tensor)
            logger.info("Dataset loss at the end of training: %s", avg_sample_loss)

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and variance of the predictive distribution for the given points."""
        y_pred_distribution = self.predict_dist(X_test)
        return (
            y_pred_distribution.mean.cpu().numpy(),
            y_pred_distribution.variance.cpu().numpy(),
        )

    @torch.no_grad()
    def predict_dist(self, x_test: Tensor | np.ndarray) -> Normal:
        """Return the predictive distribution for the given points."""
        if self._predict_dist is None:
            raise RuntimeError(
                "No predictive distribution available since model hasn't been trained."
            )
        x_test = torch.as_tensor(x_test).type_as(self.network.x_mean)
        y_pred_dist = self._predict_dist(x_test)
        return y_pred_dist

    def get_incumbent(self) -> tuple[np.ndarray, float]:
        x, y = super().get_incumbent()
        if isinstance(x, Tensor):
            x = x.cpu().numpy()
        if isinstance(y, Tensor):
            y = y.cpu().numpy()
        return x, y
