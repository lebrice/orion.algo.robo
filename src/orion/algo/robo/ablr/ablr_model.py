""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from functools import partial
from logging import getLogger as get_logger
from typing import Any, Callable, OrderedDict, TypeVar, overload

import numpy as np
import torch
import tqdm
from orion.algo.space import Space
from pybnn.base_model import BaseModel
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Protocol

from orion.algo.robo.ablr.encoders import Encoder, NeuralNetEncoder
from orion.algo.robo.ablr.normal import Normal
from orion.algo.robo.base import Model, build_bounds

T = TypeVar("T", np.ndarray, Tensor)
logger = get_logger(__name__)


# pylint: disable=too-many-instance-attributes
class ABLR(nn.Module, BaseModel, Model):
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
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.space: Space = space
        if isinstance(hparams, dict):
            hparams = self.HParams(**hparams)
        self.hparams = hparams or self.HParams()
        self.normalize_inputs = normalize_inputs

        if not isinstance(feature_map, nn.Module):
            feature_map_type = feature_map
            feature_map = feature_map_type(
                input_space=self.space, out_features=self.hparams.feature_space_dims
            )
        self.feature_map = feature_map
        self.alpha = Parameter(torch.as_tensor([self.hparams.alpha], dtype=torch.float))
        self.beta = Parameter(torch.as_tensor([self.hparams.beta], dtype=torch.float))

        # NOTE: The "'LBFGS' object doesn't have a _params attribute" bug is fixed with the
        # PatchedLBFGS class.
        # self.optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        self.optimizer = PatchedLBFGS(self.parameters(), lr=self.hparams.learning_rate)
        self._predict_dist: Callable[[Tensor], Normal] | None = None

        # NOTE: This is OK since the algo has requirements for flattened reals.
        input_dims = len(self.space)
        self.x_mean: Tensor
        self.x_var: Tensor
        self.register_buffer("x_mean", torch.zeros([input_dims]))
        self.register_buffer("x_var", torch.ones([input_dims]))

        self.y_mean: Tensor
        self.y_var: Tensor
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_var", torch.ones(()))

        self.lower, self.upper = build_bounds(self.space)

    def state_dict(  # type: ignore
        self,
        destination=None,
        prefix="",
        keep_vars: bool = False,
    ) -> OrderedDict[str, Tensor | Any]:
        state_dict: OrderedDict[str, Any] = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars  # type: ignore
        )
        state_dict["rng"] = torch.random.get_rng_state()
        state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> tuple:
        optim_state: dict = state_dict.pop("optimizer")
        self.optimizer.load_state_dict(optim_state)
        rng: Tensor = state_dict.pop("rng")
        torch.random.set_rng_state(rng)
        return super().load_state_dict(state_dict, strict=strict)  # type: ignore

    def set_state(self, state_dict: dict) -> None:
        """Orion hook used to restore the state of the algorithm."""
        self.load_state_dict(state_dict)

    def seed(self, seed: int) -> None:
        """no-op. RoBO_ABLR is expected to seed all pytorch RNGs. """
        # No need to do anything here really, since the ROBO_ABLR class already seeds all the
        # pytorch RNG.

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the predictive mean and variance for the given points."""
        y_pred_distribution = self.predict_dist(X_test)
        return (
            y_pred_distribution.mean.cpu().numpy(),
            y_pred_distribution.variance.cpu().numpy(),
        )

    def predict_dist(self, x_test: Tensor | np.ndarray) -> Normal:
        """Return the predictive distribution for the given points."""
        with torch.no_grad():
            if self._predict_dist is None:
                # re-create the predictive distribution.
                # NOTE: This gets overwritten every time a new point is observed, so we know for
                # sure that this is always the most up-to-date predictions.
                _, self._predict_dist = self(self.X, self.y)
            y_pred_dist = self._predict_dist(x_test)
        return y_pred_dist

    def forward(
        self, x_train: Tensor, y_train: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Normal]]:
        """Return the training loss and a function that will give the
        predictive distribution over y for a given (un-normalized) input x_test.

        NOTE: Assumes that `x_train`, `y_train` and an eventual `x_test` are
        NOT already rescaled/normalized!
        """
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        # TODO: Do we normalize the portion that has the contextual information?
        self.x_mean = x_train.mean(0)
        # minimum variance, because the context vectors are actually always
        # the same, so the variance along that dimension is 0, which makes
        # NaN values in the normalized inputs.
        x_var = x_train.var(0)
        x_var[x_var.isnan()] = 1e-6
        x_var[x_var < 1e-6] = 1e-6
        self.x_var = x_var

        if self.y_mean == 0:
            self.y_mean = y_train.mean(0)
        if self.y_var == 1:
            y_var = y_train.var(0)
            self.y_var = torch.maximum(y_var, torch.zeros_like(y_var) + 1e-5)

        assert x_train.shape[-1] == self.x_mean.shape[-1], (x_train.shape, self.x_mean)
        x_train = torch.as_tensor(x_train, dtype=torch.float)
        y_train = torch.as_tensor(y_train, dtype=torch.float)
        x_train = self._normalize_x(x_train) if self.normalize_inputs else x_train
        y_train = self._normalize_y(y_train) if self.normalize_inputs else y_train
        neg_mll, predictive_distribution_fn = self.predictive_mean_and_variance(
            x_train, y_train
        )

        def predict_normalized(x_test: Tensor | np.ndarray) -> Normal:
            x_test = torch.as_tensor(x_test, dtype=torch.float)
            if self.normalize_inputs:
                x_test = self._normalize_x(x_test)
            y_pred_dist = predictive_distribution_fn(x_test)
            y_pred_dist = self._unnormalize_y(y_pred_dist)
            return y_pred_dist

            # y_pred = y_pred_dist.rsample()
            # return self.unnormalize_y(y_pred)

        return neg_mll, predict_normalized

    def _get_loss(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        """Gets the loss for the given input and target."""
        return self(x, y)[0]

    @overload
    def train(self, mode: bool = True) -> ABLR:
        """Set the model in train mode following (as a nn.Module subclass)."""
        ...

    # pylint: disable=arguments-differ
    @overload
    def train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = False) -> None:
        """Train the algorithm (as a RoBO Model subclass)."""
        ...

    # pylint: disable=signature-differs
    def train(
        self,
        *args,
        **kwargs,
    ):
        if len(args) == 1 and isinstance(args[0], bool):
            mode: bool = args[0]
            return nn.Module.train(self, mode=mode)
        if "mode" in kwargs:
            mode: bool = kwargs["mode"]
            return nn.Module.train(self, mode=mode)
        bound_args = inspect.signature(self._train_robo).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self._train_robo(
            X=bound_args.arguments["X"],
            y=bound_args.arguments["y"],
            do_optimize=bound_args.arguments["do_optimize"],
        )

    def _train_robo(
        self, X: np.ndarray, y: np.ndarray, do_optimize: bool | None = None
    ) -> None:
        """Training method for the algorithm, as a RoBO Model subclass."""

        if X is None or y is None:
            raise RuntimeError(
                "Invalid arguments. X and y can't be None, they should be numpy arrays."
            )

        # Save the dataset so we can use it to predict the mean and variance later.
        # NOTE: This do_optimize isn't used, but is a parameter of the Robo base class.
        self.X = torch.as_tensor(X, dtype=torch.float)
        self.y = torch.as_tensor(y, dtype=torch.float).reshape([-1])
        self.x_mean = self.X.mean(0)
        self.x_var = self.X.var(0)
        self.y_mean = self.y.mean(0)
        self.y_var = self.y.var(0)

        dataset = TensorDataset(self.X, self.y)

        train_dataset = dataset
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )
        # TODO: No validation dataset for now.
        # n = len(dataset)
        # n_train = int(0.8 * n)
        # train_dataset = dataset[:n_train]
        # valid_dataset = dataset[n_train:]
        # valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=True)

        outer_pbar = tqdm.tqdm(range(self.hparams.epochs), desc="Epoch")
        for epoch in outer_pbar:
            logger.debug("Start of epoch %s", epoch)
            with tqdm.tqdm(train_dataloader, position=1, leave=False) as inner_pbar:
                for i, (x_batch, y_batch) in enumerate(inner_pbar):

                    self.optimizer.zero_grad()
                    loss, pred_fn = self(x_batch, y_batch)
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
                            "alpha": self.alpha.item(),
                            "beta": self.beta.item(),
                        }
                    )
                    # NOTE: LBFGS optimizer requires a closure that recomputes the loss.
                    closure = partial(self._get_loss, x_batch, y_batch)

                    try:
                        self.optimizer.step(closure=closure)  # type: ignore
                    except RuntimeError as err:
                        warnings.warn(
                            RuntimeWarning(
                                f"Ending training early because of error: {err}"
                            )
                        )
                        return

    def get_incumbent(self) -> tuple[np.ndarray, float]:
        x, y = super().get_incumbent()
        if isinstance(x, Tensor):
            x = x.cpu().numpy()
        if isinstance(y, Tensor):
            y = y.cpu().numpy()
        return x, y

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

        k_t = r_t * phi_t.T @ phi_t
        l_t = try_get_cholesky(k_t)
        l_t_inv = try_get_cholesky_inverse(l_t)
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
            mean = r_t * torch.chain_matmul(e_t.T, l_t_inv, phi_t_star.T)
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
            E_t = try_get_cholesky(k_t)
            E_t_inv = try_get_cholesky_inverse(E_t)  # TODO: Is it k_t or E_t here?
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
            return r_t * torch.chain_matmul(
                # BUG: Still debugging some shape errors.
                # (Unpacked the (E_t_inv @ y_t).T term)
                y_t.T,  # [1, 40]
                E_t_inv,  # [40, 40]
                E_t_inv.T,  # [40, 40]
                phi_t,  # [40, 50]
                phi_t_star.T,  # [50, N]
            )

        def _predict_variance(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            y_t_norm = (y_t**2).sum()
            second_term = torch.chain_matmul(
                E_t_inv,
                phi_t,
                phi_t_star.T,
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
                mean = self.y.mean()
            if variance.isnan().any() or (variance <= 0).any():
                logger.error("Variance is NaN or negative!")
                variance = torch.max(self.y_var, torch.ones_like(self.y_var) * 1e-3)
                # variance = torch.ones_like(mean) * 1000
            predictive_dist = Normal(mean, variance)
            return predictive_dist

        return negative_log_marginal_likelihood, _predictive_distribution

    T = TypeVar("T", Tensor, np.ndarray, Normal)

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
        y *= self.y_var
        y += self.y_mean
        return y


class PatchedLBFGS(torch.optim.LBFGS):
    """Patched version of LBFGS optimizer, that correctly saves the state."""

    _params: list
    _numel_cache: int | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state["_params"] = self._params
        state["_numel_cache"] = self._numel_cache
        return state

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["_params"] = self._params
        state_dict["_numel_cache"] = self._numel_cache
        return state_dict

    def __setstate__(self, state):
        super().__setstate__(state)


# pylint: disable=too-few-public-methods
class OffsetFunction(Protocol):
    """Callable that adds some offset/noise to the given matrix to make the cholesky decomposition
    work.

    Parameters
    ----------
    some_matrix : Tensor
        some matrix that will be passed to a function like `torch.linalg.cholesky`.
    attempt : int
        the current attempt number
    max_attempts : int
        The max number of attempts.
        NOTE: Currently unused, but the idea is that if we used a "maximum possible noise"
        below, we could use `attempt` and `max_attempts` to gage how much noise to add.

    Returns
    -------
    Tensor
        `some_matrix`, with some added offset / noise.
    """

    def __call__(self, some_matrix: Tensor, attempt: int, max_attempts: int) -> Tensor:
        raise NotImplementedError


# pylint: disable=unused-argument
def offset_function(some_matrix: Tensor, attempt: int, max_attempts: int) -> Tensor:
    """Offset function that adds an offset * torch.eye(n) to the matrix if it is square.
    Otherwise, adds some random noise of progressively larger value.
    """
    if some_matrix.shape[-2] == some_matrix.shape[-1]:
        # If the matrix is square, add an offset to the diagonal:
        offset = 0.1 * (2 ** (attempt - 1))
        return some_matrix + torch.eye(some_matrix.shape[-1]) * offset
    # Add progressively larger random noise?
    noise_std = 0.1 * (2 ** (attempt - 1))
    return some_matrix + torch.randn_like(some_matrix) * noise_std


def try_function(
    function: Callable[[Tensor], Tensor],
    some_matrix: Tensor,
    max_attempts: int = 10,
    offset_function: OffsetFunction = offset_function,
) -> Tensor:
    """Attempt to apply the given function of the given matrix, adding progressively
    larger offset/noise matrices until it works, else raises an error.
    """
    result: Tensor | None = None
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    attempt = 0
    error: RuntimeError | None = None
    for attempt in range(max_attempts):
        m: Tensor
        if attempt == 0:
            m = some_matrix
        else:
            m = offset_function(some_matrix, attempt, max_attempts)

        try:
            result = function(m)
            if attempt > 0:
                logger.debug(
                    "Managed to get the operation to work after %s attempts.", attempt
                )
            return result
        except RuntimeError as e:
            error = e

    raise RuntimeError(
        f"{function.__name__} didn't work, even after {attempt} attempts:\n"
        f"{error}\n"
        f"(matrix: {some_matrix})"
    ) from (error if error is not None else None)


try_get_cholesky = partial(try_function, torch.linalg.cholesky)
try_get_cholesky_inverse = partial(try_function, torch.cholesky_inverse)
