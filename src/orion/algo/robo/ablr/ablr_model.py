""" Re-implementation of the ABLR model from [1].

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
import warnings
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import tqdm
from orion.algo.space import Space
from pybnn.base_model import BaseModel
from robo.models.base_model import BaseModel as BaseModel_
from torch import Tensor, nn
from torch.linalg import norm
from torch.utils.data import DataLoader, TensorDataset

from orion.algo.robo.ablr.encoders import Encoder, NeuralNetEncoder
from orion.algo.robo.ablr.normal import Normal

logger = get_logger(__name__)


class ABLR(nn.Module, BaseModel, BaseModel_):
    """Surrogate model for a single task."""

    @dataclass
    class HParams:
        alpha: float = 1.0
        beta: float = 1.0
        learning_rate: float = 0.001
        batch_size: int = 1000
        epochs: int = 1

    def __init__(
        self,
        space: Space,
        feature_map: Union[Encoder, Type[Encoder]] = NeuralNetEncoder,
        alpha: float = 1.0,
        beta: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 1000,
        epochs: int = 1,
        normalize_inputs: bool = True,
    ):
        if feature_map is None:
            feature_map = NeuralNetEncoder
        super().__init__()
        self.space: Space = space
        self.task_id: Optional[int] = None
        # TODO: This isn't quite right I think.
        self.input_dims = len(self.space)

        self.feature_space_dims = 100
        if not isinstance(feature_map, nn.Module):
            feature_map_type = feature_map
            # TODO: change this, create the feature map directly? or not?
            feature_map = feature_map_type(
                input_space=self.space, out_features=self.feature_space_dims
            )
        self.feature_map = feature_map

        self.batch_size = batch_size
        self.epochs = epochs

        self.feature_map = feature_map
        self.alpha = nn.Parameter(torch.as_tensor([alpha], dtype=torch.float))
        self.beta = nn.Parameter(torch.as_tensor([beta], dtype=torch.float))

        self.learning_rate = learning_rate
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # BUG: Getting 'LBFGS' object doesn't have a _params attribute?!
        # self.optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        self.optimizer = PatchedLBFGS(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self._predict_dist: Optional[Callable[[Tensor], Normal]] = None

        self.normalize_inputs = normalize_inputs

        self.x_mean: Tensor
        self.x_var: Tensor
        self.register_buffer("x_mean", torch.zeros([self.input_dims]))
        self.register_buffer("x_var", torch.ones([self.input_dims]))

        self.y_mean: Tensor
        self.y_var: Tensor
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_var", torch.ones(()))

    def state_dict(
        self, destination: Any = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict:
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> Tuple:
        optim_state: Dict = state_dict.pop("optimizer")
        self.optimizer.load_state_dict(optim_state)
        assert self.optimizer._params is not None
        return super().load_state_dict(state_dict, strict=strict)

    def predict_dist(self, x_test: Union[Tensor, np.ndarray]) -> Normal:
        """Return the predictive distribution for the given points."""
        with torch.no_grad():
            # TODO: We shouldn't be doing this entire forward pass at each
            # prediction step. Just debugging atm.
            if self._predict_dist is None:
                _, self._predict_dist = self(self.X, self.y)
            y_pred_dist = self._predict_dist(x_test)
        return y_pred_dist

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the predictive mean and variance for the given points."""
        y_pred_distribution = self.predict_dist(X_test)
        return y_pred_distribution.mean.numpy(), y_pred_distribution.variance.numpy()

    def forward(
        self, x_train: Tensor, y_train: Tensor
    ) -> Tuple[Tensor, Callable[[Tensor], Normal]]:
        """Return the training loss and a function that will give the
        predictive distribution over y for a given (un-normalized) input x_test.

        NOTE: Assumes that `x_train`, `y_train` and an eventual `x_test` are
        NOT already rescaled/normalized!
        """
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        # TODO: Do we normalize the portion that has the contextual information?
        if all(self.x_mean == 0):
            self.x_mean = x_train.mean(0)
        if all(self.x_var == 1):
            x_var = x_train.var(0)
            # minimum variance, because the context vectors are actually always
            # the same, so the variance along that dimension is 0, which makes
            # NaN values in the normalized inputs.
            self.x_var = torch.maximum(x_var, torch.zeros_like(x_var) + 1e-5)
        if self.y_mean == 0:
            self.y_mean = y_train.mean(0)
        if self.y_var == 1:
            y_var = y_train.var(0)
            self.y_var = torch.maximum(y_var, torch.zeros_like(y_var) + 1e-5)

        assert x_train.shape[-1] == self.x_mean.shape[-1], (x_train.shape, self.x_mean)
        x_train = torch.as_tensor(x_train, dtype=torch.float)
        y_train = torch.as_tensor(y_train, dtype=torch.float)
        x_train = self.normalize_x(x_train) if self.normalize_inputs else x_train
        y_train = self.normalize_y(y_train) if self.normalize_inputs else y_train
        neg_mll, predictive_distribution_fn = self.predictive_mean_and_variance(
            x_train, y_train
        )

        def predict_normalized(x_test: Tensor) -> Normal:
            x_test = torch.as_tensor(x_test, dtype=torch.float)
            if self.normalize_inputs:
                x_test = self.normalize_x(x_test)
            y_pred_dist = predictive_distribution_fn(x_test)
            y_pred_dist = self.unnormalize_y(y_pred_dist)
            return y_pred_dist

            # y_pred = y_pred_dist.rsample()
            # return self.unnormalize_y(y_pred)

        return neg_mll, predict_normalized

    def get_loss(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        return self(x, y)[0]

    def train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = None):
        # Save the dataset so we can use it to predict the mean and variance later.
        # NOTE: This do_optimize isn't used, but is a parameter of the Robo base class.
        self.X = torch.as_tensor(X, dtype=torch.float)
        self.y = torch.as_tensor(y, dtype=torch.float).reshape([-1])
        self.x_mean = self.X.mean(0)
        self.x_var = self.X.var(0)
        self.y_mean = self.y.mean(0)
        self.y_var = self.y.var(0)

        dataset = TensorDataset(self.X, self.y)

        n = len(dataset)
        train_dataset = dataset
        # n_train = int(0.8 * n)
        # train_dataset = dataset[:n_train]
        # valid_dataset = dataset[n_train:]

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        # TODO: No validation dataset for now.
        # valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=True)

        outer_pbar = tqdm.tqdm(range(self.epochs), desc="Epoch")
        for epoch in outer_pbar:
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

                    # This is a bit weird, idk if this is how its supposed to be
                    # used:
                    def closure():
                        return self(self.X, self.y)[0]

                    self.optimizer.step(closure=closure)

    def get_incumbent(self):
        x, y = super().get_incumbent()
        if isinstance(x, Tensor):
            x = x.cpu().numpy()
        if isinstance(y, Tensor):
            y = y.cpu().numpy()
        return x, y

    # @staticmethod
    def _normalize(
        self, x: np.ndarray, mean: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        assert x.shape, x

        if isinstance(x, np.ndarray):
            assert x.numel()
            mean = mean.cpu().numpy() if isinstance(mean, Tensor) else mean
            var = var.cpu().numpy() if isinstance(var, Tensor) else var
            x -= mean
            x /= var
            return x

        if isinstance(x, Tensor):
            in_shape = x.shape
            assert x.numel(), f"Empty x? {x}, {self.space}, {self.task_id}, {self.task}"
            x = x.reshape([-1, x.shape[-1]])

            mean = (
                torch.as_tensor(mean).type_as(x)
                if isinstance(mean, np.ndarray)
                else mean
            )
            mean = torch.atleast_2d(mean)
            var = (
                torch.as_tensor(var).type_as(x) if isinstance(var, np.ndarray) else var
            )
            var = torch.atleast_2d(var)

            assert len(x.shape) == 2
            assert len(mean.shape) == 2
            assert len(var.shape) == 2
            mean = mean.reshape([-1, x.shape[-1]])
            mean = mean.expand_as(x)
            var = var.reshape([-1, x.shape[-1]])
            var = var.expand_as(x)

            assert x.shape == mean.shape, (x.shape, mean.shape)

            return x.sub_(mean).div_(var).reshape(in_shape)

            x /= var
            return x

        assert False

    @staticmethod
    def _unnormalize(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            mean = mean.cpu().numpy() if isinstance(mean, Tensor) else mean
            var = var.cpu().numpy() if isinstance(var, Tensor) else var
            x[:, : mean.shape[-1]] *= var
            x[:, : mean.shape[-1]] += mean
            return x

        if isinstance(x, (Tensor, Normal)):
            mean = (
                torch.as_tensor(mean).type_as(x)
                if isinstance(mean, np.ndarray)
                else mean
            )
            # mean = torch.atleast_2d(mean)
            var = (
                torch.as_tensor(var).type_as(x) if isinstance(var, np.ndarray) else var
            )
            # var = torch.atleast_2d(var)
            x *= var
            x += mean
            return x

        assert False, x

    def normalize_x(self, x: Union[np.ndarray, Tensor]):
        # return self._normalize(x, self.x_mean, self.x_var)
        x -= self.x_mean
        x /= self.x_var
        return x

    def unnormalize_x(self, x: Tensor):
        return self._unnormalize(x, self.x_mean, self.x_var)
        x *= self.x_var
        x += self.x_mean
        return x

    def normalize_y(self, y: Union[Tensor, Normal]) -> Union[Tensor, Normal]:
        return self._normalize(y.reshape([-1, 1]), self.y_mean, self.y_var).reshape(
            y.shape
        )

        y -= self.y_mean
        y /= self.y_var
        return y

    def unnormalize_y(self, y: Union[Tensor, Normal]) -> Union[Tensor, Normal]:
        return self._unnormalize(y, self.y_mean, self.y_var)

        y *= self.y_var
        y += self.y_mean
        return y

    def marginal_log_likelihood(self, theta):
        """
        Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            lnlikelihood + prior
        """
        return -self.negative_mll(theta)

    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        return self(self.X, self.y)[0]

    def predictive_mean_and_variance(
        self, x_t: Tensor, y_t: Tensor
    ) -> Tuple[Tensor, Callable[[Tensor], Normal]]:
        """Trying to replicate the first part of section 3.1 of the ABLR paper.

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
            k_t = r_t * phi_t.T @ phi_t
            l_t = try_get_cholesky(k_t)
            l_t_inv = try_get_cholesky_inverse(l_t)
            e_t = torch.chain_matmul(l_t_inv, phi_t.T, y_t)

            negative_log_marginal_likelihood = (
                -n / 2 * torch.log(self.beta)
                + self.beta / 2 * ((y_t ** 2).sum() - r_t * (e_t ** 2).sum())
                + l_t.diag().log().sum()
            )

            def predict_mean(new_x: Tensor) -> Tensor:
                phi_t_star = self.feature_map(new_x)
                # TODO: Adding the .T on phi_t_star below, since there seems to be some bugs in the shapes..
                mean = r_t * torch.chain_matmul(e_t.T, l_t_inv, phi_t_star.T)
                return mean.reshape([new_x.shape[0], 1])

            def predict_variance(new_x: Tensor) -> Tensor:
                phi_t_star = self.feature_map(new_x)
                # BUG: Seem to be getting negative values for the variance?
                variance = 1 / self.alpha * ((l_t_inv @ phi_t_star.T) ** 2).sum()
                if (variance < 0).any():
                    min_variance = variance.min()
                    logger.critical(
                        RuntimeError(
                            f"Variance has negative values! (min={min_variance})"
                        )
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

        # N <= D:
        # (Following the supplementary material)
        # TODO: This hasn't been fully tested yet.
        # k_t = r_t * phi_t @ phi_t.T + torch.eye(n)

        # BUG: `y_t` has way too low mean and variance:
        # tensor(-3.7835e-12), tensor(4.7180e-05)
        # assert False, (y_t.mean(), y_t.std())
        try:
            k_t = torch.eye(n) + r_t * phi_t @ phi_t.T
            E_t = try_get_cholesky(k_t)
            E_t_inv = try_get_cholesky_inverse(E_t)  # TODO: Is it k_t or E_t here?
            negative_log_marginal_likelihood = (
                -n / 2 * torch.log(self.beta)
                + self.beta / 2 * ((E_t_inv @ y_t) ** 2).sum()
                + torch.log(E_t.diag()).sum()
            )
        except RuntimeError as e:
            # Often happens that we get NaNs in the matrices (probably due to low
            # variance in y_t?)
            # FIXME: Trying to return random values at this point:
            return torch.zeros(1), lambda x: Normal(loc=y_t.mean(), variance=y_t.var())

        def predict_mean(new_x: Tensor) -> Tensor:
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

        def predict_variance(new_x: Tensor) -> Tensor:
            phi_t_star = self.feature_map(new_x)
            y_t_norm = (y_t ** 2).sum()
            second_term = torch.chain_matmul(
                E_t_inv,
                phi_t,
                phi_t_star.T,
            )
            variance = (1 / self.alpha) * (y_t_norm - r_t * (second_term ** 2).sum(0))
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
            # logger.debug(f"Predicted variance: {variance}")
            assert (variance > 0).all()
            predictive_dist = Normal(mean, variance)
            return predictive_dist

        return negative_log_marginal_likelihood, predictive_distribution


class PatchedLBFGS(torch.optim.LBFGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert False, self._params

    def __getstate__(self):
        state = super().__getstate__()
        state["_params"] = self._params
        state["_numel_cache"] = self._numel_cache
        return state

    def state_dict(self, *args, **kwargs) -> Dict:
        # TODO: When is __getstate__ called, vs state_dict?
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["_params"] = self._params
        state_dict["_numel_cache"] = self._numel_cache
        return state_dict

    def __setstate__(self, state):
        super().__setstate__(state)


def offset_function(some_matrix: Tensor, attempt: int, max_attempts: int) -> Tensor:
    if some_matrix.shape[-2] == some_matrix.shape[-1]:
        # If the matrix is square, add an offset to the diagonal:
        offset = 0.1 * (2 ** (attempt - 1))
        return some_matrix + torch.eye(some_matrix.shape[-1]) * offset
    else:
        # Add progressively larger random noise?
        noise_std = 0.1 * (2 ** (attempt - 1))
        return some_matrix + torch.randn_like(some_matrix) * noise_std


def try_function(
    function: Callable[[Tensor], Tensor],
    some_matrix: Tensor,
    max_attempts: int = 10,
    offset_function: Callable[[Tensor, int, int], Tensor] = offset_function,
) -> Tensor:
    """Attempt to get the choleksy of the given matrix, adding progressively
    larger offset/noise matrices until it works, else raises an error.
    """
    try:
        return function(some_matrix)
    except RuntimeError:
        pass

    result: Optional[Tensor] = None
    for attempt in range(max_attempts + 1):
        m: Tensor
        if attempt == 0:
            m = some_matrix
        else:
            m = offset_function(some_matrix, attempt, max_attempts)

        try:
            result = function(m)
            if attempt > 0:
                logger.debug(
                    f"Managed to get the operation to work after {attempt} attemps."
                )
            return result

        except RuntimeError as e:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"{function.__name__} didn't work, even after {attempt} attempts:\n"
                    f"{e}\n"
                    f"(matrix: {some_matrix})"
                ) from e


from functools import partial

try_get_cholesky = partial(try_function, torch.cholesky)
try_get_cholesky_inverse = partial(try_function, torch.cholesky_inverse)


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
