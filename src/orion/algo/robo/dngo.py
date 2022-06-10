"""
Wrapper for RoBO with DNGO
"""
from __future__ import annotations

import logging
from typing import OrderedDict, Sequence

import numpy as np
import torch
from orion.algo.space import Space
from pybnn.dngo import (
    DNGO,
    BaseModel,
    BayesianLinearRegression,
    Net,
    Prior,
    emcee,
    optim,
    optimize,
    time,
    zero_mean_unit_var_denormalization,
    zero_mean_unit_var_normalization,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from orion.algo.robo.base import (
    AcquisitionFnName,
    MaximizerName,
    RoBO,
    WrappedRoboModel,
    build_bounds,
    build_kernel,
    infer_n_hypers,
)

logger = logging.getLogger(__name__)


# pylint: disable = too-many-instance-attributes
class OrionDNGOWrapper(DNGO, WrappedRoboModel):
    """
    Wrapper for PyBNN's DNGO model

    Parameters
    ----------
    batch_size: int
        Batch size for training the neural network
    num_epochs: int
        Number of epochs for training
    learning_rate: float
        Initial learning rate for Adam
    adapt_epoch: int
        Defines after how many epochs the learning rate will be decayed by a factor 10
    n_units_1: int
        Number of units in layer 1
    n_units_2: int
        Number of units in layer 2
    n_units_3: int
        Number of units in layer 3
    alpha: float
        Hyperparameter of the Bayesian linear regression
    beta: float
        Hyperparameter of the Bayesian linear regression
    prior: Prior object
        Prior for alpa and beta. If set to None the default prior is used
    do_mcmc: bool
        If set to true different values for alpha and beta are sampled via MCMC from the marginal
        log likelihood. Otherwise the marginal log likehood is optimized with scipy fmin function.
    n_hypers : int
        Number of samples for alpha and beta
    chain_length : int
        The chain length of the MCMC sampler
    burnin_steps: int
        The number of burnin steps before the sampling procedure starts
    normalize_output : bool
        Zero mean unit variance normalization of the output values
    normalize_input : bool
        Zero mean unit variance normalization of the input values
    rng: np.random.RandomState
        Random number generator

    """

    # pylint: disable = too-many-locals
    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        device: torch.device | str | None = None,
        batch_size: int = 10,
        num_epochs: int = 500,
        learning_rate: float = 0.01,
        adapt_epoch: int = 5000,
        n_units_1: int = 50,
        n_units_2: int = 50,
        n_units_3: int = 50,
        alpha: float = 1.0,
        beta: float = 1000,
        prior: Prior | None = None,
        do_mcmc: bool = True,
        n_hypers: int = 20,
        chain_length: int = 2000,
        burnin_steps: int = 2000,
        normalize_input: int = True,
        normalize_output: int = True,
        rng: np.random.RandomState | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            adapt_epoch=adapt_epoch,
            n_units_1=n_units_1,
            n_units_2=n_units_2,
            n_units_3=n_units_3,
            alpha=alpha,
            beta=beta,
            prior=prior,
            do_mcmc=do_mcmc,
            n_hypers=n_hypers,
            chain_length=chain_length,
            burnin_steps=burnin_steps,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            rng=rng,
        )
        self.burned: bool = False
        self.lower = lower
        self.upper = upper
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.prior: Prior
        self.network: Net
        self.optimizer: optim.Adam
        self._initial_network_weights: OrderedDict[str, torch.Tensor]
        self._initial_optimizer_state: dict
        self._create_modules()

    def _create_modules(self):
        """Create the network and the optimizer, using the global torch RNG."""
        assert self.lower.ndim == 1
        self.network = Net(
            n_inputs=self.lower.shape[0],
            n_units=[self.n_units_1, self.n_units_2, self.n_units_3],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.init_learning_rate
        )
        # Copies of the network and optimizer parameters, so we just reload the weights rather than
        # recreate a new network at each iteration.
        self._initial_network_weights = self.network.state_dict()
        self._initial_optimizer_state = self.optimizer.state_dict()

    def set_state(self, state_dict: dict) -> None:
        """Restore the state of the optimizer"""
        torch.random.set_rng_state(state_dict["torch"])
        if torch.cuda.is_available() and state_dict["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(state_dict["torch_cuda"])
        self.rng.set_state(state_dict["rng"])
        self.prior.rng.set_state(state_dict["prior_rng"])

    def state_dict(self) -> dict:
        """Return the current state algorithm so that it can be restored."""
        # NOTE: In the case of DNGO, we just need to save the RNG state, since the networks are
        # not reused between iterations and are created using the RNG state.
        return {
            "torch": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "rng": self.rng.get_state(),
            "prior_rng": self.prior.rng.get_state(),
        }

    def seed(self, seed: int) -> None:
        """Seed all internal RNGs"""
        self.rng = np.random.RandomState(seed)
        rand_nums = self.rng.randint(1, int(10e8), 2)
        pytorch_seed = rand_nums[0]

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # type: ignore
            torch.cuda.manual_seed_all(pytorch_seed)
            torch.backends.cudnn.deterministic = True  # type: ignore

        torch.manual_seed(pytorch_seed)

        self.prior.rng.seed(rand_nums[1])
        # Recreate the modules using that RNG seed.
        self._create_modules()

    def _to_tensor(self, v: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(v, device=self.device, dtype=torch.float32)

    @staticmethod
    def _to_numpy(v: torch.Tensor) -> np.ndarray:
        return v.detach().cpu().numpy()

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform one training epoch, and return the average loss per batch."""
        train_err: torch.Tensor | float = 0.0
        batch_index = 0

        # Check if we have enough points to create a minibatch otherwise use all data points
        batch_size = min(self.batch_size, self.X.shape[0])
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        epoch_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_index, (inputs, targets) in enumerate(epoch_dataloader):
            self.optimizer.zero_grad()
            output = self.network(inputs)
            loss = F.mse_loss(output, targets)
            loss.backward()
            self.optimizer.step()

            train_err += loss.detach()

        average_error = train_err / (batch_index + 1)

        logger.debug(  # pylint: disable=logging-too-many-args
            "Training loss:\t\t{:.5g}",
            average_error,
        )
        return float(average_error)

    @BaseModel._check_shapes_train  # type: ignore  pylint: disable=protected-access
    def train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True):
        """
        Trains the model on the provided data.

        NOTE: Overwritten here to fix the following issues:
        - memory leak in the training loop due to accumulating (and storing) live losses.
        - Add cuda training when available
        - reload initial weights instead of re-creating the model and optimizer at each iteration.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters are used.

        """
        start_time = time.time()
        logger.info("Starting training with %s samples", X.shape[0])

        self.network.load_state_dict(self._initial_network_weights)
        self.optimizer.load_state_dict(self._initial_optimizer_state)

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        # Normalize outputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.y = self.y[:, None]

        # Start training
        loss_per_epoch = np.zeros([self.num_epochs])

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            average_batch_loss = self.train_epoch(self.X, self.y)

            loss_per_epoch[epoch] = average_batch_loss

            logger.debug(  # pylint: disable=logging-too-many-args
                "Epoch {} of {}",
                epoch + 1,
                self.num_epochs,
            )
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time

            logger.debug(  # pylint: disable=logging-too-many-args
                "Epoch time {:.3f}s, total time {:.3f}s",
                epoch_time,
                total_time,
            )

        # Design matrix
        self.Theta = self._to_numpy(self.network.basis_funcs(self._to_tensor(self.X)))

        if do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(
                    self.n_hypers, 2, self.marginal_log_likelihood
                )

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    # NOTE: This has changed with the newer emcee versions:
                    self.p0, _, _ = self.sampler.run_mcmc(
                        self.p0, self.burnin_steps, rstate0=self.rng
                    )

                    self.burned = True

                # Start sampling
                pos, _, _ = self.sampler.run_mcmc(
                    self.p0, self.chain_length, rstate0=self.rng
                )

                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos

                # Take the last samples from each walker set them back on a linear scale
                linear_theta = np.exp(self.sampler.chain[:, -1])
                self.hypers = linear_theta
                self.hypers[:, 1] = 1 / self.hypers[:, 1]
            else:
                # Optimize hyperparameters of the Bayesian linear regression
                p0 = self.prior.sample_from_prior(n_samples=1)
                res = optimize.fmin(self.negative_mll, p0)
                self.hypers = [[np.exp(res[0]), 1 / np.exp(res[1])]]
        else:

            self.hypers = [[self.alpha, self.beta]]

        logger.info("Hypers: %s", self.hypers)
        self.models: list = []
        for sample in self.hypers:
            # Instantiate a model for each hyperparameter configuration
            model = BayesianLinearRegression(
                alpha=sample[0], beta=sample[1], basis_func=None
            )
            model.train(self.Theta, self.y[:, 0], do_optimize=False)

            self.models.append(model)

    @BaseModel._check_shapes_predict  # type: ignore  pylint: disable=protected-access
    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Get features from the net
        theta = self._to_numpy(self.network.basis_funcs(self._to_tensor(X_)))

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu**2 + var, axis=0) - m**2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std**2

        return m, v


# pylint: disable=unsubscriptable-object, too-few-public-methods
class RoBO_DNGO(RoBO[OrionDNGOWrapper]):
    """
    Wrapper for RoBO with DNGO

    For more information on the algorithm,
    see original paper at http://proceedings.mlr.press/v37/snoek15.html.

    J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
    N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
    Scalable Bayesian Optimization Using Deep Neural Networks
    Proc. of ICML'15

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
    normalize_input: bool
        Normalize the input based on the provided bounds (zero mean and unit standard deviation).
        Defaults to ``True``.
    normalize_output: bool
        Normalize the output based on data (zero mean and unit standard deviation).
        Defaults to ``False``.
    chain_length : int
        The chain length of the MCMC sampler
    burnin_steps: int
        The number of burnin steps before the sampling procedure starts
    batch_size: int
        Batch size for training the neural network
    num_epochs: int
        Number of epochs for training
    learning_rate: float
        Initial learning rate for Adam
    adapt_epoch: int
        Defines after how many epochs the learning rate will be decayed by a factor 10

    """

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = 0,
        n_initial_points: int = 20,
        maximizer: MaximizerName = "random",
        acquisition_func: AcquisitionFnName = "log_ei",
        normalize_input: bool = True,
        normalize_output: bool = False,
        chain_length: int = 2000,
        burnin_steps: int = 2000,
        batch_size: int = 10,
        num_epochs: int = 500,
        learning_rate: float = 1e-2,
        adapt_epoch: int = 5000,
    ):

        super().__init__(
            space=space,
            seed=seed,
            n_initial_points=n_initial_points,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
        )
        self.seed = seed
        self.n_initial_points = n_initial_points
        self.maximizer = maximizer
        self.acquisition_func = acquisition_func
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.chain_length = chain_length
        self.burnin_steps = burnin_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.adapt_epoch = adapt_epoch

    def build_model(self) -> OrionDNGOWrapper:
        """Build the model."""
        lower, upper = build_bounds(self.space)
        n_hypers = infer_n_hypers(build_kernel(lower, upper))
        return OrionDNGOWrapper(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            adapt_epoch=self.adapt_epoch,
            n_units_1=50,
            n_units_2=50,
            n_units_3=50,
            alpha=1.0,
            beta=1000,
            prior=None,
            do_mcmc=True,
            n_hypers=n_hypers,
            chain_length=self.chain_length,
            burnin_steps=self.burnin_steps,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            rng=None,
            lower=lower,
            upper=upper,
        )
