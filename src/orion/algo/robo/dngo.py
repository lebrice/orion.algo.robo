"""
Wrapper for RoBO with DNGO
"""
from __future__ import annotations

from typing import Sequence
import numpy
import torch
from pybnn.dngo import DNGO
from typing_extensions import Literal

from orion.algo.robo.base import (
    RoBO,
    build_bounds,
    build_kernel,
    infer_n_hypers,
    MaximizerName,
    AcquisitionFnName,
    WrappedRoboModel,
)
from orion.algo.space import Space


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

    def __init__(self, lower, upper, **kwargs):

        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        torch.random.set_rng_state(state_dict["torch"])
        self.rng.set_state(state_dict["rng"])
        self.prior.rng.set_state(state_dict["prior_rng"])

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        return {
            "torch": torch.random.get_rng_state(),
            "rng": self.rng.get_state(),
            "prior_rng": self.prior.rng.get_state(),
        }

    def seed(self, seed):
        """Seed all internal RNGs"""
        self.rng = numpy.random.RandomState(seed)
        rand_nums = self.rng.randint(1, int(10e8), 2)
        pytorch_seed = rand_nums[0]

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(pytorch_seed)
            torch.backends.cudnn.deterministic = True

        torch.manual_seed(pytorch_seed)

        self.prior.rng.seed(rand_nums[1])


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

