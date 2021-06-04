import numpy
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC

from orion.algo.robo.base import (
    RoBO,
    build_acquisition_func,
    build_bounds,
    build_kernel,
    build_prior,
    infer_n_hypers,
)


class RoBO_GP(RoBO):
    """
    Wrapper for RoBO with Gaussian processes

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

    """

    def __init__(
        self,
        space,
        seed=0,
        n_initial_points=20,
        maximizer="random",
        acquisition_func="log_ei",
        normalize_input=True,
        normalize_output=False,
    ):

        super(RoBO_GP, self).__init__(
            space,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            n_initial_points=n_initial_points,
            seed=seed,
        )

    def _initialize_model(self):
        lower, upper = build_bounds(self.space)
        kernel = build_kernel(lower, upper)
        prior = build_prior(kernel)
        self.model = OrionGaussianProcessWrapper(
            kernel,
            prior=prior,
            rng=None,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            lower=lower,
            upper=upper,
        )


class RoBO_GP_MCMC(RoBO):
    """
    Wrapper for RoBO with Gaussian processes using Markov chain Monte Carlo
    to marginalize out hyperparameters of the Bayesian Optimization.

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
    chain_length: int
        The length of the MCMC chain. We start ``n_hypers`` walker for chain_length
        steps and we use the last sample in the chain as a hyperparameter sample.
        ``n_hypers`` is automatically infered based on dimensionality of the search space.
        Defaults to 2000.
    burnin_steps: int
        The number of burnin steps before the actual MCMC sampling starts.
        Defaults to 2000.

    """

    def __init__(
        self,
        space,
        seed=0,
        n_initial_points=20,
        maximizer="random",
        acquisition_func="log_ei",
        normalize_input=True,
        normalize_output=False,
        chain_length=2000,
        burnin_steps=2000,
    ):

        super(RoBO_GP_MCMC, self).__init__(
            space,
            seed=seed,
            n_initial_points=n_initial_points,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            chain_length=chain_length,
            burnin_steps=burnin_steps,
        )

    def build_acquisition_func(self):
        """Build a marginalized acquisition function with MCMC."""
        return MarginalizationGPMCMC(super(RoBO_GP_MCMC, self).build_acquisition_func())

    def _initialize_model(self):
        lower, upper = build_bounds(self.space)
        kernel = build_kernel(lower, upper)
        prior = build_prior(kernel)
        n_hypers = infer_n_hypers(kernel)
        self.model = OrionGaussianProcessMCMCWrapper(
            kernel,
            prior=prior,
            n_hypers=n_hypers,
            chain_length=self.chain_length,
            burnin_steps=self.burnin_steps,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            rng=None,
            lower=lower,
            upper=upper,
        )


class OrionGaussianProcessWrapper(GaussianProcess):
    """
    Wrapper for RoBO's Gaussian processes model

    Parameters
    ----------
    kernel : george kernel object
        Specifies the kernel that is used for all Gaussian Process
    prior : prior object
        Defines a prior for the hyperparameters of the GP. Make sure that
        it implements the Prior interface.
    noise : float
        Noise term that is added to the diagonal of the covariance matrix
        for the Cholesky decomposition.
    use_gradients : bool
        Use gradient information to optimize the negative log likelihood
    lower : numpy.array(D,)
        Lower bound of the input space which is used for the input space normalization
    upper : numpy.array(D,)
        Upper bound of the input space which is used for the input space normalization
    normalize_output : bool
        Zero mean unit variance normalization of the output values
    normalize_input : bool
        Normalize all inputs to be in [0, 1]. This is important to define good priors for the
        length scales.
    rng: numpy.random.RandomState
        Random number generator

    """

    # pylint:disable=attribute-defined-outside-init
    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        self.rng.set_state(state_dict["model_rng_state"])
        self.prior.rng.set_state(state_dict["prior_rng_state"])
        self.kernel.set_parameter_vector(state_dict["model_kernel_parameter_vector"])
        self.noise = state_dict["noise"]

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        return {
            "prior_rng_state": self.prior.rng.get_state(),
            "model_rng_state": self.rng.get_state(),
            "model_kernel_parameter_vector": self.kernel.get_parameter_vector().tolist(),
            "noise": self.noise,
        }

    def seed(self, seed):
        """Seed all internal RNGs"""
        seeds = numpy.random.RandomState(seed).randint(1, 10e8, size=2)
        self.rng.seed(seeds[0])
        self.prior.rng.seed(seeds[1])


class OrionGaussianProcessMCMCWrapper(GaussianProcessMCMC):
    """
    Wrapper for RoBO's Gaussian processes with MCMC model

    Parameters
    ----------
    kernel : george kernel object
        Specifies the kernel that is used for all Gaussian Process
    prior : prior object
        Defines a prior for the hyperparameters of the GP. Make sure that
        it implements the Prior interface. During MCMC sampling the
        lnlikelihood is multiplied with the prior.
    n_hypers : int
        The number of hyperparameter samples. This also determines the
        number of walker for MCMC sampling as each walker will
        return one hyperparameter sample.
    chain_length : int
        The length of the MCMC chain. We start n_hypers walker for
        chain_length steps and we use the last sample
        in the chain as a hyperparameter sample.
    lower : np.array(D,)
        Lower bound of the input space which is used for the input space normalization
    upper : np.array(D,)
        Upper bound of the input space which is used for the input space normalization
    burnin_steps : int
        The number of burnin steps before the actual MCMC sampling starts.
    rng: np.random.RandomState
        Random number generator

    """

    # pylint:disable=attribute-defined-outside-init
    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        self.rng.set_state(state_dict["model_rng_state"])
        self.prior.rng.set_state(state_dict["prior_rng_state"])

        if state_dict.get("model_p0", None) is not None:
            # pylint:disable=invalid-name
            self.p0 = numpy.array(state_dict["model_p0"])
            self.burned = True
        elif hasattr(self, "p0"):
            delattr(self, "p0")
            self.burned = False

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        s_dict = {
            "prior_rng_state": self.prior.rng.get_state(),
            "model_rng_state": self.rng.get_state(),
        }

        if hasattr(self, "p0"):
            s_dict["model_p0"] = self.p0.tolist()

        return s_dict

    def seed(self, seed):
        """Seed all internal RNGs"""
        seeds = numpy.random.RandomState(seed).randint(1, 10e8, size=2)
        self.rng.seed(seeds[0])
        self.prior.rng.seed(seeds[1])
