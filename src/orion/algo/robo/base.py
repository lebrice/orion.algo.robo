"""
Base class for RoBO algorithms.
"""
from __future__ import annotations
from abc import ABC
from typing import Any, Callable, Iterable, Literal, Optional
import typing
import george
import numpy
from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.pi import PI
from robo.initial_design import init_latin_hypercube_sampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
import numpy as np
from george.kernels import Kernel
from robo.models.base_model import BaseModel
from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction
from abc import abstractmethod
from orion.core.worker.trial import Trial
from orion.core.utils.format_trials import tuple_to_trial, trial_to_tuple


AcquisitionFnName = Literal["ei", "log_ei", "pi", "lcb"]
MaximizerName = Literal["random", "scipy", "differential_evolution"]


def build_bounds(space: Space) -> tuple[np.ndarray, np.ndarray]:
    """
    Build bounds of optimization space

    Parameters
    ----------
    space: ``orion.algo.space.Space``
        Search space for the optimization.

    """
    lower = []
    upper = []
    for dim in space.values():
        low, high = dim.interval()

        shape = dim.shape
        assert not shape or shape == [1]

        lower.append(low)
        upper.append(high)

    return np.array(lower), np.array(upper)


def build_kernel(lower: np.ndarray, upper: np.ndarray) -> Kernel:
    """
    Build kernels for GPs.

    Parameters
    ----------
    lower: numpy.ndarray (D,)
        The lower bound of the search space
    upper: numpy.ndarray (D,)
        The upper bound of the search space
    """

    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert numpy.all(lower < upper), "Lower bound >= upper bound"

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = numpy.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
    kernel = cov_amp * exp_kernel

    return kernel


def infer_n_hypers(kernel: Kernel) -> int:
    """Infer number of MCMC chains that should be used based on size of kernel"""
    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    return n_hypers


def build_prior(kernel: Kernel) -> DefaultPrior:
    """Build default GP prior based on kernel"""
    return DefaultPrior(len(kernel) + 1, numpy.random.RandomState(None))


def build_acquisition_func(acquisition_func: AcquisitionFnName, model: BaseModel):
    """
    Build acquisition function

    Parameters
    ----------
    acquisition_func: str
        Name of the acquisition function. Can be one of ``['ei', 'log_ei', 'pi', 'lcb']``.
    model: ``robo.models.base_model.BaseModel``
        Model used for the Bayesian optimization.

    """
    if acquisition_func == "ei":
        acquisition_function = EI(model)
    elif acquisition_func == "log_ei":
        acquisition_function = LogEI(model)
    elif acquisition_func == "pi":
        acquisition_function = PI(model)
    elif acquisition_func == "lcb":
        acquisition_function = LCB(model)
    else:
        raise ValueError(
            "'{}' is not a valid acquisition function".format(acquisition_func)
        )

    return acquisition_function


def build_optimizer(
    model, maximizer: MaximizerName, acquisition_func: BaseAcquisitionFunction,
) -> BayesianOptimization:
    """
    General interface for Bayesian optimization for global black box
    optimization problems.

    Parameters
    ----------
    maximizer: str
        The optimizer for the acquisition function.
        Can be one of ``{"random", "scipy", "differential_evolution"}``
    acquisition_func:
        The instantiated acquisition function

    Returns
    -------
        Optimizer

    """
    if maximizer == "random":
        max_func = RandomSampling(acquisition_func, model.lower, model.upper, rng=None)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(acquisition_func, model.lower, model.upper, rng=None)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(
            acquisition_func, model.lower, model.upper, rng=None
        )
    else:
        raise ValueError(
            "'{}' is not a valid function to maximize the "
            "acquisition function".format(maximizer)
        )

    # NOTE: Internal RNG of BO won't be used.
    # NOTE: Nb of initial points won't be used within BO, but rather outside
    bo = BayesianOptimization(
        lambda: None,
        model.lower,
        model.upper,
        acquisition_func,
        model,
        max_func,
        initial_points=None,
        rng=None,
        initial_design=init_latin_hypercube_sampling,
        output_path=None,
    )

    return bo


class RoBO(BaseAlgorithm, ABC):
    """
    Base class to wrap RoBO algorithms.


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

    requires_type = "real"
    requires_dist = "linear"
    requires_shape = "flattened"

    def __init__(
        self,
        space,
        seed: int = 0,
        n_initial_points: int = 20,
        maximizer: MaximizerName = "random",
        acquisition_func: AcquisitionFnName = "log_ei",
        **kwargs,
    ):

        self.model: Optional[BaseModel] = None
        self.robo: Optional[BayesianOptimization] = None
        self._bo_duplicates: list[tuple[str, tuple[Trial, Trial.Result]]] = []

        super().__init__(
            space,
            n_initial_points=n_initial_points,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            seed=seed,
        )
        self.n_initial_points = n_initial_points
        self.maximizer: MaximizerName = maximizer
        self.acquisition_func: AcquisitionFnName = acquisition_func
        self.seed = seed

        # Otherwise it is turned no 'random' because of BaseAlgorithms constructor... -_-
        self.maximizer = maximizer

        self._param_names += list(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def space(self) -> Space:
        """Space of the optimizer"""
        return self._space

    @space.setter
    def space(self, space: Space) -> None:
        """Setter of optimizer's space.

        Side-effect: Will initialize optimizer.
        """
        self._original = self._space
        self._space = space
        self._initialize()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Build model and register it as ``self.model``"""
        raise NotImplementedError()

    def build_acquisition_func(self) -> BaseAcquisitionFunction:
        """Build and return the acquisition function."""
        assert self.model is not None
        return build_acquisition_func(self.acquisition_func, self.model)

    def _initialize(self) -> None:
        """Initialize the optimizer once the space is transformed"""
        self._initialize_model()
        self.robo = build_optimizer(
            self.model,
            maximizer=self.maximizer,
            acquisition_func=self.build_acquisition_func(),
        )

        self.seed_rng(self.seed)

    @property
    def XY(self) -> np.ndarray:
        """ Matrix containing trial points and their results. """
        trials_and_results: list[tuple[Trial, dict]] = list(
            self._trials_info.values()
        ) + self._bo_duplicates
        # Keep only the trials that have a result.
        trials_with_a_result = [
            (trial, result)
            for trial, result in trials_and_results
            if result is not None
        ]
        x: list[tuple] = []
        y: list[float] = []
        for trial, result in trials_with_a_result:
            x.append(trial_to_tuple(trial, space=self.space))
            y.append(result["objective"])
        return np.array(x), np.array(y)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        """
        self.rng = numpy.random.RandomState(seed)

        rand_nums = self.rng.randint(1, 10e8, 4)

        if self.robo:
            self.robo.rng = numpy.random.RandomState(rand_nums[0])
            self.robo.maximize_func.rng.seed(rand_nums[1])

        if self.model:
            self.model.seed(rand_nums[2])

        numpy.random.seed(rand_nums[3])

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        s_dict: dict[str, Any] = super().state_dict

        s_dict.update(
            {
                "rng_state": self.rng.get_state(),
                "global_numpy_rng_state": numpy.random.get_state(),
                "maximizer_rng_state": self.robo.maximize_func.rng.get_state(),
                "bo_duplicates": self._bo_duplicates,
            }
        )
        if self.model is not None:
            s_dict["model"] = self.model.state_dict()

        return s_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        """
        super().set_state(state_dict)

        self.rng.set_state(state_dict["rng_state"])
        numpy.random.set_state(state_dict["global_numpy_rng_state"])
        self.robo.maximize_func.rng.set_state(state_dict["maximizer_rng_state"])
        self.model.set_state(state_dict["model"])
        self._bo_duplicates = state_dict["bo_duplicates"]

    def suggest(self, num: int) -> list[Trial] | None:
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        num = min(num, max(self.n_initial_points - self.n_suggested, 1))

        samples = []
        candidates = []
        while len(samples) < num:
            if candidates:
                candidate = candidates.pop(0)
                if candidate:
                    self.register(candidate)
                    samples.append(candidate)
            elif self.n_observed < self.n_initial_points:
                candidates = self._suggest_random(num)
            else:
                candidates = self._suggest_bo(max(num - len(samples), 0))

            if not candidates:
                break
        assert all(isinstance(sample, Trial) for sample in samples), samples
        return samples
        return [tuple_to_trial(sample, self.space) for sample in samples]

    def _suggest(
        self, num: int, function: Callable[[int], Iterable[Trial]]
    ) -> list[Trial]:
        trials: list[Trial] = []
        attempts = 0
        max_attempts = 100
        while len(trials) < num and attempts < max_attempts:
            for candidate in function(num - len(trials)):
                if not self.has_suggested(candidate):
                    self.register(candidate)
                    trials.append(candidate)

                if self.is_done:
                    return trials

            attempts += 1

            # print(attempts)

        return trials

    def _suggest_random(self, num: int) -> list[Trial]:
        def sample(num: int) -> list[Trial]:
            return self.space.sample(
                num, seed=tuple(self.rng.randint(0, 1000000, size=3))
            )

        return self._suggest(num, sample)

    def _suggest_bo(self, num: int) -> list[Trial]:
        # pylint: disable = unused-argument
        def suggest_bo(num: int) -> list[Trial]:
            # pylint: disable = protected-access
            assert self.robo is not None
            X, y = self.XY
            point = list(self.robo.choose_next(X, y))
            trial = tuple_to_trial(point, self.space)
            # If already suggested, give corresponding result to BO to sample another point
            if self.has_suggested(trial):
                result = self._trials_info[self.get_id(trial)][1]
                if result is None:
                    results = []
                    for _, other_result in self._trials_info.values():
                        if other_result is not None:
                            results.append(other_result["objective"])
                    result = Trial.Result(objective=numpy.array(results).mean())

                self._bo_duplicates.append((trial, result))

                # self.optimizer.tell([point], [result])
                return []

            return [trial]

        return self._suggest(num, suggest_bo)

    @property
    def is_done(self) -> bool:
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if self.n_suggested >= self._original.cardinality:
            return True

        if self.n_suggested >= getattr(self, "max_trials", float("inf")):
            return True

        return False
