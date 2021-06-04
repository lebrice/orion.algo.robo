import george
import numpy
from orion.algo.base import BaseAlgorithm
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.acquisition_functions.pi import PI
from robo.initial_design import init_latin_hypercube_sampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization


def build_bounds(space):
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

    return list(map(numpy.array, (lower, upper)))


def build_kernel(lower, upper):
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


def infer_n_hypers(kernel):
    """Infer number of MCMC chains that should be used based on size of kernel"""
    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    return n_hypers


def build_prior(kernel):
    """Build default GP prior based on kernel"""
    return DefaultPrior(len(kernel) + 1, numpy.random.RandomState(None))


def build_acquisition_func(acquisition_func, model):
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
        acquisition_func = EI(model)
    elif acquisition_func == "log_ei":
        acquisition_func = LogEI(model)
    elif acquisition_func == "pi":
        acquisition_func = PI(model)
    elif acquisition_func == "lcb":
        acquisition_func = LCB(model)
    else:
        raise ValueError(
            "'{}' is not a valid acquisition function".format(acquisition_func)
        )

    return acquisition_func


def build_optimizer(model, maximizer, acquisition_func):
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
    maximizer_rng = numpy.random.RandomState(maximizer_seed)
    if maximizer == "random":
        max_func = RandomSampling(
            acquisition_func, model.lower, model.upper, rng=maximizer_rng
        )
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(
            acquisition_func, model.lower, model.upper, rng=maximizer_rng
        )
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(
            acquisition_func, model.lower, model.upper, rng=maximizer_rng
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


class RoBO(BaseAlgorithm):
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

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        space,
        seed=0,
        n_init=20,
        maximizer="random",
        acquisition_func="log_ei",
        **kwargs,
    ):

        self.model = None
        self.robo = None
        self._bo_duplicates = []

        super(RoBO, self).__init__(
            space,
            n_init=n_init,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            seed=seed,
        )

        # Otherwise it is turned no 'random' because of BaseAlgorithms constructor... -_-
        self.maximizer = maximizer

        self._param_names += list(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def space(self):
        """Space of the optimizer"""
        return self._space

    @space.setter
    def space(self, space):
        """Setter of optimizer's space.

        Side-effect: Will initialize optimizer.
        """
        self._original = self._space
        self._space = space
        self._initialize()

    def _initialize_model(self):
        """Build model and register it as ``self.model``"""
        raise NotImplementedError()

    def build_acquisition_func(self):
        """Build and return the acquisition function."""
        return build_acquisition_func(self.acquisition_func, self.model)

    def _initialize(self):
        """Initialize the optimizer once the space is transformed"""
        self._initialize_model()
        self.robo = build_optimizer(
            self.model,
            maximizer=self.maximizer,
            acquisition_func=self.build_acquisition_func(),
        )

        self.seed_rng(self.seed)

    # pylint:disable=invalid-name
    @property
    def X(self):
        """Matrix containing trial points"""
        ref_point = self.space.sample(1, seed=0)[0]
        points = list(self._trials_info.values()) + self._bo_duplicates
        points = list(filter(lambda point: point[1] is not None, points))
        X = numpy.zeros((len(points), len(ref_point)))
        for i, (point, _result) in enumerate(points):
            X[i] = point

        return X

    # pylint:disable=invalid-name
    @property
    def y(self):
        """Vector containing trial results"""
        points = list(self._trials_info.values()) + self._bo_duplicates
        points = list(filter(lambda point: point[1] is not None, points))
        y = numpy.zeros(len(points))
        for i, (_point, result) in enumerate(points):
            y[i] = result["objective"]

        return y

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
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        s_dict = super(RoBO, self).state_dict

        s_dict.update(
            {
                "rng_state": self.rng.get_state(),
                "global_numpy_rng_state": numpy.random.get_state(),
                "maximizer_rng_state": self.robo.maximize_func.rng.get_state(),
                "bo_duplicates": self._bo_duplicates,
            }
        )

        s_dict["model"] = self.model.state_dict()

        return s_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        """
        super(RoBO, self).set_state(state_dict)

        self.rng.set_state(state_dict["rng_state"])
        numpy.random.set_state(state_dict["global_numpy_rng_state"])
        self.robo.maximize_func.rng.set_state(state_dict["maximizer_rng_state"])
        self.model.set_state(state_dict["model"])
        self._bo_duplicates = state_dict["bo_duplicates"]

    def suggest(self, num=None):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        num = min(num, max(self.n_init - self.n_suggested, 1))

        samples = []
        candidates = []
        while len(samples) < num:
            if candidates:
                candidate = candidates.pop(0)
                if candidate:
                    self.register(candidate)
                    samples.append(candidate)
            elif self.n_observed < self.n_init:
                candidates = self._suggest_random(num)
            else:
                candidates = self._suggest_bo(max(num - len(samples), 0))

            if not candidates:
                break

        return samples

    def _suggest(self, num, function):
        points = []

        attempts = 0
        max_attempts = 100
        while len(points) < num and attempts < max_attempts:
            for candidate in function(num - len(points)):
                if not self.has_suggested(candidate):
                    self.register(candidate)
                    points.append(candidate)

                if self.is_done:
                    return points

            attempts += 1
            print(attempts)

        return points

    def _suggest_random(self, num):
        def sample(num):
            return self.space.sample(
                num, seed=tuple(self.rng.randint(0, 1000000, size=3))
            )

        return self._suggest(num, sample)

    def _suggest_bo(self, num):
        # pylint: disable = unused-argument
        def suggest_bo(num):
            # pylint: disable = protected-access
            point = list(self.robo.choose_next(self.X, self.y))

            # If already suggested, give corresponding result to BO to sample another point
            if self.has_suggested(point):
                result = self._trials_info[self.get_id(point)][1]
                if result is None:
                    results = []
                    for _, other_result in self._trials_info.values():
                        if other_result is not None:
                            results.append(other_result["objective"])
                    result = {"objective": numpy.array(results).mean()}

                self._bo_duplicates.append((point, result))

                # self.optimizer.tell([point], [result])
                return []

            return [point]

        return self._suggest(num, suggest_bo)

    @property
    def is_done(self):
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
