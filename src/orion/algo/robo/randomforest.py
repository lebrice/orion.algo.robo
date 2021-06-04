"""
Wrapper for RoBO with Random Forest
"""
import pyrfr.regression as reg
from robo.models.random_forest import RandomForest

from orion.algo.robo.base import RoBO, build_bounds


class RoBO_RandomForest(RoBO):
    """
    Wrapper for RoBO with

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
    num_trees: int
        The number of trees in the random forest. Defaults to 30.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest. Defaults to ``True``.
    n_points_per_tree: int
        Number of data point per tree. If set to 0 then we will use all data points in each tree.
        Defaults to 0.
    compute_oob_error: bool
        Turns on / off calculation of out-of-bag error. Defaults to ``False``.
    return_total_variance: bool
        Return law of total variance (mean of variances + variance of means, if True)
        or explained variance (variance of means, if False). Defaults to ``True``.

    """

    def __init__(
        self,
        space,
        seed=0,
        n_initial_points=20,
        maximizer="random",
        acquisition_func="log_ei",
        num_trees=30,
        do_bootstrapping=True,
        n_points_per_tree=0,
        compute_oob_error=False,
        return_total_variance=True,
    ):

        super(RoBO_RandomForest, self).__init__(
            space,
            maximizer=maximizer,
            acquisition_func=acquisition_func,
            num_trees=num_trees,
            do_bootstrapping=do_bootstrapping,
            n_points_per_tree=n_points_per_tree,
            compute_oob_error=compute_oob_error,
            return_total_variance=return_total_variance,
            n_initial_points=n_initial_points,
            seed=seed,
        )

    def _initialize_model(self):
        lower, upper = build_bounds(self.space)
        self.model = OrionRandomForestWrapper(
            rng=None,
            num_trees=self.num_trees,
            do_bootstrapping=self.do_bootstrapping,
            n_points_per_tree=self.n_points_per_tree,
            compute_oob_error=self.compute_oob_error,
            return_total_variance=self.return_total_variance,
            lower=lower,
            upper=upper,
        )


class OrionRandomForestWrapper(RandomForest):
    """
    Wrapper for RoBO's RandomForest model

    Parameters
    ----------
    lower : np.array(D,)
        Lower bound of the input space which is used for the input space normalization
    upper : np.array(D,)
        Upper bound of the input space which is used for the input space normalization
    num_trees: int
        The number of trees in the random forest.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest.
    n_points_per_tree: int
        Number of data point per tree. If set to 0 then we will use all data points in each tree
    compute_oob_error: bool
        Turns on / off calculation of out-of-bag error. Default: False
    return_total_variance: bool
        Return law of total variance (mean of variances + variance of means, if True)
        or explained variance (variance of means, if False). Default: True
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(
        self,
        lower,
        upper,
        num_trees=30,
        do_bootstrapping=True,
        n_points_per_tree=0,
        compute_oob_error=False,
        return_total_variance=True,
        rng=None,
    ):

        super(OrionRandomForestWrapper, self).__init__(
            num_trees=num_trees,
            do_bootstrapping=do_bootstrapping,
            n_points_per_tree=n_points_per_tree,
            compute_oob_error=compute_oob_error,
            return_total_variance=return_total_variance,
            rng=rng,
        )

        self.lower = lower
        self.upper = upper

    def train(self, X, y, **kwargs):
        """
        Seeds the RNG of Random Forest before calling parent's train().
        """
        # NOTE: We cannot save `reg_rng` state so instead we control it
        #       with random integers sampled from `rng` and keep track of `rng` state.
        self.reg_rng = reg.default_random_engine(int(self.rng.randint(10e8)))
        super(OrionRandomForestWrapper, self).train(X, y, **kwargs)

    def predict(self, X_test, **kwargs):
        """
        Seeds the RNG of Random Forest before calling parent's predict().
        """
        # NOTE: We cannot save `reg_rng` state so instead we control it
        #       with random integers sampled from `rng` and keep track of `rng` state.
        self.reg_rng = reg.default_random_engine(int(self.rng.randint(10e8)))
        return super(OrionRandomForestWrapper, self).predict(X_test, **kwargs)

    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        self.rng.set_state(state_dict["model_rng_state"])

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        return {
            "model_rng_state": self.rng.get_state(),
        }

    def seed(self, seed):
        """Seed all internal RNGs"""
        self.rng.seed(seed)
