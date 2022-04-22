#!/usr/bin/env python
"""Perform integration tests for `orion.algo.robo`."""
from __future__ import annotations

import copy
import itertools
from typing import ClassVar, TypeVar

import pytest
from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.trial import Trial
from orion.testing.algo import BaseAlgoTests, TestPhase, first_phase_only

from orion.algo.robo.base import RoBO
from orion.algo.robo.bohamiann import RoBO_BOHAMIANN
from orion.algo.robo.dngo import RoBO_DNGO
from orion.algo.robo.gp import RoBO_GP, RoBO_GP_MCMC
from orion.algo.robo.randomforest import RoBO_RandomForest

N_INIT = 10


def modified_config(config, **kwargs):
    modified = copy.deepcopy(config)
    modified.update(kwargs)
    return modified


RoboAlgoType = TypeVar("RoboAlgoType", bound=RoBO)


class BaseRoBOTests(BaseAlgoTests[RoboAlgoType]):
    # To be overwritten by subclasses:
    algo_type: ClassVar[type[RoBO]] = RoBO

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.algo_name = cls.algo_type.__name__.lower()

    def test_suggest_init(self, phase: TestPhase):
        algo = self.create_algo()
        trials = algo.suggest(num=phase.length)
        assert len(trials) == phase.length

    @first_phase_only
    def test_suggest_init_missing(self, phase: TestPhase):
        """Test that when in the first phase, the algorithm"""
        algo = self.create_algo()
        missing = 3

        self.force_observe(algo=algo, num=phase.length - missing)
        # NOTE: Ask for a large number of additional trials, which shouldn't be given back by the
        # algo.
        trials = algo.suggest(self.max_trials)
        assert len(trials) == missing

    @first_phase_only
    def test_suggest_init_overflow(self, mocker, first_phase: TestPhase):
        algo = self.create_algo()

        self.force_observe(algo=algo, num=first_phase.length - 1)
        spy = mocker.spy(algo.algorithm.space, "sample")
        # Now reaching end of the first phase, by asking more trials than the length of the
        # first phase.
        trials = algo.suggest(first_phase.length * 3)
        assert trials is not None
        assert len(trials) == 1

        # Verify trial was sampled randomly, not using BO
        assert spy.call_count == 1

        # Next call to suggest should still be in first phase, since we still haven't observed that
        # missing random trial.
        trials = algo.suggest(first_phase.length * 3)
        assert trials is not None
        assert len(trials) == 1
        # Verify trial was sampled randomly, not using BO
        assert spy.call_count == 2

    def test_is_done_cardinality(self):
        # TODO: Support correctly loguniform(discrete=True)
        #       See https://github.com/Epistimio/orion/issues/566
        space = {
            "x": "uniform(0, 4, discrete=True)",
            "y": "choices(['a', 'b', 'c'])",
            "z": "uniform(1, 6, discrete=True)",
        }

        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        i = 0
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = len(algo.algorithm.registry)
            trial = tuple_to_trial((x, y, z), space=algo.space)
            trial.results = [Trial.Result(type="objective", value=i)]
            algo.observe([trial])
            assert len(algo.algorithm.registry) == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


class TestRoBO_GP(BaseRoBOTests[RoBO_GP]):
    algo_type: ClassVar[type[RoBO]] = RoBO_GP
    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "n_initial_points": N_INIT,
        "normalize_input": False,
        "normalize_output": True,
        "seed": 1234,
    }

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("gp", N_INIT, "robo.choose_next"),
    ]


class TestRoBO_GP_MCMC(BaseRoBOTests[RoBO_GP_MCMC]):
    algo_type: ClassVar[type[RoBO]] = RoBO_GP_MCMC
    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "normalize_input": True,
        "normalize_output": False,
        "chain_length": 10,
        "burnin_steps": 2,
        "n_initial_points": N_INIT,
        "seed": 1234,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("gp_mcmc", N_INIT, "robo.choose_next"),
    ]


@pytest.mark.skip(reason="pyrfr seems to have changed.")
class TestRoBO_RandomForest(BaseRoBOTests[RoBO_RandomForest]):
    algo_type: ClassVar[type[RoBO]] = RoBO_RandomForest
    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "num_trees": 10,
        "do_bootstrapping": False,
        "n_points_per_tree": 5,
        "compute_oob_error": True,
        "return_total_variance": False,
        "n_initial_points": N_INIT,
        "seed": 1234,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("randomforest", N_INIT, "robo.choose_next"),
    ]


class TestRoBO_DNGO(BaseRoBOTests[RoBO_DNGO]):
    algo_type: ClassVar[type[RoBO]] = RoBO_DNGO

    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "normalize_input": True,
        "normalize_output": False,
        "chain_length": 10,
        "burnin_steps": 2,
        "learning_rate": 1e-2,
        "batch_size": 10,
        "num_epochs": 10,
        "adapt_epoch": 20,
        "n_initial_points": N_INIT // 2,
        "seed": 1234,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("dngo", N_INIT // 2, "robo.choose_next"),
    ]

    def test_configuration_to_model(self):
        """Test that the values passed in the configuration make their way to the model."""
        train_config = dict(
            **{
                key: self.config[key] * 2
                for key in [
                    "chain_length",
                    "burnin_steps",
                    "num_epochs",
                    "adapt_epoch",
                    "learning_rate",
                ]
            },
            batch_size=self.config["batch_size"] + 1,
        )

        tmp_config = modified_config(
            self.config, n_initial_points=N_INIT + 1, **train_config
        )

        algo = self.create_algo(config=tmp_config)
        algo.algorithm._initialize()

        model = algo.algorithm.model

        assert model.chain_length == tmp_config["chain_length"]
        assert model.burnin_steps == tmp_config["burnin_steps"]
        assert model.num_epochs == tmp_config["num_epochs"]
        assert model.adapt_epoch == tmp_config["adapt_epoch"]
        assert model.init_learning_rate == tmp_config["learning_rate"]
        assert model.batch_size == tmp_config["batch_size"]


class TestRoBO_BOHAMIANN(BaseRoBOTests[RoBO_BOHAMIANN]):
    algo_type: ClassVar[type[RoBO]] = RoBO_BOHAMIANN
    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "normalize_input": True,
        "normalize_output": False,
        "burnin_steps": 2,
        "sampling_method": "adaptive_sghmc",
        "use_double_precision": True,
        "num_steps": 100,
        "keep_every": 10,
        "learning_rate": 1e-2,
        "batch_size": 10,
        "epsilon": 1e-10,
        "mdecay": 0.05,
        "verbose": False,
        "n_initial_points": N_INIT,
        "seed": 1234,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("bohamiann", N_INIT, "robo.choose_next"),
    ]

    def test_configuration_to_model(self, mocker):

        train_config = dict(
            burnin_steps=self.config["burnin_steps"] * 2,
            num_steps=500,
            keep_every=self.config["keep_every"] * 2,
            learning_rate=self.config["learning_rate"] * 2,
            batch_size=self.config["batch_size"] + 1,
            epsilon=self.config["epsilon"] * 2,
            mdecay=self.config["mdecay"] + 0.01,
            verbose=True,
        )

        tmp_config = modified_config(
            self.config,
            sampling_method="sgld",
            use_double_precision=False,
            n_initial_points=N_INIT + 1,
            **train_config,
        )

        # Adapt to bnn.train interface
        train_config["num_burn_in_steps"] = train_config.pop("burnin_steps")
        train_config["lr"] = train_config.pop("learning_rate")

        # Add arguments that are not configurable
        train_config["do_optimize"] = True
        train_config["continue_training"] = False

        algo = self.create_algo(tmp_config)
        algo.algorithm._initialize()

        bnn = algo.algorithm.model.bnn
        assert bnn.sampling_method == tmp_config["sampling_method"]
        assert bnn.use_double_precision == tmp_config["use_double_precision"]

        spy = mocker.spy(bnn, "train")
        self.force_observe(algo=algo, num=tmp_config["n_initial_points"] + 1)
        algo.suggest(1)
        assert spy.call_count > 0
        assert spy.call_args[1] == train_config
