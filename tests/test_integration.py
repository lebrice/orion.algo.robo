#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.robo`."""
from __future__ import annotations

from abc import ABC
import copy
import itertools
from typing import ClassVar, Type

from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.trial import Trial
from orion.testing.algo import BaseAlgoTests, TestPhase, first_phase_only
from orion.algo.robo.base import RoBO
from orion.algo.robo.gp import RoBO_GP, RoBO_GP_MCMC
from orion.algo.robo.bohamiann import RoBO_BOHAMIANN
from orion.algo.robo.randomforest import RoBO_RandomForest
from orion.algo.robo.dngo import RoBO_DNGO


N_INIT = 10


def modified_config(config, **kwargs):
    modified = copy.deepcopy(config)
    modified.update(kwargs)
    return modified


class BaseRoBOTests(BaseAlgoTests, ABC):
    # To be overwritten by subclasses:
    algo_type: ClassVar[type[RoBO]] = RoBO

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.algo_name = cls.algo_type.__name__.lower()

    @classmethod
    def duration_of(cls, phase: TestPhase) -> int:
        phase_index = cls.phases.index(phase)
        end_n_trials = (
            cls.phases[phase_index + 1].n_trials
            if phase_index + 1 < len(cls.phases)
            else cls.max_trials
        )
        start_n_trials = phase.n_trials
        return end_n_trials - start_n_trials

    def test_suggest_init(self, phase: TestPhase):
        algo = self.create_algo()
        trials_in_phase = self.duration_of(phase)
        trials = algo.suggest(trials_in_phase)
        assert len(trials) == trials_in_phase

    def test_suggest_init_missing(self, phase: TestPhase):
        algo = self.create_algo()

        trials_in_phase = self.duration_of(phase)
        missing = 3

        self.force_observe(algo=algo, num=trials_in_phase - missing)
        # NOTE: Why ask for 1000? What's the point? Couldn't we just ask for slightly more than
        # max_trials?
        trials = algo.suggest(1000)

        assert len(trials) == missing

    @first_phase_only
    def test_suggest_init_overflow(self, phase: TestPhase, mocker):
        algo = self.create_algo()
        n_init = self.duration_of(phase)
        self.force_observe(algo=algo, num=n_init - 1)
        # Now reaching N_INIT
        spy = mocker.spy(algo.algorithm.space, "sample")
        trials = algo.suggest(1000)
        assert trials is not None
        assert len(trials) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 1
        # Overflow above N_INIT
        trials = algo.suggest(1000)
        assert trials is not None
        assert len(trials) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 2

    def test_suggest_n(self, phase: TestPhase):
        algo = self.create_algo()
        trials = algo.suggest(5)
        assert trials is not None
        if phase.n_trials == 0:
            assert len(trials) == 5
        else:
            assert len(trials) == 1  # TODO: HUH? Why is this supposed to be the case?

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


class TestRoBO_GP(BaseRoBOTests):
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


class TestRoBO_GP_MCMC(BaseRoBOTests):
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


class TestRoBO_RandomForest(BaseRoBOTests):
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


class TestRoBO_DNGO(TestRoBO_GP):
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
        "n_initial_points": N_INIT,
        "seed": 1234,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("dngo", N_INIT, "robo.choose_next"),
    ]

    def test_configuration_to_model(self, mocker):

        train_config = dict(
            chain_length=self.config["chain_length"] * 2,
            burnin_steps=self.config["burnin_steps"] * 2,
            num_epochs=self.config["num_epochs"] * 2,
            adapt_epoch=self.config["adapt_epoch"] * 2,
            learning_rate=self.config["learning_rate"] * 2,
            batch_size=self.config["batch_size"] + 1,
        )

        tmp_config = modified_config(
            self.config, n_initial_points=N_INIT + 1, **train_config
        )

        algo = self.create_algo(tmp_config)

        model = algo.algorithm.model

        assert model.chain_length == tmp_config["chain_length"]
        assert model.burnin_steps == tmp_config["burnin_steps"]
        assert model.num_epochs == tmp_config["num_epochs"]
        assert model.adapt_epoch == tmp_config["adapt_epoch"]
        assert model.init_learning_rate == tmp_config["learning_rate"]
        assert model.batch_size == tmp_config["batch_size"]


class TestRoBO_BOHAMIANN(BaseRoBOTests):
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

    def test_configuration_to_model(self):

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

        assert algo.algorithm.model.bnn.sampling_method == tmp_config["sampling_method"]
        assert (
            algo.algorithm.model.bnn.use_double_precision
            == tmp_config["use_double_precision"]
        )

        spy = self.spy_phase(
            mocker, tmp_config["n_initial_points"] + 1, algo, "model.bnn.train"
        )
        algo.suggest(1)
        assert spy.call_count > 0
        assert spy.call_args[1] == train_config
