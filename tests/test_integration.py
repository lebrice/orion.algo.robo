#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.robo`."""
import copy
import functools
import itertools
import os

import numpy
import orion.core.cli
import pytest
from orion.testing.algo import BaseAlgoTests

N_INIT = 10


def modified_config(config, **kwargs):
    modified = copy.deepcopy(config)
    modified.update(kwargs)
    return modified


class BaseRoBOTests(BaseAlgoTests):
    def test_suggest_init(self, mocker):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, 0, algo, "space.sample")
        points = algo.suggest(1000)
        assert len(points) == N_INIT

    def test_suggest_init_missing(self, mocker):
        algo = self.create_algo()
        missing = 3
        spy = self.spy_phase(mocker, N_INIT - missing, algo, "space.sample")
        points = algo.suggest(1000)
        assert len(points) == missing

    def test_suggest_init_overflow(self, mocker):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, N_INIT - 1, algo, "space.sample")
        # Now reaching N_INIT
        points = algo.suggest(1000)
        assert len(points) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 1
        # Overflow above N_INIT
        points = algo.suggest(1000)
        assert len(points) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 2

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(5)
        if num == 0:
            assert len(points) == 5
        else:
            assert len(points) == 1

    def test_is_done_cardinality(self):
        # TODO: Support correctly loguniform(discrete=True)
        #       See https://github.com/Epistimio/orion/issues/566
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "uniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = len(algo.algorithm._trials_info)
            algo.observe([[x, y, z]], [dict(objective=i)])
            assert len(algo.algorithm._trials_info) == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


class TestRoBO_GP(BaseRoBOTests):

    algo_name = "robo_gp"
    config = {
        "maximizer": "random",
        "acquisition_func": "log_ei",
        "n_initial_points": N_INIT,
        "normalize_input": False,
        "normalize_output": True,
        "seed": 1234,
    }


class TestRoBO_GP_MCMC(BaseRoBOTests):
    algo_name = "robo_gp_mcmc"
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


class TestRoBO_RandomForest(BaseRoBOTests):
    algo_name = "robo_randomforest"
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


class TestRoBO_DNGO(TestRoBO_GP):
    algo_name = "robo_dngo"

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
    algo_name = "robo_bohamiann"
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
        "continue_training": False,
        "verbose": False,
        "n_initial_points": N_INIT,
        "seed": 1234,
    }

    def test_configuration_to_model(self, mocker):

        train_config = dict(
            burnin_steps=self.config["burnin_steps"] * 2,
            num_steps=500,
            keep_every=self.config["keep_every"] * 2,
            learning_rate=self.config["learning_rate"] * 2,
            batch_size=self.config["batch_size"] + 1,
            epsilon=self.config["epsilon"] * 2,
            mdecay=self.config["mdecay"] + 0.01,
            continue_training=False,
            verbose=True,
        )

        tmp_config = modified_config(
            self.config,
            sampling_method="sgld",
            use_double_precision=False,
            n_initial_points=N_INIT + 1,
            **train_config
        )

        # Adapt to bnn.train interface
        train_config["num_burn_in_steps"] = train_config.pop("burnin_steps")
        train_config["lr"] = train_config.pop("learning_rate")

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


TestRoBO_GP.set_phases(
    [("random", 0, "space.sample"), ("gp", N_INIT + 1, "robo.choose_next")]
)

TestRoBO_GP_MCMC.set_phases([("gp_mcmc", N_INIT + 1, "robo.choose_next")])

TestRoBO_RandomForest.set_phases([("randomforest", N_INIT + 1, "robo.choose_next")])

TestRoBO_DNGO.set_phases([("dngo", N_INIT + 1, "robo.choose_next")])

TestRoBO_BOHAMIANN.set_phases([("bohamiann", N_INIT + 1, "robo.choose_next")])
