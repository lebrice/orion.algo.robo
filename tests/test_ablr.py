from __future__ import annotations

from typing import ClassVar

from test_integration import N_INIT, BaseRoBOTests

from orion.algo.robo.ablr import RoBO_ABLR


class TestRoBO_ABLR(BaseRoBOTests):
    """Tests for the ABLR algorithm."""

    algo_type: ClassVar[type[RoBO_ABLR]] = RoBO_ABLR
    config = {
        "seed": 1234,
        "n_initial_points": N_INIT,
        "maximizer": "random",
        "acquisition_func": "ei",
        "hparams": {
            "alpha": 1.0,
            "beta": 1.0,
            "learning_rate": 0.001,
            "batch_size": 10,
            "epochs": 1,
        },
        "normalize_inputs": True,
    }
