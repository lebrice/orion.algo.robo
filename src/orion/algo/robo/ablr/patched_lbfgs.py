""" Patch for the LBFGS optimizer. """
from __future__ import annotations

import torch


class PatchedLBFGS(torch.optim.LBFGS):
    """Patched version of LBFGS optimizer, that correctly saves the state."""

    _params: list
    _numel_cache: int | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()  # type: ignore
        state["_params"] = self._params
        state["_numel_cache"] = self._numel_cache
        return state

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["_params"] = self._params
        state_dict["_numel_cache"] = self._numel_cache
        return state_dict

    def __setstate__(self, state):
        super().__setstate__(state)
