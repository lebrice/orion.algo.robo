"""Patch for `torch.distributions.Normal`, so that we can add/multiply distributions by constants.
"""

from typing import Any, Optional, Union

from torch import Tensor
from torch.distributions import Normal as NormalBase


class Normal(NormalBase):
    def __init__(
        self, loc: Tensor, scale: Tensor, validate_args: Optional[bool] = None
    ):
        super().__init__(loc=loc, scale=scale, validate_args=validate_args)

    def __add__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            return Normal(self.mean + other, scale=self.scale)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support addition with a constant for now")
        return NotImplemented

    def __radd__(self, other: Union["Normal", Any]) -> "Normal":
        return self + other

    def __sub__(self, other: Union["Normal", Any]) -> "Normal":
        return self + (-other)

    def __rsub__(self, other: Union["Normal", Any]) -> "Normal":
        return (-self) + other

    def __neg__(self) -> "Normal":
        return (-1) * self

    def __mul__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            # TODO: Double-check that this is correct.
            return Normal(self.mean * other, scale=self.scale * other**2)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support multiplication by constant for now")
        return NotImplemented

    def __rmul__(self, other: Union["Normal", Any]) -> "Normal":
        return self * other

    def __truediv__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            return self * (1 / other)

        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support division by a constant for now")
        return NotImplemented
