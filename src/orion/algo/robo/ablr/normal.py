""" 'Patch' for torch.distributions.Normal, so we can add or multiply
distributions with constants of other distributions.

Basically just having fun here.
"""

from torch.distributions import Distribution
from torch.distributions import Normal as NormalBase
from typing import Union, Any
from torch import Tensor
import torch


class Normal(NormalBase):
    def __init__(self, loc, scale=None, variance=None, *args, **kwargs):
        if variance is not None:
            assert scale is None
            scale = torch.sqrt(variance)
        super().__init__(loc=loc, scale=scale, *args, **kwargs)

    def __add__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            return Normal(self.mean + other, scale=self.scale)
        if isinstance(other, NormalBase):
            # NOTE: Assuming that the two variables are independent
            # raise NotImplementedError("Only support addition with a constant for now")
            return Normal(
                self.mean + other.mean, variance=self.variance + other.variance
            )
        return NotImplemented

    def __radd__(self, other: Union["Normal", Any]) -> "Normal":
        return self + other

    def __sub__(self, other: Union["Normal", Any]) -> "Normal":
        return self + (-other)

    def __rsub__(self, other: Union["Normal", Any]) -> "Normal":
        return (-self) + other

    def __rsub__(self) -> "Normal":
        return (-1) * self

    def __mul__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            return Normal(self.mean * other, variance=self.scale * other ** 2)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support multiplication by constant for now")
            return Normal(
                self.mean * other.mean,
                variance=(self.mean ** 2 * other.variance)
                * (other.mean ** 2 * self.variance),
            )
        return NotImplemented

    def __rmul__(self, other: Union["Normal", Any]) -> "Normal":
        return self * other

    def __truediv__(self, other: Union["Normal", Any]) -> "Normal":
        if isinstance(other, (int, float, Tensor)):
            return self * (1 / other)

        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support division by a constant for now")
        return NotImplemented

    def __rtruediv__(self, other: Union["Normal", Any]) -> "Normal":
        # other / self failed.
        return other * (1 / self)
