"""Patch for `torch.distributions.Normal`, so that we can add/multiply distributions by constants.
"""
from __future__ import annotations

from typing import Any, TypeVar

from torch import Tensor
from torch.distributions import Normal as NormalBase

C = TypeVar("C", int, float, Tensor)


class Normal(NormalBase):
    def __init__(self, loc: Tensor, scale: Tensor, validate_args: bool | None = None):
        super().__init__(loc=loc, scale=scale, validate_args=validate_args)

    def __add__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            return Normal(self.mean + other, scale=self.scale)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support addition with a constant for now")
        return NotImplemented

    def __iadd__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            self.loc += other
            return self
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support addition with a constant for now")
        return NotImplemented

    def __radd__(self, other: C | Any) -> Normal:
        return self + other

    def __sub__(self, other: C) -> Normal:
        return self + (-other)

    def __rsub__(self, other: C | Any) -> Normal:
        return (-self) + other

    def __neg__(self) -> Normal:
        return (-1) * self

    def __mul__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            # TODO: Double-check that this is correct.
            return Normal(self.mean * other, scale=self.scale * other**2)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support multiplication by constant for now")
        return NotImplemented

    def __imul__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            self.loc *= other
            self.scale *= other**2
            return self
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support multiplication by constant for now")
        return NotImplemented

    def __rmul__(self, other: C | Any) -> Normal:
        return self * other

    def __truediv__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            return self * (1 / other)
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support division by a constant for now")
        return NotImplemented

    def __itruediv__(self, other: C) -> Normal:
        if isinstance(other, (int, float, Tensor)):
            self *= 1 / other
            return self
        if isinstance(other, NormalBase):
            raise NotImplementedError("Only support division by a constant for now")
        return NotImplemented
