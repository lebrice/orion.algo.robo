""" Utility functions for linalg operations that may fail."""
from __future__ import annotations

from functools import partial
from logging import getLogger as get_logger
from typing import Callable

import torch
from torch import Tensor
from typing_extensions import Protocol

logger = get_logger(__name__)


def minimum_variance(v: Tensor, minimum_variance=1e-8):
    """
    Minimum variance, because the context vectors are actually always
    the same, so the variance along that dimension is 0, which makes
    NaN values in the normalized inputs.
    """
    variance = v.var(0)
    variance = variance.nan_to_num(minimum_variance)
    variance[variance < minimum_variance] = minimum_variance
    return variance


# pylint: disable=too-few-public-methods
class OffsetFunction(Protocol):
    """Callable that adds some offset/noise to the given matrix to make the cholesky decomposition
    work.

    Parameters
    ----------
    some_matrix : Tensor
        some matrix that will be passed to a function like `torch.linalg.cholesky`.
    attempt : int
        the current attempt number
    max_attempts : int
        The max number of attempts.
        NOTE: Currently unused, but the idea is that if we used a "maximum possible noise"
        below, we could use `attempt` and `max_attempts` to gage how much noise to add.

    Returns
    -------
    Tensor
        `some_matrix`, with some added offset / noise.
    """

    def __call__(self, some_matrix: Tensor, attempt: int, max_attempts: int) -> Tensor:
        raise NotImplementedError


# pylint: disable=unused-argument
def offset_function(some_matrix: Tensor, attempt: int, max_attempts: int) -> Tensor:
    """Offset function that adds an offset * torch.eye(n) to the matrix if it is square.
    Otherwise, adds some random noise of progressively larger value.
    """
    if some_matrix.shape[-2] == some_matrix.shape[-1]:
        # If the matrix is square, add an offset to the diagonal:
        offset = 0.1 * (2 ** (attempt - 1))
        return some_matrix + torch.eye(some_matrix.shape[-1]) * offset
    # Add progressively larger random noise?
    noise_std = 0.1 * (2 ** (attempt - 1))
    return some_matrix + torch.randn_like(some_matrix) * noise_std


def try_function(
    function: Callable[[Tensor], Tensor],
    some_matrix: Tensor,
    max_attempts: int = 10,
    offset_function: OffsetFunction = offset_function,
) -> Tensor:
    """Attempt to apply the given function of the given matrix, adding progressively
    larger offset/noise matrices until it works, else raises an error.
    """
    result: Tensor | None = None
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    attempt = 0
    error: RuntimeError | None = None
    for attempt in range(max_attempts):
        m: Tensor
        if attempt == 0:
            m = some_matrix
        else:
            m = offset_function(some_matrix, attempt, max_attempts)

        try:
            result = function(m)
            if attempt > 0:
                logger.debug(
                    "Managed to get the operation to work after %s attempts.", attempt
                )
            return result
        except RuntimeError as e:
            error = e

    raise RuntimeError(
        f"{function.__name__} didn't work, even after {attempt} attempts:\n"
        f"{error}\n"
        f"(matrix: {some_matrix})"
    ) from (error if error is not None else None)


try_get_cholesky = partial(try_function, torch.linalg.cholesky)
try_get_cholesky_inverse = partial(try_function, torch.cholesky_inverse)
