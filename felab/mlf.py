"""
felab/mlf.py

Mittag–Leffler functions computed via the :mod:`pymittagleffler` library.
This wrapper exposes the same API as the previous internal implementation
but delegates the heavy lifting to the external package.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import numpy as np
from pymittagleffler import mittag_leffler as _mlf

Number = Union[float, complex]
ArrayLike = Union[Number, Iterable[Number], np.ndarray]


def mittag_leffler(
    alpha: float,
    z: ArrayLike,
    beta: float = 1.0,
    tol: float = 1e-14,
    max_terms: int = 2000,
    z_switch: Optional[float] = None,
    max_asymp: int = 20,
) -> ArrayLike:
    """Compute :math:`E_{\alpha,\beta}(z)` using ``pymittagleffler``.

    Parameters other than ``alpha``, ``z`` and ``beta`` are accepted for
    backward compatibility but are not used.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    result = _mlf(z, alpha, beta)

    # Clean up tiny imaginary parts for real inputs
    if np.isscalar(result):
        result = complex(result)
        if abs(result.imag) < 1e-15:
            return result.real
        return result

    result = np.asarray(result)
    if np.max(np.abs(result.imag)) < 1e-15:
        return result.real
    return result


def E(alpha: float, z: ArrayLike, tol: float = 1e-14, **kwargs: Any) -> ArrayLike:
    """Shortcut for :math:`E_{\alpha}(z) = E_{\alpha,1}(z)`."""
    return mittag_leffler(alpha, z, beta=1.0, tol=tol, **kwargs)


def Eab(alpha: float, beta: float, z: ArrayLike, tol: float = 1e-14, **kwargs: Any) -> ArrayLike:
    """Shortcut for :math:`E_{\alpha,\beta}(z)`."""
    return mittag_leffler(alpha, z, beta=beta, tol=tol, **kwargs)


if __name__ == "__main__":
    import cmath
    import math

    # 1) Special case: E_{1,1}(z) = exp(z)
    for val in [0.0, -1.0, 2.3, 1 + 2j, -3 + 0j]:
        e1 = mittag_leffler(1.0, val, beta=1.0)
        assert abs(e1 - cmath.exp(val)) < 1e-12, f"E_{1,1} mismatch at {val}"

    # 2) Small z, alpha=0.5, beta=1
    z_small = np.array([0.0, -0.1, 0.1, 0.2])
    E_small = mittag_leffler(0.5, z_small, beta=1.0)
    approx = []
    for zz in z_small:
        t = 1.0 + zz / math.gamma(0.5 + 1.0) + (zz ** 2) / math.gamma(2 * 0.5 + 1.0)
        approx.append(t)
    approx = np.array(approx)
    assert np.allclose(E_small, approx, atol=1e-2), "Small-z series sanity check failed"

    print("mlf.py basic self-tests passed.")
