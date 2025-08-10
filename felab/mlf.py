"""
felab/mlf.py

Mittag–Leffler functions:
  - E_{α,β}(z)  for α > 0, β ∈ ℝ (commonly β > 0)
  - E_{α}(z) := E_{α,1}(z)

Design goals for FELAB:
  * Numerically stable on the negative real axis (z < 0), which is crucial for
    φ₀(x; ε) = E_α(− a(0) x^α / ε^α).
  * Works for scalars and NumPy arrays (complex supported).
  * Two regimes with automatic switch:
      (i) power series for small/medium |z|
     (ii) algebraic asymptotics for large |z| with arg(z) ≈ π (negative real).
  * Minimal dependencies: uses `mpmath` for Γ(·) and complex arithmetic.

References (identities used):
  - Series definition:  E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(α k + β)
  - Asymptotics (0 < α < 2, |arg z| ≤ π−δ):
        E_{α,β}(z) ~ - Σ_{k=1}^M z^{-k} / Γ(β - α k),   as |z| → ∞,
    with exponentially small contributions away from principal Stokes lines.
  - Special cases: E_{1,1}(z) = e^z.

API:
  mittag_leffler(alpha, z, beta=1.0, tol=1e-14, max_terms=2000,
                 z_switch=None, max_asymp=20)

Notes:
  * For general complex z far from the negative real axis, the series often
    converges well; the asymptotic part is primarily used/stable for z ≪ 0.
  * If you know your use-case is strictly z ≤ 0 real and 0<α<1, the chosen
    defaults are well suited.

"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union
import math
import cmath

import numpy as _np
import mpmath as _mp


Number = Union[float, complex]
ArrayLike = Union[Number, Iterable[Number], "._np.ndarray"]


# ------------------------------- HELPERS ----------------------------------- #
def _is_real_neg(z: complex, tol_angle: float = 1e-12) -> bool:
    """
    Heuristic: is z on (or very close to) the negative real axis?
    """
    if z == 0:
        return False
    ang = cmath.phase(z)
    # Compare to ±π
    return abs(abs(ang) - math.pi) <= tol_angle


def _as_mpc(z: Number) -> _mp.mpf | _mp.mpc:
    """Convert Python/NumPy number to mpmath mpf/mpc."""
    if isinstance(z, complex):
        return _mp.mpc(z.real, z.imag)
    # NumPy scalars
    if hasattr(z, "dtype"):
        z = complex(z)
        if abs(z.imag) > 0:
            return _mp.mpc(z.real, z.imag)
        return _mp.mpf(z.real)
    # Python floats/ints
    if isinstance(z, (int, float)):
        return _mp.mpf(z)
    # Fallback
    zc = complex(z)
    if abs(zc.imag) > 0:
        return _mp.mpc(zc.real, zc.imag)
    return _mp.mpf(zc.real)


def _series_Eab(alpha: float, beta: float, z: Number,
                tol: float, max_terms: int) -> complex:
    """
    Power series for E_{α,β}(z). Uses mp.gamma; partial sums until term < tol.
    """
    mpz = _as_mpc(z)
    # Fast paths
    if alpha == 1.0 and beta == 1.0:
        # E_{1,1}(z) = e^z
        return complex(_mp.e**mpz)

    s = _mp.mpc(0)
    k = 0
    # Accumulate until next term is small in absolute value
    while k < max_terms:
        ak = alpha * k + beta
        try:
            g = _mp.gamma(ak)
        except Exception:
            # If gamma under/overflows, break (rare for reasonable k)
            break
        term = (mpz ** k) / g
        s += term
        if abs(term) < tol:
            break
        k += 1
    return complex(s)


def _asymptotic_Eab(alpha: float, beta: float, z: Number,
                    M: int) -> complex:
    """
    Algebraic asymptotic expansion for large |z| on/near the negative real axis:

        E_{α,β}(z) ~ - Σ_{k=1}^M z^{-k} / Γ(β - α k)

    This provides a real, rapidly decaying tail for z < 0 (dominant regime in FELAB).
    Valid for 0 < α < 2, |arg z| ≤ π (excluding the Stokes switching intricacies).
    """
    mpz = _as_mpc(z)
    s = _mp.mpc(0)
    for k in range(1, M + 1):
        ak = beta - alpha * k
        try:
            g = _mp.gamma(ak)
        except Exception:
            # Γ(β-αk) may hit poles for some β; then that term is undefined.
            # We skip such terms (they are zero by analytic continuation in many cases).
            continue
        # Avoid division by zero if gamma is inf
        if _mp.isnan(g) or _mp.isinf(g) or g == 0:
            continue
        s += (mpz ** (-k)) / g
    s = -s
    return complex(s)


def _auto_switch_threshold(alpha: float, beta: float, tol: float) -> float:
    """
    Heuristic threshold for |z| to switch from series to asymptotics on the
    negative real axis. The larger α or stricter tol, the larger threshold.
    """
    # Simple heuristic based on observed convergence:
    # For 0<α<1, series decays ~k! growth in Γ(αk+β), but |z|^k grows.
    # Set threshold ~ 20..60 depending on tol.
    base = 30.0
    # Tighten with smaller tol
    adj = 10.0 * max(0.0, math.log10(1.0 / max(tol, 1e-16)))
    # α close to 0 slows Γ(αk+β) growth ⇒ be more conservative
    alpha_adj = 10.0 * (0.7 - min(max(alpha, 0.05), 0.7))
    return max(10.0, base + 0.25 * adj + alpha_adj)


# ------------------------------ PUBLIC API --------------------------------- #
def mittag_leffler(alpha: float,
                   z: ArrayLike,
                   beta: float = 1.0,
                   tol: float = 1e-14,
                   max_terms: int = 2000,
                   z_switch: Optional[float] = None,
                   max_asymp: int = 20) -> ArrayLike:
    """
    Compute E_{α,β}(z) for scalar or array `z`.

    Parameters
    ----------
    alpha : float
        Parameter α > 0. For FELAB core use-cases: 0 < α < 1.
    z : scalar or array-like (real/complex)
        Evaluation points.
    beta : float, optional
        Parameter β (default 1.0). Common cases: β=1 for E_α.
    tol : float, optional
        Absolute tolerance for terminating the series.
    max_terms : int, optional
        Maximum number of series terms.
    z_switch : float or None, optional
        If provided, for real negative z with |z| >= z_switch, uses asymptotics.
        If None, a heuristic threshold is chosen.
    max_asymp : int, optional
        Number of terms in algebraic asymptotic expansion.

    Returns
    -------
    E : complex or ndarray of complex
        Values of E_{α,β}(z). If input is real and result is real (within 1e-15),
        we return real dtype for convenience.

    Notes
    -----
    * Special case α=1, β=1 uses exp(z).
    * Asymptotic branch is only activated for real negative z; otherwise, series is used.
    * This routine is designed to be robust and adequate for FELAB needs; for extreme
      parameters/domains you might prefer specialized libraries.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if max_terms <= 0:
        raise ValueError("max_terms must be > 0")
    if max_asymp <= 0:
        raise ValueError("max_asymp must be > 0")

    # Set mpmath precision based on tol (heuristic).
    # 53 bits (double) ~ 16 decimal digits; we add margin.
    dps = max(25, int(10 - math.log10(max(tol, 1e-30))) + 20)
    _mp.mp.dps = min(max(dps, 25), 100)  # cap to 100 to avoid slowdowns

    # Scalar path
    if _np.isscalar(z):
        return _mlf_scalar(alpha, beta, z, tol, max_terms, z_switch, max_asymp)

    # Vectorized path
    z_arr = _np.asarray(z)
    out = _np.empty(z_arr.shape, dtype=complex)
    it = _np.nditer(z_arr, flags=['multi_index', 'refs_ok'])
    # Automatic threshold if needed
    thr = z_switch if (z_switch is not None) else _auto_switch_threshold(alpha, beta, tol)

    while not it.finished:
        zz = it[0].item()
        out[it.multi_index] = _mlf_scalar(alpha, beta, zz, tol, max_terms, thr, max_asymp)
        it.iternext()

    # Cast to real if imaginary parts are numerically zero
    if _np.all(_np.isfinite(out)):
        imag_max = _np.max(_np.abs(out.imag))
        if imag_max < 1e-15:
            return out.real
    return out


def _mlf_scalar(alpha: float, beta: float, z: Number,
                tol: float, max_terms: int,
                z_switch: Optional[float], max_asymp: int) -> complex:
    """
    Decide between series and asymptotics (for real negative, large |z|).
    """
    # Special case: E_{1,1}(z) = exp(z)
    if alpha == 1.0 and beta == 1.0:
        return complex(cmath.exp(z))

    # Real-negative asymptotic branch
    if z_switch is not None:
        # Treat "close to negative real axis" as real negative (we only switch for exactly real negative).
        if isinstance(z, (int, float)) and (z < 0.0) and (abs(z) >= z_switch) and (0.0 < alpha < 2.0):
            return _asymptotic_Eab(alpha, beta, z, M=max_asymp)

        # Handle numpy scalar real types
        if hasattr(z, "dtype") and _np.isrealobj(z):
            zf = float(z)
            if (zf < 0.0) and (abs(zf) >= z_switch) and (0.0 < alpha < 2.0):
                return _asymptotic_Eab(alpha, beta, zf, M=max_asymp)

    # Default: series
    return _series_Eab(alpha, beta, z, tol=tol, max_terms=max_terms)


# ----------------------- CONVENIENCE SHORTCUTS ----------------------------- #
def E(alpha: float, z: ArrayLike, tol: float = 1e-14, **kwargs: Any) -> ArrayLike:
    """
    Shortcut for E_{α}(z) = E_{α,1}(z). Additional kwargs are passed through.
    """
    return mittag_leffler(alpha, z, beta=1.0, tol=tol, **kwargs)


def Eab(alpha: float, beta: float, z: ArrayLike, tol: float = 1e-14, **kwargs: Any) -> ArrayLike:
    """
    Shortcut for E_{α,β}(z). Additional kwargs are passed through.
    """
    return mittag_leffler(alpha, z, beta=beta, tol=tol, **kwargs)


# ------------------------------ SELF-TESTS --------------------------------- #
if __name__ == "__main__":
    # Basic sanity tests
    import numpy as np

    # 1) Special case: E_{1,1}(z) = exp(z)
    for val in [0.0, -1.0, 2.3, 1+2j, -3+0j]:
        e1 = mittag_leffler(1.0, val, beta=1.0)
        assert abs(e1 - cmath.exp(val)) < 1e-12, f"E_{1,1} mismatch at {val}"

    # 2) Small z, α=0.5, β=1
    z_small = np.array([0.0, -0.1, 0.1, 0.2])
    E_small = mittag_leffler(0.5, z_small, beta=1.0)
    # Compare symmetry around 0 for small z via first 2-3 terms
    # E_{α}(z) ≈ 1 + z/Γ(α+1) + z^2/Γ(2α+1)
    approx = []
    for zz in z_small:
        t = 1.0 + zz / math.gamma(0.5 + 1.0) + (zz**2) / math.gamma(2*0.5 + 1.0)
        approx.append(t)
    approx = np.array(approx)
    assert np.allclose(E_small, approx, atol=1e-2), "Small-z series sanity check failed"

    # 3) Negative large z: compare series vs asymptotics transition (no strict reference)
    z_vals = np.array([-5, -10, -50, -100], dtype=float)
    Ea = mittag_leffler(0.6, z_vals, beta=1.0, z_switch=30.0, max_asymp=10)
    # Ensure finite and decreasing magnitude
    assert np.all(np.isfinite(Ea)), "Asymptotic values must be finite"
    assert abs(Ea[-1]) <= abs(Ea[0]) + 1e-9, "Magnitude should decay for more negative z"

    print("mlf.py basic self-tests passed.")