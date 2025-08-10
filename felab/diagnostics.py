"""
felab/diagnostics.py

Диагностика для FELAB:
- Оценка числа обусловленности матриц.
- Апостериорные оценки через остаток r = f - L u_N.
- Построение «дуальной» оценки ||r||_{V*} в подпространстве span{ψ}
  (решается вспомогательная Рисс-задача A(z, v) = (r, v) для всех v∈span{ψ}).

Опорные обозначения:
  L u = ε^α D_C^α u + a(x) u,
  A(u,v) = ε^α (D_C^{α/2}u, D_C^{α/2}v) + (a u, v).

Публичное API:
  - condition_number(A) -> float
  - residual_norms(basis, coeffs, u0, f_fun, *, return_details: bool=False) -> dict

Зависимости: numpy, felab.basis, felab.assemble, felab.fracops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .basis import Basis
from .assemble import build_solution_callable, l2_inner
from .fracops import caputo_pi_on_grid


Array = np.ndarray
Func = Callable[[Array | float], Array]


# --------------------------- КОНДИЦИОНИРОВАНИЕ ----------------------------- #
def condition_number(A: Array, ord: int = 2) -> float:
    """
    Число обусловленности матрицы A в норме 2 (по SVD). Возвращает NaN при сбое.
    """
    try:
        s = np.linalg.svd(A, compute_uv=False)
        smin = s.min()
        if smin == 0 or not np.isfinite(smin):
            return float("inf")
        return float(s.max() / smin)
    except Exception:
        return float("nan")


# ----------------------- ВСПОМОГАТЕЛЬНЫЕ ВЫЧИСЛЕНИЯ ------------------------ #
def _build_stiffness_from_basis(basis: Basis) -> Array:
    """
    Сборка матрицы жёсткости K_{ij} = A(ψ_j, ψ_i) только из Basis.
    """
    psi = basis.psi
    N = len(psi)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        qi = psi[i]
        for j in range(i, N):
            qj = psi[j]
            val = basis.inner(qj, qi)
            K[i, j] = val
            K[j, i] = val
    return K


def _caputo_alpha_on_nodes(u_vals: Array, nodes: Array, alpha: float) -> Array:
    """
    D_C^α u на узлах nodes по значениями u(nodes) (продукт-интегрирование).
    """
    return caputo_pi_on_grid(nodes, u_vals, nu=float(alpha))


def _residual_on_quad_nodes(
    basis: Basis,
    coeffs: Array,
    u0: float,
    f_fun: Callable[[Array | float], Array],
) -> Tuple[Array, Array, Array]:
    """
    Вычислить остаток r(nodes) = f(nodes) - [ε^α D_C^α u_N + a u_N] на узлах квадратуры.
    Возвращает (x, w, r_x).
    """
    quad = basis.quad
    x, w = quad.nodes, quad.weights

    # Построим u_N и выборки на узлах
    uN = build_solution_callable(basis, coeffs, u0)
    u_vals = np.asarray(uN(x), dtype=float)

    # D_C^α u_N на узлах
    Dalpha_u = _caputo_alpha_on_nodes(u_vals, x, alpha=basis.alpha)

    # Правая часть и коэффициент a(x)
    f_vals = np.asarray(f_fun(x), dtype=float)
    a_vals = np.asarray(basis.a_fun(x), dtype=float)

    # L u_N
    Lu = (basis.eps ** basis.alpha) * Dalpha_u + a_vals * u_vals

    r = f_vals - Lu
    return x, w, r


# ------------------------------ ОСНОВНЫЕ ОЦЕНКИ ---------------------------- #
def residual_norms(
    basis: Basis,
    coeffs: Array,
    u0: float,
    f_fun: Callable[[Array | float], Array],
    *,
    return_details: bool = False,
) -> Dict[str, float]:
    """
    Апостериорные оценки на узлах квадратуры basis.quad.

    Возвращает словарь с ключами:
      - "res_l2":        ||r||_{L2(0,T)}
      - "res_linf_quad": max_{узлы квадратуры} |r|
      - "vstar_est":     оценка ||r||_{V*} через Рисс-проекцию на span{ψ}
                         (решаем K z = b_r, где (b_r)_i = (r, ψ_i)_{L2};
                          тогда ||r||_{V*} ≥ ||z||_{ε,α,a} = sqrt(z^T K z))
      - "cond_K":        число обусловленности K (диагностика)

    Если return_details=True, добавляет:
      - "N":        размерность подпространства
      - "eps":      ε, "alpha": α, "T": длина интервала
    """
    # Остаток на узлах
    x, w, r = _residual_on_quad_nodes(basis, coeffs, u0, f_fun)

    # L2-норма остатка
    res_l2 = float(np.sqrt(np.dot(w, r * r)))

    # Linf-на узлах квадратуры
    res_linf = float(np.max(np.abs(r)))

    # Соберём правую часть для Рисс-задачи на span{ψ}: b_r[i] = (r, ψ_i)_{L2}
    psi = basis.psi
    N = len(psi)
    b_r = np.empty(N, dtype=float)
    for i, qi in enumerate(psi):
        b_r[i] = l2_inner(basis, lambda t, qi=qi, x=x, r=r: np.interp(t, x, r), qi)

    # Матрица жёсткости K
    K = _build_stiffness_from_basis(basis)

    # Решим K z = b_r (плотно; K, как правило, SPD)
    try:
        L = np.linalg.cholesky(K)
        y = np.linalg.solve(L, b_r)
        z = np.linalg.solve(L.T, y)
        backend = "cholesky"
    except np.linalg.LinAlgError:
        z = np.linalg.solve(K, b_r)
        backend = "lu"

    # Оценка ||z||_{ε,α,a} = sqrt(z^T K z) — нижняя оценка ||r||_{V*}
    vstar_est = float(np.sqrt(z @ (K @ z)))

    out: Dict[str, float] = {
        "res_l2": res_l2,
        "res_linf_quad": res_linf,
        "vstar_est": vstar_est,
        "cond_K": condition_number(K),
    }

    if return_details:
        out.update({
            "N": float(N),
            "alpha": float(basis.alpha),
            "eps": float(basis.eps),
            "T": float(basis.T),
        })

    return out


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Небольшой e2e тест на синтетической задаче (как в assemble/api)
    from .basis import build_basis
    import mpmath as mp

    T = 1.0
    alpha = 0.6
    eps = 1e-3
    a_fun = lambda x: 1.0
    u0 = 0.8

    B = build_basis(N=10, alpha=alpha, eps=eps, a_fun=a_fun, T=T, raw_family="chebyshev", quad_n=256)
    lam = -B.a0 / (eps ** alpha)

    def u_true(x):
        xv = np.asarray(x, dtype=float)
        return u0 * B.phi0(xv) + xv + 0.2 * xv**2

    def Dcap_alpha_poly(x):
        xv = np.asarray(x, dtype=float)
        return (mp.gamma(2)/mp.gamma(2-alpha)) * xv**(1-alpha) + \
               0.2 * (mp.gamma(3)/mp.gamma(3-alpha)) * xv**(2-alpha)

    def f_fun(x):
        xv = np.asarray(x, dtype=float)
        return (eps**alpha) * (lam * B.phi0(xv) + Dcap_alpha_poly(xv)) + u_true(xv)

    # Соберём систему и решим (локально, чтобы получить coeffs)
    from .assemble import assemble_system
    from .solvers import solve_system, SolveOptions

    K, b, extras = assemble_system(B, f_fun, u0=u0)
    c, _ = solve_system(K, b, SolveOptions(backend="cholesky"))

    # Диагностика остатка
    diag = residual_norms(B, c, u0, f_fun, return_details=True)
    assert diag["res_l2"] < 1e-2, f"Слишком большой остаток L2: {diag}"
    assert np.isfinite(diag["vstar_est"]), "Оценка V* должна быть конечной"

    print("diagnostics.py basic self-tests passed.")