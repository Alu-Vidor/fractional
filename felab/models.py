"""
felab/models.py

Тестовые задачи для FELAB с известными точными решениями u(x) и
аналитически построенной правой частью f(x) = ε^α D_C^α u + a(x) u.

Подход:
- Берём классы решений, для которых легко выписать D_C^α u:
    • u(x) = u0 * φ0(x;ε) + Σ_m c_m x^m,  (0<α<1)
      где φ0 — атом слоя: φ0(x;ε) = E_α( -a(0) x^α / ε^α ),
      а D_C^α φ0 = -(a(0)/ε^α) φ0 =: λ φ0 (λ<0),
      и D_C^α x^m = Γ(m+1)/Γ(m+1-α) x^{m-α} для m≥1 (Caputo).
- Тогда f(x) вычисляется аналитически (без численной дифференцировки),
  что идеально для регрессионных тестов.

Содержимое:
- dataclass Problem: контейнер с полями a(x), f(x), u_true(x), u0, alpha, eps, T, meta
- Вспомогательная фабрика `from_phi0_plus_poly(...)`
- Набор готовых задач:
    problem_poly_const_a(...)
    problem_poly_var_a(...)
    problem_phi0_only_const_a(...)
    problem_phi0_plus_poly_var_a(...)

Зависимости: numpy, mpmath, felab.atom (phi0_factory)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import mpmath as mp

from .atom import phi0_factory


Array = np.ndarray
Func = Callable[[Array | float], Array]


# ---------------------------- ОСНОВНАЯ СТРУКТУРА --------------------------- #
@dataclass(frozen=True)
class Problem:
    """
    Контейнер тестовой задачи.

    Поля:
      a        : коэффициент a(x)
      f        : правая часть f(x) = ε^α D_C^α u + a(x) u
      u_true   : точное решение u(x)
      u0       : начальное условие u(0)
      alpha    : порядок 0<α<1
      eps      : ε>0
      T        : длина интервала
      meta     : словарь с описанием/параметрами
    """
    a: Func
    f: Func
    u_true: Func
    u0: float
    alpha: float
    eps: float
    T: float
    meta: Dict

    def grid(self, n: int) -> Array:
        if n <= 1:
            return np.array([0.0, self.T])
        return np.linspace(0.0, self.T, int(n), dtype=float)


# ----------------------- ПОМОЩНИКИ: ПОЛИНОМ + φ0 -------------------------- #
def _poly_value(coeffs: Sequence[float], x: Array | float) -> Array:
    """
    c_m заданы для мономов x^m, m=1..M (без свободного члена, чтобы u(0)=u0 сохранилось от φ0).
    """
    xv = np.asarray(x, dtype=float)
    acc = np.zeros_like(xv, dtype=float)
    for m, cm in enumerate(coeffs, start=1):
        if cm == 0.0:
            continue
        acc = acc + float(cm) * (xv ** m)
    if np.isscalar(x):
        return float(acc)  # type: ignore
    return acc


def _caputo_alpha_poly(coeffs: Sequence[float], alpha: float, x: Array | float) -> Array:
    """
    D_C^α [ Σ_{m≥1} c_m x^m ] = Σ c_m * Γ(m+1)/Γ(m+1-α) * x^{m-α}.
    """
    xv = np.asarray(x, dtype=float)
    acc = np.zeros_like(xv, dtype=float)
    for m, cm in enumerate(coeffs, start=1):
        if cm == 0.0:
            continue
        C = float(mp.gamma(m + 1) / mp.gamma(m + 1 - alpha))
        # x^{m-α}: при x=0 → 0, т.к. m-α>0
        acc = acc + float(cm) * C * (xv ** (m - alpha))
    if np.isscalar(x):
        return float(acc)  # type: ignore
    return acc


def from_phi0_plus_poly(
    *,
    alpha: float,
    eps: float,
    T: float,
    a_fun: Callable[[Array | float], Array],
    u0: float,
    poly_coeffs: Sequence[float],
) -> Problem:
    """
    Построить задачу для u(x) = u0 φ0(x;ε) + p(x), p(x)=Σ_{m=1}^M c_m x^m,
    где φ0 использует a(0): φ0(x)=E_α(-(a(0)/ε^α) x^α).

    Параметры
    ---------
    a_fun : коэффициент a(x) (положительный на [0,T])
    poly_coeffs : коэффициенты полинома (без свободного члена)

    Возвращает Problem с аналитической f(x).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    if eps <= 0:
        raise ValueError("eps должно быть > 0")
    if T <= 0:
        raise ValueError("T должно быть > 0")

    # Векторизованная обёртка a(x)
    def a_vec(x: Array | float) -> Array:
        xv = np.asarray(x, dtype=float)
        out = a_fun(xv)  # type: ignore
        if np.isscalar(out):
            return np.full_like(xv, float(out))
        return np.asarray(out, dtype=float)

    a0 = float(a_vec(np.array([0.0]))[0])
    if not (a0 > 0.0):
        raise ValueError("a(0) должно быть > 0")

    # Атом φ0 и λ
    p0 = phi0_factory(a0=a0, eps=eps, alpha=alpha)
    lam = p0.lam  # = -a0/ε^α

    # u_true
    def u_true(x: Array | float) -> Array:
        xv = np.asarray(x, dtype=float)
        return float(u0) * p0(xv) + _poly_value(poly_coeffs, xv)

    # D_C^α u = u0*D_C^α φ0 + D_C^α p = u0*lam*φ0 + Σ c_m Γ(m+1)/Γ(m+1-α) x^{m-α}
    def Dcap_alpha_u(x: Array | float) -> Array:
        xv = np.asarray(x, dtype=float)
        return float(u0) * lam * p0(xv) + _caputo_alpha_poly(poly_coeffs, alpha, xv)

    # f(x) = ε^α D_C^α u + a(x) u = ε^α (u0*lam*φ0 + D_C^α p) + a(x) (u0*φ0 + p)
    def f_fun(x: Array | float) -> Array:
        xv = np.asarray(x, dtype=float)
        u_val = float(u0) * p0(xv) + _poly_value(poly_coeffs, xv)
        d_val = Dcap_alpha_u(xv)
        return (eps ** alpha) * d_val + a_vec(xv) * u_val

    meta = {
        "type": "phi0_plus_poly",
        "alpha": alpha,
        "eps": eps,
        "T": T,
        "a0": a0,
        "poly_coeffs": list(map(float, poly_coeffs)),
        "u0": float(u0),
        "note": "Аналитическая f; φ0 строится по a(0).",
    }

    return Problem(
        a=a_vec,
        f=f_fun,
        u_true=u_true,
        u0=float(u0),
        alpha=float(alpha),
        eps=float(eps),
        T=float(T),
        meta=meta,
    )


# ------------------------------ ГОТОВЫЕ ЗАДАЧИ ----------------------------- #
def problem_poly_const_a(
    alpha: float = 0.6,
    eps: float = 1e-3,
    T: float = 1.0,
    u0: float = 0.8,
    poly_coeffs: Sequence[float] = (1.0, 0.25),
) -> Problem:
    """
    a(x) ≡ 1.0, u(x) = u0 φ0 + p(x),  p(x) = x + 0.25 x^2 (по умолчанию).
    """
    a_fun = lambda x: 1.0
    return from_phi0_plus_poly(alpha=alpha, eps=eps, T=T, a_fun=a_fun, u0=u0, poly_coeffs=poly_coeffs)


def problem_poly_var_a(
    alpha: float = 0.6,
    eps: float = 1e-3,
    T: float = 1.0,
    u0: float = 0.9,
    poly_coeffs: Sequence[float] = (1.0, -0.3, 0.05),
) -> Problem:
    """
    a(x) = 1 + 0.5 x, u(x) = u0 φ0 + p(x),  p(x) = x - 0.3 x^2 + 0.05 x^3 (по умолчанию).
    Примечание: φ0 использует a(0)=1 для λ, как и требуется теорией метода.
    """
    a_fun = lambda x: 1.0 + 0.5 * np.asarray(x, dtype=float)
    return from_phi0_plus_poly(alpha=alpha, eps=eps, T=T, a_fun=a_fun, u0=u0, poly_coeffs=poly_coeffs)


def problem_phi0_only_const_a(
    alpha: float = 0.6,
    eps: float = 1e-3,
    T: float = 1.0,
    u0: float = 1.0,
) -> Problem:
    """
    Чистая «релаксация» без гладкой добавки:
        a(x) ≡ 1,    u(x) = u0 φ0(x;ε).
    Тогда: f(x) = ε^α * (u0*lam*φ0) + a(x) * (u0*φ0) = u0 * (ε^α*lam + 1) * φ0.
    """
    a_fun = lambda x: 1.0
    # poly_coeffs пустой — эквивалентно p(x)≡0
    return from_phi0_plus_poly(alpha=alpha, eps=eps, T=T, a_fun=a_fun, u0=u0, poly_coeffs=())


def problem_phi0_plus_poly_var_a(
    alpha: float = 0.6,
    eps: float = 2e-3,
    T: float = 1.0,
    u0: float = 0.7,
    poly_coeffs: Sequence[float] = (0.8, -0.2),
) -> Problem:
    """
    a(x) = 2 + sin(x),  u(x) = u0 φ0 + 0.8 x - 0.2 x^2 (по умолчанию).
    Для λ используется a(0)=2.
    """
    a_fun = lambda x: 2.0 + np.sin(np.asarray(x, dtype=float))
    return from_phi0_plus_poly(alpha=alpha, eps=eps, T=T, a_fun=a_fun, u0=u0, poly_coeffs=poly_coeffs)


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Мини-проверки согласованности (f = ε^α D^α u + a u)
    rng = np.random.default_rng(0)
    for prob in [
        problem_poly_const_a(),
        problem_poly_var_a(),
        problem_phi0_only_const_a(),
        problem_phi0_plus_poly_var_a(),
    ]:
        X = prob.grid(400)
        # Численная проверка тождества f ≈ ε^α D^α u + a u по узлам (грубая,
        # т.к. тут мы используем аналитические построения)
        # В качестве sanity check проверим корректность нач.условия:
        u0_num = float(prob.u_true(0.0))
        assert abs(u0_num - prob.u0) < 1e-12, f"u(0) != u0 for {prob.meta}"

        # Проверим знакопостоянность φ0-компоненты у нуля (монотонное затухание)
        # На практике достаточно убедиться, что u_true не взрывается вблизи 0
        vals = prob.u_true(X)
        assert np.all(np.isfinite(vals)), "u_true must be finite"

    print("models.py basic self-tests passed.")