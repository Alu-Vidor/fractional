"""
felab/energy.py

ε-энергетическое скалярное произведение для FELAB:

    ⟨u, v⟩_{ε,α,a} := ε^α ( D_C^{α/2} u , D_C^{α/2} v )_{L^2(0,T)} + ( a u , v )_{L^2(0,T)}.

Цели:
- Производственный API с возможностью передать уже известные полу-производные
  (например, для φ₀ и полиномов), чтобы избежать лишней численной аппроксимации.
- Надёжный fallback на численную Caputo-полу-производную через продукт-интегрирование
  на заданной квадратуре (см. felab.fracops и felab.quadrature).
- Минимум зависимостей: numpy, mpmath для Γ (косвенно в fracops при необходимости).

Основные функции:
- inner_energy(u, v, alpha, eps, a_fun, quad, dhalf_u=None, dhalf_v=None)
- energy_norm(u, alpha, eps, a_fun, quad, dhalf_u=None)
- project_L2(f, quad) — вспомогательная интеграция произвольной функции
- make_dhalf_numeric(u, alpha, quad) — построить численный D_C^{α/2} u (интерполятор)

Типы:
- u, v, a_fun: callables от массива numpy (векторизованные) или скаляра.
- quad: объект Quadrature (см. felab.quadrature) или пара (nodes, weights).

Примечание:
- Если dhalf_u/v переданы (callable), они используются напрямую. Иначе по quad.nodes
  строится численная полу-производная через продукт-интегрирование (CaputoPI) и затем
  линейная интерполяция на узлах квадратуры.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .fracops import caputo_interp_callable
from .quadrature import Quadrature


Array = np.ndarray
Callable1 = Callable[[Array], Array]
QuadLike = Union[Quadrature, Tuple[Array, Array]]


# ----------------------------- ВСПОМОГАТЕЛЬНОЕ ----------------------------- #
def _ensure_quad(quad: QuadLike) -> Tuple[Array, Array]:
    """Привести quad к (nodes, weights)."""
    if isinstance(quad, Quadrature):
        return quad.nodes, quad.weights
    nodes, weights = quad  # предположим кортеж
    x = np.asarray(nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or x.size != w.size:
        raise ValueError("Некорректная квадратура: ожидаются одинаковой длины 1D узлы/веса.")
    return x, w


def _as_callable1(f: Callable1 | Callable[[float], float]) -> Callable1:
    """
    Обёртка: гарантировать работу на ndarray (векторизуем при необходимости).
    """
    def g(x: Array) -> Array:
        xv = np.asarray(x, dtype=float)
        out = f(xv)  # type: ignore
        if np.isscalar(out):
            return np.full_like(xv, float(out))
        return np.asarray(out, dtype=float)
    return g


def project_L2(f: Callable1, quad: QuadLike) -> float:
    """
    Быстрое вычисление ∫_0^T f(x) dx по заданной квадратуре.
    """
    x, w = _ensure_quad(quad)
    fv = f(x)
    return float(np.dot(w, fv))


# ------------------------- ЧИСЛЕННАЯ ПОЛУ-ПРОИЗВОДНАЯ ---------------------- #
def make_dhalf_numeric(
    u: Callable1 | Callable[[float], float],
    alpha: float,
    quad: QuadLike,
) -> Callable1:
    """
    Построить численный интерполятор D_C^{α/2} u(x) по узлам квадратуры.

    Используется продукт-интегрирование на узлах quad.nodes и линейная интерполяция.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    u_fun = _as_callable1(u)
    x, _ = _ensure_quad(quad)
    cap = caputo_interp_callable(u_fun, x, nu=0.5 * float(alpha))
    return lambda t: cap(t)  # t может быть скаляром или массивом


# -------------------------- ЭНЕРГЕТИЧЕСКАЯ ФОРМА --------------------------- #
def inner_energy(
    u: Callable1 | Callable[[float], float],
    v: Callable1 | Callable[[float], float],
    alpha: float,
    eps: float,
    a_fun: Callable1 | Callable[[float], float],
    quad: QuadLike,
    *,
    dhalf_u: Optional[Callable1] = None,
    dhalf_v: Optional[Callable1] = None,
) -> float:
    r"""
    Вычислить ⟨u, v⟩_{ε,α,a} :=
        ε^α ∫_0^T (D_C^{α/2} u)(x) (D_C^{α/2} v)(x) dx  +  ∫_0^T a(x) u(x) v(x) dx.

    Параметры
    ---------
    u, v : callable
        Функции на [0, T]. Могут быть векторизованы или скалярны.
    alpha : float (0<α<1)
        Порядок Caputo.
    eps : float (>0)
        Малый параметр ε.
    a_fun : callable
        Коэффициент a(x) ≥ a0 > 0 (предположительно).
    quad : Quadrature или (nodes, weights)
        Квадратура на [0, T].
    dhalf_u, dhalf_v : callable или None
        Если заданы — используются как полу-производные. Если None — построятся
        численно по quad.nodes (caputo_interp_callable).

    Возвращает
    ----------
    float
        Значение энергетического скалярного произведения.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    if not (eps > 0.0):
        raise ValueError("eps должно быть > 0")
    x, w = _ensure_quad(quad)
    u_fun = _as_callable1(u)
    v_fun = _as_callable1(v)
    a = _as_callable1(a_fun)

    # Полу-производные
    Du = dhalf_u if dhalf_u is not None else make_dhalf_numeric(u_fun, alpha, (x, w))
    Dv = dhalf_v if dhalf_v is not None else make_dhalf_numeric(v_fun, alpha, (x, w))

    # Интегрируем по квадратуре
    Du_x = Du(x)
    Dv_x = Dv(x)
    with np.errstate(over="ignore", invalid="ignore"):
        energy_vec = Du_x * Dv_x
    if not np.all(np.isfinite(energy_vec)):
        raise FloatingPointError(
            "overflow in energy integrand; check fractional derivatives"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        mass_vec = a(x) * u_fun(x) * v_fun(x)
    if not np.all(np.isfinite(mass_vec)):
        raise FloatingPointError(
            "overflow in mass integrand; check functions or coefficient"
        )

    term_energy = float(np.dot(w, energy_vec))
    term_mass = float(np.dot(w, mass_vec))
    return (eps ** alpha) * term_energy + term_mass


def energy_norm(
    u: Callable1 | Callable[[float], float],
    alpha: float,
    eps: float,
    a_fun: Callable1 | Callable[[float], float],
    quad: QuadLike,
    *,
    dhalf_u: Optional[Callable1] = None,
) -> float:
    r"""
    Норма в ε-энергии:
        ||u||_{ε,α,a}^2 := ε^α ||D_C^{α/2} u||_{L^2}^2 + ||a^{1/2} u||_{L^2}^2.
    """
    return inner_energy(u, u, alpha, eps, a_fun, quad, dhalf_u=dhalf_u, dhalf_v=dhalf_u)


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Базовые проверки на полиномах x^m, для которых D_C^ν x^m = Γ(m+1)/Γ(m+1-ν) x^{m-ν}.
    import mpmath as mp
    from .quadrature import gauss_legendre_on_0T

    T = 1.0
    n = 200
    x, w = gauss_legendre_on_0T(n, T)

    # Тест 1: сравнить inner_energy(u,u) с аналитикой через dhalf (поданным явно) vs численный dhalf
    alpha = 0.6
    eps = 1e-3
    nu = 0.5 * alpha

    # u(x) = x^m, m>=1
    m = 2
    u = lambda t: np.asarray(t, dtype=float) ** m

    # Аналитический D_C^{ν} x^m
    C = float(mp.gamma(m + 1) / mp.gamma(m + 1 - nu))
    dhalf = lambda t: C * (np.asarray(t, dtype=float) ** (m - nu))

    a_fun = lambda t: np.ones_like(np.asarray(t, dtype=float))  # a(x)=1

    # Энергия с явным dhalf
    E_analytic = inner_energy(u, u, alpha, eps, a_fun, (x, w), dhalf_u=dhalf, dhalf_v=dhalf)

    # Энергия с численным dhalf
    E_numeric = inner_energy(u, u, alpha, eps, a_fun, (x, w))

    # Численно должны совпадать с высокой точностью на гладкой функции
    assert abs(E_numeric - E_analytic) / max(1.0, abs(E_analytic)) < 5e-3, "energy: numeric vs analytic mismatch"

    # Тест 2: energy_norm согласуется с inner_energy(u,u)
    En = energy_norm(u, alpha, eps, a_fun, (x, w))
    assert abs(En - E_numeric) < 1e-12, "energy_norm mismatch"

    print("energy.py basic self-tests passed.")