"""
felab/assemble.py

Сборка линейной системы для FELAB в энергетическом базисе 𝔅_N(ε) = {φ₀, ψ₁,…,ψ_N}.

Уравнение:
    A(u, v) = (f, v)_{L^2(0,T)}   для всех v ∈ span{ψ_k},
где
    A(u, v) = ε^α ( D_C^{α/2} u, D_C^{α/2} v ) + (a u, v).

Представление решения:
    u_N(x) = u0 * φ₀(x; ε) + Σ_{k=1}^N c_k ψ_k(x).

Линейная система на коэффициенты c = (c_1,…,c_N)^T:
    K c = b,
где
    K_{ij} = A(ψ_j, ψ_i),
    b_i    = (f, ψ_i) - u0 * A(φ₀, ψ_i).

Публичное API:
- assemble_system(basis, f_fun, u0=0.0) -> (K, b, extras)
- l2_inner(basis, u, v) → (u, v)_{L^2}
- evaluate_solution(basis, coeffs, u0, x) → значения u_N(x)
- build_solution_callable(basis, coeffs, u0) → callable x↦u_N(x)

Зависимости: numpy, felab.basis.Basis (basis.inner уже частично применённая ε-энергия),
             felab.quadrature.Quadrature для L2-интегралов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import numpy as np

from .basis import Basis
from .quadrature import Quadrature


Array = np.ndarray
Func = Callable[[Array | float], Array]


# ------------------------------ ВСПОМОГАТЕЛЬНО ----------------------------- #
def _as_callable_vec(f: Callable[[float], float] | Callable[[Array], Array]) -> Callable[[Array], Array]:
    def g(x: Array) -> Array:
        xv = np.asarray(x, dtype=float)
        out = f(xv)  # type: ignore
        if np.isscalar(out):
            return np.full_like(xv, float(out))
        return np.asarray(out, dtype=float)
    return g


def l2_inner(
    basis: Basis,
    u: Func | Callable[[float], float],
    v: Optional[Func | Callable[[float], float]] = None,
) -> float:
    """
    Вычислить L2-скалярное произведение на [0,T] с использованием квадратуры basis.quad:
        (u, v) = ∫_0^T u(x) v(x) dx.
    Если v=None, возвращает ∫_0^T u(x) dx.
    """
    quad: Quadrature = basis.quad
    x, w = quad.nodes, quad.weights
    uu = _as_callable_vec(u)(x)
    if v is None:
        return float(np.dot(w, uu))
    vv = _as_callable_vec(v)(x)
    return float(np.dot(w, uu * vv))


# ------------------------------ СБОРКА K, b -------------------------------- #
@dataclass(frozen=True)
class AssemblyExtras:
    """
    Вспомогательная диагностическая информация:
      - F: вектор F_i = (f, ψ_i)_{L2}
      - G: вектор G_i = A(φ₀, ψ_i)
      - SPD_hint: эвристика, что K симм. положительно определена
    """
    F: Array
    G: Array
    SPD_hint: bool = True


def assemble_system(
    basis: Basis,
    f_fun: Callable[[float], float] | Callable[[Array], Array],
    u0: float = 0.0,
) -> Tuple[Array, Array, AssemblyExtras]:
    """
    Сборка матрицы жёсткости K и правой части b для системы K c = b.

    Параметры
    ---------
    basis : Basis
        Энергетический базис {φ₀, ψ_k} и частично применённая форма A(·,·).
    f_fun : callable
        Правая часть f(x).
    u0 : float
        Начальное значение u(0) = u0 (влияет только на правую часть через A(φ₀, ·)).

    Возвращает
    ----------
    (K, b, extras) :
        K : ndarray (N×N)
        b : ndarray (N,)
        extras : AssemblyExtras с векторами F и G.
    """
    psi = basis.psi
    N = len(psi)
    if N == 0:
        raise ValueError("В базисе нет функций ψ (N=0). Увеличьте N при построении basis.")

    # --- Матрица жёсткости K_{ij} = A(ψ_j, ψ_i) ---
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        qi = psi[i]
        # Используем симметрию, чтобы заполнять только треугольник
        for j in range(i, N):
            qj = psi[j]
            val = basis.inner(qj, qi)  # A(qj, qi)
            K[i, j] = val
            K[j, i] = val

    # --- Вектор F_i = (f, ψ_i)_{L2} ---
    F = np.empty(N, dtype=float)
    for i in range(N):
        F[i] = l2_inner(basis, f_fun, psi[i])

    # --- Вектор G_i = A(φ₀, ψ_i) ---
    G = np.empty(N, dtype=float)
    for i in range(N):
        G[i] = basis.inner(basis.phi0, psi[i])

    # --- Правая часть b = F - u0 * G ---
    b = F - float(u0) * G

    extras = AssemblyExtras(F=F, G=G, SPD_hint=True)
    return K, b, extras


# -------------------------- ОЦЕНКА/ВОССТАНОВЛЕНИЕ -------------------------- #
def build_solution_callable(
    basis: Basis,
    coeffs: Array,
    u0: float = 0.0,
) -> Func:
    """
    Сформировать функцию x ↦ u_N(x) = u0 φ₀(x) + Σ c_k ψ_k(x).
    """
    psi = basis.psi
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size != len(psi):
        raise ValueError(f"Размер coeffs ({c.size}) не совпадает с числом базисных функций ({len(psi)}).")

    def uN(x: Array | float) -> Array:
        xv = np.asarray(x, dtype=float)
        acc = basis.phi0(xv) * float(u0)
        for ck, qk in zip(c, psi):
            if ck != 0.0:
                acc = acc + ck * np.asarray(qk(xv), dtype=float)
        if np.isscalar(x):
            return float(acc)  # type: ignore
        return acc

    return uN


def evaluate_solution(
    basis: Basis,
    coeffs: Array,
    u0: float,
    x: Array | float,
) -> Array:
    """
    Вычислить u_N(x) на заданных точках x.
    """
    uN = build_solution_callable(basis, coeffs, u0)
    return uN(x)


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Мини-проверка сборки на синтетике: a(x)=1, f подобрано так, чтобы
    # целевое решение имело вид u = u0*φ0 + гладкая часть (полином).
    from .basis import build_basis

    T = 1.0
    alpha = 0.6
    eps = 1e-3
    a_fun = lambda x: 1.0
    u0 = 1.2

    # Строим базис
    B = build_basis(N=12, alpha=alpha, eps=eps, a_fun=a_fun, T=T, raw_family="chebyshev", quad_n=256)

    # Возьмём «истинное» решение: u_true = u0*φ0 + p(x), p(x)=x + 0.3 x^2
    def u_true(x):
        xv = np.asarray(x, dtype=float)
        return u0 * B.phi0(xv) + xv + 0.3 * xv**2

    # Вычислим f = L u_true = ε^α D_C^α u_true + a(x) u_true.
    # Для φ0 используется аналитика D_C^α φ0 = lam φ0, а для полинома — формула Капуто.
    import mpmath as mp
    lam = -B.a0 / (eps ** alpha)
    def Dcap_alpha_poly(x):
        # D_C^α (x) = Γ(2)/Γ(2-α) x^{1-α}
        # D_C^α (x^2) = Γ(3)/Γ(3-α) x^{2-α}
        xv = np.asarray(x, dtype=float)
        return (mp.gamma(2)/mp.gamma(2-alpha)) * xv**(1-alpha) + \
               0.3 * (mp.gamma(3)/mp.gamma(3-alpha)) * xv**(2-alpha)

    def f_fun(x):
        xv = np.asarray(x, dtype=float)
        return (eps**alpha) * (lam * B.phi0(xv) + Dcap_alpha_poly(xv)) + u_true(xv)

    # Собираем систему и решаем
    K, b, extras = assemble_system(B, f_fun, u0=u0)
    c = np.linalg.solve(K, b)

    # Оцениваем ошибку аппроксимации на сетке
    X = B.grid(400)
    err = np.max(np.abs(evaluate_solution(B, c, u0, X) - u_true(X)))
    # Для умеренного N ошибка должна быть небольшой
    assert err < 2e-2, f"Слишком большая погрешность сборки/решения: {err:.3e}"

    print("assemble.py basic self-tests passed.")