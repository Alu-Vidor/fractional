"""
felab/quadrature.py

Квадратурные формулы на [0, T]:

1) Гаусс–Лежандр (безвесовая, классическая) — для ∫_0^T f(x) dx.
2) Гаусс–Якоби (весовая) — для ∫_{-1}^{1} f(x) (1-x)^α (1+x)^β dx
   и её аффинное отображение на [0, T] (по необходимости).

3) Адаптивная интеграция (рекурсивный Симпсон) на [a, b]
   с допусками abs_tol/rel_tol и ограничением max_subdiv.

4) «Слоистая» интеграция на [0, T] со сплитом [0, c·eps] ∪ [c·eps, T]
   (полезно при наличии тонкого слоя).

Зависимости: numpy, mpmath (для Γ в Якоби).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import mpmath as mp


Array = np.ndarray


# ----------------------------- МАППИНГИ ------------------------------------ #
def _affine_map_to_ab(x: Array, a: float, b: float) -> Array:
    """Отображение узлов x∈[-1,1] → t∈[a,b]."""
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def _affine_scale_weights(w: Array, a: float, b: float) -> Array:
    """Масштабирование весов при аффинном отображении [-1,1]→[a,b]."""
    return 0.5 * (b - a) * w


# ------------------------- ГАУСС–ЛЕЖАНДР (без веса) ------------------------ #
def gauss_legendre(n: int) -> Tuple[Array, Array]:
    """
    Узлы и веса на [-1,1] для ∫ f(x) dx по формуле Гаусса–Лежандра.
    """
    if n <= 0:
        raise ValueError("gauss_legendre: n должно быть > 0")
    # Используем проверенную реализацию из numpy
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x.astype(float), w.astype(float)


def gauss_legendre_on_0T(n: int, T: float) -> Tuple[Array, Array]:
    """
    Узлы и веса на [0, T] для ∫_0^T f(t) dt по Гауссу–Лежандру.
    """
    x, w = gauss_legendre(n)         # на [-1,1]
    t = _affine_map_to_ab(x, 0.0, T) # → [0, T]
    wt = _affine_scale_weights(w, 0.0, T)
    return t, wt


# ------------------------- ГАУСС–ЯКОБИ (с весом) --------------------------- #
def gauss_jacobi(n: int, alpha: float, beta: float) -> Tuple[Array, Array]:
    r"""
    Узлы и веса на [-1,1] для ∫_{-1}^{1} f(x) (1-x)^α (1+x)^β dx.

    Реализация через алгоритм Голуба–Велша (eig симметричной тридиагональной матрицы).
    Рекуррентные коэффициенты для ортонормированных полиномов Якоби:
        a_k = (β² - α²) / ((2k+α+β)(2k+α+β+2))
        b_k = 2 * sqrt( (k+1)(k+α+1)(k+β+1)(k+α+β+1) /
                        ((2k+α+β+1)(2k+α+β+3)(2k+α+β+2)²) )
    но для симметричной тридиагональной матрицы (Jacobi matrix) обычно используют:
        diag_k = (β² - α²) / ((2k+α+β)(2k+α+β+2)),     k=0..n-1
        off_k  = sqrt( k (k+α) (k+β) (k+α+β) /
                        ((2k+α+β)² (2k+α+β+1)(2k+α+β-1)) ),   k=1..n-1
    где diag_k — диагональные элементы J, off_k — под-/наддиагональные (симметрично).

    Веса:
        w_i = C * v_{0i}^2,
    где v_{0i} — первая компонента нормированного собственвектора λ_i,
    а константа C = 2^{α+β+1} Γ(α+1) Γ(β+1) / Γ(α+β+2).

    Возвращает:
        x (узлы), w (веса) — на [-1,1] именно для весовой формулы Якоби.

    ВНИМАНИЕ: эти веса предназначены для интеграла с весом (1-x)^α(1+x)^β.
              Для безвесового интеграла используйте Лежандра.
    """
    if n <= 0:
        raise ValueError("gauss_jacobi: n должно быть > 0")
    if alpha <= -1 or beta <= -1:
        raise ValueError("alpha,beta должны быть > -1")

    k = np.arange(n, dtype=float)
    # Диагональ
    denom = (2 * k + alpha + beta) * (2 * k + alpha + beta + 2.0)
    diag = (beta**2 - alpha**2) / np.where(denom != 0.0, denom, np.inf)
    # Под-/наддиагональ (индексация с 1..n-1)
    kk = np.arange(1, n, dtype=float)
    num = kk * (kk + alpha) * (kk + beta) * (kk + alpha + beta)
    den = (2 * kk + alpha + beta) ** 2 * (2 * kk + alpha + beta + 1.0) * (2 * kk + alpha + beta - 1.0)
    off = np.sqrt(np.where(den != 0.0, num / den, 0.0))

    # Собственные значения/векторы тридиагональной матрицы
    J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    # eigh гарантирует упорядоченные по возрастанию собственные значения
    lam, vecs = np.linalg.eigh(J)
    x = lam.astype(float)

    # Константа для весов
    C = (2.0 ** (alpha + beta + 1.0)) * float(mp.gamma(alpha + 1.0) * mp.gamma(beta + 1.0) / mp.gamma(alpha + beta + 2.0))
    v0 = vecs[0, :]
    w = C * (v0 ** 2)
    return x, w.astype(float)


def gauss_jacobi_on_0T_weighted(n: int, alpha: float, beta: float, T: float) -> Tuple[Array, Array]:
    r"""
    Узлы и «преобразованные» веса для интеграла на [0, T] с якобианским весом:

        ∫_0^T g(t) dt  при замене t = (x+1) T/2,  dt = (T/2) dx
        ∫_0^T g(t) ( (1 - x(t))^α (1 + x(t))^β ) dt   ↔   ∫_{-1}^{1} f(x) (1-x)^α (1+x)^β dx

    На практике эта функция нужна, если вы хотите использовать Гаусс–Якоби
    для интегралов вида ∫_0^T f(t) w_J(t) dt, где w_J(t) = (1 - x(t))^α (1 + x(t))^β,
    x(t) = 2t/T - 1. Для безвесового интеграла используйте Лежандра.
    """
    x, w = gauss_jacobi(n, alpha, beta)
    t = _affine_map_to_ab(x, 0.0, T)
    # Масштаб по dt = (T/2) dx:
    wt = (T / 2.0) * w
    return t, wt


# ------------------------- АДАПТИВНЫЙ СИМПСОН ------------------------------ #
def _simpson(fa: float, fm: float, fb: float, a: float, b: float) -> float:
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0


def _adaptive_simpson(
    f: Callable[[Array], Array] | Callable[[float], float],
    a: float,
    b: float,
    fa: Optional[float],
    fm: Optional[float],
    fb: Optional[float],
    abs_tol: float,
    rel_tol: float,
    max_subdiv: int,
    depth: int,
) -> float:
    if fa is None:
        fa = float(f(a))
    if fb is None:
        fb = float(f(b))
    m = 0.5 * (a + b)
    if fm is None:
        fm = float(f(m))

    I1 = _simpson(fa, fm, fb, a, b)

    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)
    fl = float(f(lm))
    fr = float(f(rm))

    I2_left = _simpson(fa, fl, fm, a, m)
    I2_right = _simpson(fm, fr, fb, m, b)
    I2 = I2_left + I2_right

    err = abs(I2 - I1)
    tol = max(abs_tol, rel_tol * abs(I2))

    if (err < 15.0 * tol) or (depth >= max_subdiv):
        # Рунге-коррекция
        return I2 + (I2 - I1) / 15.0

    # Рекурсивно уточняем
    left = _adaptive_simpson(f, a, m, fa, fl, fm, abs_tol, rel_tol, max_subdiv, depth + 1)
    right = _adaptive_simpson(f, m, b, fm, fr, fb, abs_tol, rel_tol, max_subdiv, depth + 1)
    return left + right


def adaptive_integrate(
    f: Callable[[Array], Array] | Callable[[float], float],
    a: float,
    b: float,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-10,
    max_subdiv: int = 12,
) -> float:
    """
    Рекурсивный адаптивный Симпсон на [a,b].
    Поддерживает как скалярные, так и векторизированные f (используем скалярные вызовы).
    """
    if a == b:
        return 0.0
    if b < a:
        a, b = b, a
    return _adaptive_simpson(f, a, b, None, None, None, abs_tol, rel_tol, max_subdiv, 0)


# ---------------------- «СЛОИСТАЯ» ИНТЕГРАЦИЯ НА [0,T] -------------------- #
def layer_split_integrate(
    f: Callable[[Array], Array] | Callable[[float], float],
    T: float,
    eps: float,
    c: float = 8.0,
    n_layer: int = 120,
    n_bulk: int = 200,
    scheme_layer: Literal["gauss-legendre", "adaptive"] = "gauss-legendre",
    scheme_bulk: Literal["gauss-legendre", "adaptive"] = "gauss-legendre",
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-10,
) -> float:
    """
    Интеграл ∫_0^T f(x) dx, используя разбиение на [0, min(c·eps, T)] и [min(c·eps, T), T].

    Внутри на каждом подинтервале можно выбрать:
      - gauss-legendre (быстро и точно для гладких участков),
      - adaptive (при резких особенностях).
    """
    xR = min(max(c * float(eps), 0.0), float(T))
    # Левая часть (слой)
    if scheme_layer == "gauss-legendre":
        xl, wl = gauss_legendre_on_0T(max(2, n_layer), xR if xR > 0 else 0.0)
        il = float(np.dot(wl, f(xl))) if xR > 0 else 0.0
    elif scheme_layer == "adaptive":
        il = adaptive_integrate(f, 0.0, xR, abs_tol=abs_tol, rel_tol=rel_tol)
    else:
        raise ValueError("Неизвестная схема для слоя")

    # Правая часть (bulk)
    if xR < T:
        if scheme_bulk == "gauss-legendre":
            xr, wr = gauss_legendre_on_0T(max(2, n_bulk), T - xR)
            # Сдвигаем узлы [0, T-xR] → [xR, T]
            xr = xr + xR
            ir = float(np.dot(wr, f(xr)))
        elif scheme_bulk == "adaptive":
            ir = adaptive_integrate(f, xR, T, abs_tol=abs_tol, rel_tol=rel_tol)
        else:
            raise ValueError("Неизвестная схема для bulk")
    else:
        ir = 0.0

    return il + ir


# -------------------------- ВЫСОКОУРОВНЕВЫЕ АПИ --------------------------- #
@dataclass(frozen=True)
class Quadrature:
    """Простая обёртка над узлами/весами."""
    nodes: Array
    weights: Array

    def integrate(self, f: Callable[[Array], Array]) -> float:
        return float(np.dot(self.weights, f(self.nodes)))


def build_quadrature_on_0T(
    scheme: Literal["gauss-legendre", "gauss-jacobi"],
    T: float,
    n: int,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Quadrature:
    """
    Сборщик квадратуры на [0, T].

    ВАЖНО:
    - "gauss-legendre": предназначен для ∫_0^T f(x) dx (без веса).
    - "gauss-jacobi": возвращает узлы/веса, подходящие для ∫_0^T f(x) w_J(x) dx,
      где w_J(x) = (1 - x̂)^α (1 + x̂)^β, x̂ = 2x/T - 1.
      Для безвесового интеграла используйте 'gauss-legendre'.
    """
    if scheme == "gauss-legendre":
        t, w = gauss_legendre_on_0T(n, T)
        return Quadrature(nodes=t, weights=w)
    elif scheme == "gauss-jacobi":
        t, w = gauss_jacobi_on_0T_weighted(n, alpha, beta, T)
        return Quadrature(nodes=t, weights=w)
    else:
        raise ValueError("Неизвестная схема квадратуры")


def integrate_function_on_0T(
    f: Callable[[Array], Array],
    T: float,
    n: int = 200,
    scheme: Literal["gauss-legendre", "adaptive"] = "gauss-legendre",
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-10,
) -> float:
    """
    Удобная функция для ∫_0^T f(x) dx.
    """
    if scheme == "gauss-legendre":
        x, w = gauss_legendre_on_0T(n, T)
        return float(np.dot(w, f(x)))
    elif scheme == "adaptive":
        return adaptive_integrate(f, 0.0, T, abs_tol=abs_tol, rel_tol=rel_tol)
    else:
        raise ValueError("Неизвестная схема")


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # 1) Тест Лежандра: ∫_0^T x^2 dx = T^3/3
    T = 1.7
    x2 = lambda x: x**2
    val_GL = integrate_function_on_0T(x2, T, n=64, scheme="gauss-legendre")
    ref = T**3 / 3.0
    assert abs(val_GL - ref) < 1e-14, "Gauss–Legendre test failed"

    # 2) Тест Якоби с весом: ∫_{-1}^{1} (1-x)^α(1+x)^β dx = 2^{α+β+1} Γ(α+1)Γ(β+1)/Γ(α+β+2)
    alpha, beta = 0.3, 0.7
    n = 64
    x, w = gauss_jacobi(n, alpha, beta)
    val_J = float(np.dot(w, np.ones_like(x)))
    ref_J = (2.0 ** (alpha + beta + 1.0)) * float(mp.gamma(alpha + 1.0) * mp.gamma(beta + 1.0) / mp.gamma(alpha + beta + 2.0))
    assert abs(val_J - ref_J) < 1e-12, "Gauss–Jacobi weight test failed"

    # 3) Адаптивный Симпсон: ∫_0^1 sqrt(x) dx = 2/3
    val_ad = integrate_function_on_0T(np.sqrt, 1.0, scheme="adaptive")
    assert abs(val_ad - (2.0/3.0)) < 1e-10, "Adaptive Simpson test failed"

    # 4) Слоистая интеграция: проверим на гладкой f (должно совпасть с эталоном)
    eps = 1e-3
    val_layer = layer_split_integrate(x2, T=1.0, eps=eps, c=8.0, n_layer=64, n_bulk=64,
                                      scheme_layer="gauss-legendre", scheme_bulk="gauss-legendre")
    assert abs(val_layer - (1.0/3.0)) < 1e-12, "Layer-split integration failed"

    print("quadrature.py basic self-tests passed.")
