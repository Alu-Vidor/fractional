"""
felab/atom.py

Атом начального слоя φ₀ для FELAB и его производные.

Определение:
    φ₀(x; ε) = E_α( - a(0) x^α / ε^α ),   0 < α < 1,
где E_α — функция Миттага–Леффлера.

Ключевые свойства:
- φ₀(0; ε) = 1 (по определению ряда E_α).
- Точная (аналитическая) Caputo-производная порядка α:
      D_C^α φ₀(x; ε) = -(a(0)/ε^α) * φ₀(x; ε).
  Это следует из тождества D_C^α E_α(λ x^α) = λ E_α(λ x^α), 0<α<1.
- Полупроизводная D_C^{α/2} не имеет столь простой замкнутой формы;
  для неё по умолчанию строится устойчивая численная аппроксимация
  (product-integration на узлах, с линейной интерполяцией).

API:
- phi0_factory(a0, eps, alpha) -> Phi0
    Возвращает объект с полями:
      .alpha, .eps, .a0, .lam, .__call__(x)=φ₀(x), .dalpha(x), .dhalf(nodes)->callable
- phi0(a0, eps, alpha) -> callable
    Короткий синоним: вернуть функцию x↦φ₀(x;ε).
- dcaputo_alpha_of_phi0(a0, eps, alpha) -> callable
    Возвращает аналитическую D_C^α φ₀(x; ε).
- dcaputo_half_numeric(phi0_fun, alpha, quad_or_nodes) -> callable
    Строит численный интерполятор D_C^{α/2} φ₀(x; ε) по узлам.

Зависимости: numpy, felab.mlf (E), felab.fracops (caputo_interp_callable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .mlf import E as _Ealpha, Eab as _Eab
from .fracops import caputo_interp_callable
from .quadrature import Quadrature


Array = np.ndarray
QuadLike = Union[Quadrature, Tuple[Array, Array]]


# ------------------------------- ОСНОВНОЕ ---------------------------------- #
@dataclass(frozen=True)
class Phi0:
    """
    Контейнер атома начального слоя.

    Свойства:
      alpha ∈ (0,1), eps > 0, a0 > 0,
      lam = -a0/eps^α (скалирование аргумента E_α).
    Методы:
      __call__(x)        — φ₀(x;ε)
      dalpha(x)          — D_C^α φ₀(x;ε) (аналитически)
      dhalf(quad|nodes)  — построить численный интерполятор D_C^{α/2} φ₀(x;ε)
    """
    alpha: float
    eps: float
    a0: float

    @property
    def lam(self) -> float:
        return -float(self.a0) / (float(self.eps) ** float(self.alpha))

    # --- значения φ₀ ---
    def __call__(self, x: Array | float) -> Array:
        """
        Возвращает φ₀(x;ε) = E_α(lam * x^α). Сохраняет действительность для x≥0.
        """
        xv = np.asarray(x, dtype=float)
        # Для x=0: E_α(0)=1
        z = self.lam * np.power(np.clip(xv, 0.0, np.inf), self.alpha, where=(xv >= 0), out=np.zeros_like(xv))
        val = _Ealpha(self.alpha, z)
        # _Ealpha возвращает ndarray того же shape; гарантируем float для скаляра
        if np.isscalar(x):
            return float(val)  # type: ignore
        return np.asarray(val, dtype=float)

    # --- точная D_C^α φ₀ ---
    def dalpha(self, x: Array | float) -> Array:
        """
        Аналитическая Caputo-производная порядка α:
            D_C^α φ₀(x) = lam * φ₀(x)  (где lam = -a0/eps^α)
        """
        return self.lam * self(x)

    # --- численная D_C^{α/2} φ₀ ---
    def dhalf(self, quad: QuadLike | Array) -> Callable[[Array | float], Array]:
        """
        Построить численный интерполятор D_C^{α/2} φ₀(x) по узлам квадратуры.

        Параметры
        ---------
        quad : Quadrature или (nodes,weights) или просто nodes (1D массив).
            Узлы, на которых строится product-integration и далее линейная интерполяция.

        Возвращает
        ---------
        callable
            Функция t ↦ D_C^{α/2} φ₀(t).
        """
        if isinstance(quad, Quadrature):
            nodes = quad.nodes
        elif isinstance(quad, tuple):
            nodes = np.asarray(quad[0], dtype=float)
        else:
            nodes = np.asarray(quad, dtype=float)
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("dhalf: ожидается одномерный массив узлов длины ≥ 2")

        # замкнуть φ₀ для caputo_interp_callable
        def _phi(x: Array) -> Array:
            return self(x)

        return caputo_interp_callable(_phi, nodes, nu=0.5 * float(self.alpha))


# ----------------------------- ФАБРИКИ УДОБСТВА ---------------------------- #
def phi0_factory(a0: float, eps: float, alpha: float) -> Phi0:
    """
    Собрать объект атома φ₀ с валидацией параметров.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    if eps <= 0.0:
        raise ValueError("eps должно быть > 0")
    if a0 <= 0.0:
        raise ValueError("a0 должно быть > 0")
    return Phi0(alpha=float(alpha), eps=float(eps), a0=float(a0))


def phi0(a0: float, eps: float, alpha: float) -> Callable[[Array | float], Array]:
    """
    Краткий синоним: вернуть функцию x ↦ φ₀(x;ε).
    """
    obj = phi0_factory(a0, eps, alpha)
    return lambda x: obj(x)


def dcaputo_alpha_of_phi0(a0: float, eps: float, alpha: float) -> Callable[[Array | float], Array]:
    """
    Вернуть аналитическую D_C^α φ₀(x;ε).
    """
    obj = phi0_factory(a0, eps, alpha)
    return lambda x: obj.dalpha(x)


def dcaputo_half_numeric(
    phi0_fun: Callable[[Array], Array],
    alpha: float,
    quad: QuadLike | Array,
) -> Callable[[Array | float], Array]:
    """
    Построить численный интерполятор D_C^{α/2} заданной функции φ₀ (не обязательно фабричной).
    """
    if isinstance(quad, Quadrature):
        nodes = quad.nodes
    elif isinstance(quad, tuple):
        nodes = np.asarray(quad[0], dtype=float)
    else:
        nodes = np.asarray(quad, dtype=float)
    return caputo_interp_callable(phi0_fun, nodes, nu=0.5 * float(alpha))


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    import mpmath as mp
    from .quadrature import gauss_legendre_on_0T
    from .fracops import caputo_pi_on_grid

    # Параметры
    alpha = 0.6
    eps = 1e-3
    a0 = 2.0
    T = 1.0

    p = phi0_factory(a0=a0, eps=eps, alpha=alpha)

    # 1) Проверка значений в нуле и знакопостоянности (φ₀ >= 0 для λ<0 и x≥0)
    x = np.linspace(0.0, T, 11)
    vals = p(x)
    assert abs(p(0.0) - 1.0) < 1e-15, "phi0(0) ≠ 1"
    assert np.all(vals >= -1e-14), "phi0 должна быть неотрицательной на [0,T] для λ<0 (численно)"

    # 2) Проверка тождества D_C^α φ₀ = lam * φ₀ численно на узлах
    nodes, _ = gauss_legendre_on_0T(300, T)
    # Численная Caputo D^α
    Dnum = caputo_pi_on_grid(nodes, p(nodes), nu=alpha)
    # Аналитика
    Dan = p.lam * p(nodes)
    # Сравнение (кроме x=0)
    mask = nodes > 1e-12
    rel = np.max(np.abs(Dnum[mask] - Dan[mask]) / np.maximum(1.0, np.abs(Dan[mask])))
    assert rel < 5e-3, f"Mismatch in D_C^α φ0 (rel={rel:g})"

    # 3) dhalf: построение интерполятора и базовая проверка непротиворечивости
    Dhalf = p.dhalf(nodes)
    v_mid = Dhalf(T * 0.37)  # просто вызов без точного эталона
    assert np.isfinite(v_mid), "D_C^{α/2} φ0 должна вычисляться"

    print("atom.py basic self-tests passed.")