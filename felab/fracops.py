"""
felab/fracops.py

Дробные операторы (Caputo и Римана–Лиувилля) для 0<ν<1 на [0, T].

В этом модуле реализованы две ветви вычислений:

1) ЧИСЛЕННАЯ (продукт-интегрирование на неравномерной сетке)
   * Caputo D_C^ν u(x_j) по табличным значениям u(t_k) на узлах 0=t_0<...<t_M=T.
     Формула использует кусочно-линейную аппроксимацию u и точные интегралы
     ядра (x_j - t)^{-ν} на каждом отрезке:
         D_C^ν u(x_j) ≈ (1/Γ(1-ν)) * Σ_{k=1}^j (Δu_k / h_k) *
                        [ (x_j - t_{k-1})^{1-ν} - (x_j - t_k)^{1-ν} ] / (1-ν)
     где Δu_k = u(t_k)-u(t_{k-1}), h_k = t_k - t_{k-1}.
     Эта схема корректна на НЕРАВНОМЕРНЫХ сетках, устойчива и проста.

   * Риман–Лиувилль интеграл I^μ u(x_j) (0<μ<1) по табличным значениям u(t_k)
     с кусочно-постоянной аппроксимацией на каждом отрезке:
         I^μ u(x_j) ≈ (1/Γ(μ)) * Σ_{k=1}^j u(t_k^*) *
                       [ (x_j - t_{k-1})^{μ} - (x_j - t_k)^{μ} ] / μ
     где t_k^* — левая точка или середина отрезка (по выбору).

2) ОБЁРТКИ ДЛЯ ПОЛЬЗОВАТЕЛЯ
   * Вычисление D_C^ν на всей сетке: caputo_pi_on_grid(...)
   * Построение callable-приближения D_C^ν u(x) через линейную интерполяцию
     по узлам (для внешнего кода): caputo_interp_callable(...)

Для базисов FELAB:
- Полиномы-«сырьё» p_k и их D_C^ν вычисляются аналитически в poly.py.
- Для спецфункции φ₀ (Mittag–Leffler) удобнее пользоваться аналитикой
  непосредственно в модуле atom.py (см. комментарии там).

Зависимости: numpy, mpmath (только для Γ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import mpmath as mp


Array = np.ndarray


# ----------------------------- УТИЛИТЫ ------------------------------------- #
def _require_monotone_nodes(nodes: Array) -> None:
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("nodes должен быть одномерным массивом длины ≥ 2")
    if not np.all(np.isfinite(x)):
        raise ValueError("nodes содержит NaN/Inf")
    if not (x[0] >= 0.0):
        raise ValueError("nodes: первый узел должен быть ≥ 0")
    if not np.all(np.diff(x) > 0):
        raise ValueError("nodes должен строго возрастать (t_0 < t_1 < ... < t_M)")


def _gamma(x: float) -> float:
    return float(mp.gamma(x))


# ------------------------ CAPUTO: ПРОДУКТ-ИНТЕГРИРОВАНИЕ ------------------- #
def caputo_pi_on_grid(
    nodes: Array,
    values: Array,
    nu: float,
) -> Array:
    """
    Численное D_C^ν u(x) для 0<ν<1 на узлах nodes, используя значения u(nodes).

    Реализация: продукт-интегрирование по формуле
        D_C^ν u(x_j) ≈ (1/Γ(1-ν)) * Σ_{k=1}^j (Δu_k / h_k) *
                        [ (x_j - t_{k-1})^{1-ν} - (x_j - t_k)^{1-ν} ] / (1-ν)

    Параметры
    ---------
    nodes : ndarray, shape (M+1,)
        Узлы 0=t_0 < t_1 < ... < t_M = T (строго возрастающие).
    values : ndarray, shape (M+1,)
        Значения u(t_k) в узлах nodes.
    nu : float, 0<ν<1
        Порядок дробной производной Капуто.

    Возвращает
    ----------
    D : ndarray, shape (M+1,)
        Приближения D_C^ν u(nodes[j]). Первое значение D[0] = 0.

    Замечания
    ---------
    * Схема корректна на неравномерных сетках.
    * На узлах очень близких к нулю возможна усиленная чувствительность;
      при необходимости сгущайте сетку около 0.
    """
    if not (0.0 < nu < 1.0):
        raise ValueError("Порядок ν должен быть в (0,1).")
    _require_monotone_nodes(nodes)
    t = np.asarray(nodes, dtype=float)
    u = np.asarray(values, dtype=float)
    if u.shape != t.shape:
        raise ValueError("values и nodes должны иметь одинаковую форму")

    M = t.size - 1
    D = np.zeros_like(t)
    inv_gamma = 1.0 / _gamma(1.0 - nu)
    inv_1mnu = 1.0 / (1.0 - nu)

    # Предвычислим шаги и разности
    h = np.diff(t)                      # shape (M,)
    du = np.diff(u)                     # shape (M,)
    ratio = du / h                      # shape (M,)

    # Внешняя петля по j (узел x_j)
    # Можно ускорить O(M^2) → O(M log M) через свёртки на равномерных сетках; здесь — общая неравномерная версия.
    for j in range(1, M + 1):
        xj = t[j]
        # Интегральные вклады по отрезкам [t_{k-1}, t_k], k=1..j
        # ΔI_k(x_j) = ((x_j - t_{k-1})^{1-ν} - (x_j - t_k)^{1-ν}) / (1-ν)
        # Аккуратно обращаемся со степенями при x_j == t_k (возникает 0^{1-ν} = 0).
        left = xj - t[:j]          # x_j - t_{k-1}, shape (j,)
        right = xj - t[1:j+1]      # x_j - t_k,     shape (j,)
        # отрицательных нет, т.к. t_k ≤ x_j
        pow_left = np.power(left, 1.0 - nu, where=(left > 0), out=np.zeros_like(left))
        pow_right = np.power(right, 1.0 - nu, where=(right > 0), out=np.zeros_like(right))
        dI = (pow_left - pow_right) * inv_1mnu  # shape (j,)
        D[j] = inv_gamma * np.dot(ratio[:j], dI)

    # По определению Caputo, D_C^ν u(0) = 0 для u(0) конечного
    D[0] = 0.0
    return D


# ------------------- РИМАН–ЛИУВИЛЛЬ: ФРАКЦ. ИНТЕГРАЛ ---------------------- #
def rl_integral_on_grid(
    nodes: Array,
    values: Array,
    mu: float,
    sample: Literal["left", "mid"] = "left",
) -> Array:
    """
    Численное I^μ u(x) для 0<μ<1 на узлах nodes, используя значения u(nodes).

    Реализация: продукт-интегрирование с кусочно-постоянной аппроксимацией u
    на каждом отрезке (левая точка или середина).
        I^μ u(x_j) ≈ (1/Γ(μ)) * Σ_{k=1}^j u_k^* *
                     [ (x_j - t_{k-1})^{μ} - (x_j - t_k)^{μ} ] / μ

    Параметры
    ---------
    nodes : ndarray
        Узлы 0=t_0 < ... < t_M.
    values : ndarray
        Значения u(t_k).
    mu : float, 0<μ<1
        Порядок интеграла Римана–Лиувилля.
    sample : {"left","mid"}
        Где брать значение u на отрезке: левая точка (по умолчанию) или середина.

    Возвращает
    ----------
    I : ndarray
        Значения I^μ u(x_j).
    """
    if not (0.0 < mu < 1.0):
        raise ValueError("Порядок μ должен быть в (0,1).")
    _require_monotone_nodes(nodes)
    t = np.asarray(nodes, dtype=float)
    u = np.asarray(values, dtype=float)
    if u.shape != t.shape:
        raise ValueError("values и nodes должны иметь одинаковую форму")

    M = t.size - 1
    I = np.zeros_like(t)
    inv_gamma = 1.0 / _gamma(mu)
    inv_mu = 1.0 / mu
    h = np.diff(t)

    # Значение u на отрезке
    if sample == "left":
        u_seg = u[:-1]  # u(t_{k-1})
    elif sample == "mid":
        # Интерполяция на середину (кусочно-линейная)
        u_seg = 0.5 * (u[:-1] + u[1:])
    else:
        raise ValueError("sample должен быть 'left' или 'mid'")

    for j in range(1, M + 1):
        xj = t[j]
        left = xj - t[:j]          # x_j - t_{k-1}
        right = xj - t[1:j+1]      # x_j - t_k
        pow_left = np.power(left, mu, where=(left > 0), out=np.zeros_like(left))
        pow_right = np.power(right, mu, where=(right > 0), out=np.zeros_like(right))
        dW = (pow_left - pow_right) * inv_mu      # shape (j,)
        I[j] = inv_gamma * np.dot(u_seg[:j], dW)

    I[0] = 0.0
    return I


# --------------------- ОБЁРТКА: CALLABLE ДЛЯ D_C^ν u(x) -------------------- #
@dataclass
class CaputoPI:
    """
    Обёртка для удобного вычисления D_C^ν u(x) на лету.

    Строится по:
      - узлам nodes (растущим),
      - табличным значениям u(nodes),
      - порядку ν∈(0,1).

    Затем:
      - значения D_C^ν у узлах берутся из caputo_pi_on_grid,
      - для промежуточных x используется линейная интерполяция.
    """
    nodes: Array
    values: Array
    nu: float
    _D_nodes: Array

    @staticmethod
    def build(nodes: Array, values: Array, nu: float) -> "CaputoPI":
        D = caputo_pi_on_grid(nodes, values, nu)
        return CaputoPI(nodes=np.asarray(nodes, dtype=float),
                        values=np.asarray(values, dtype=float),
                        nu=float(nu),
                        _D_nodes=D)

    def __call__(self, x: Array | float) -> Array:
        """
        Линейная интерполяция значений D_C^ν u(x) на [0, T].
        Вне [0,T] значения не определены (выбрасывается исключение).
        """
        t = self.nodes
        D = self._D_nodes
        if np.isscalar(x):
            xv = float(x)
            if xv < t[0] - 1e-15 or xv > t[-1] + 1e-15:
                raise ValueError("x вне диапазона [0, T]")
            return float(np.interp(xv, t, D))
        xx = np.asarray(x, dtype=float)
        if (xx.min() < t[0] - 1e-15) or (xx.max() > t[-1] + 1e-15):
            raise ValueError("x вне диапазона [0, T]")
        return np.interp(xx, t, D)


def caputo_interp_callable(
    u_fun: Callable[[Array], Array],
    nodes: Array,
    nu: float,
) -> CaputoPI:
    """
    Удобная фабрика:
      - семплирует u_fun на узлах nodes,
      - строит CaputoPI для последующей интерполяции D_C^ν u(x).

    Пример:
        mesh = np.linspace(0, 1, 200)
        u = lambda x: x**2
        D = caputo_interp_callable(u, mesh, nu=0.4)
        y = D(np.linspace(0,1,1000))
    """
    _require_monotone_nodes(nodes)
    vals = u_fun(np.asarray(nodes, dtype=float))
    return CaputoPI.build(nodes, vals, nu)


# -------------------------- ПОЛУ-ПРОИЗВОДНАЯ α/2 --------------------------- #
def caputo_half_from_alpha(
    nodes: Array,
    values: Array,
    alpha: float,
) -> Array:
    """
    Удобная обёртка для FELAB: D_C^{α/2} u на узлах (0<α<1).

    Эквивалентно:
        caputo_pi_on_grid(nodes, values, nu=alpha/2)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    return caputo_pi_on_grid(nodes, values, nu=0.5 * float(alpha))


# ------------------------------ САМООТЛАДКА -------------------------------- #
if __name__ == "__main__":
    # Простые проверки корректности
    import numpy as _np

    T = 1.0
    nodes = _np.linspace(0.0, T, 201)
    # Тестовая функция: u(x) = x  ⇒ D_C^ν u(x) = Γ(2)/Γ(2-ν) * x^{1-ν}
    nu = 0.4
    u_vals = nodes.copy()
    D_num = caputo_pi_on_grid(nodes, u_vals, nu)
    D_ref = (mp.gamma(2)/mp.gamma(2-nu)) * (nodes ** (1-nu))
    # Вблизи 0 сравнение бессмысленно (обнуляем первую точку)
    assert _np.allclose(D_num[1:], D_ref[1:], rtol=2e-3, atol=2e-6), "Caputo PI test failed"

    # РЛ-интеграл: I^μ 1 = x^{μ}/Γ(μ+1)
    mu = 0.3
    ones = _np.ones_like(nodes)
    I_num = rl_integral_on_grid(nodes, ones, mu, sample="left")
    I_ref = (nodes ** mu) / mp.gamma(mu + 1.0)
    assert _np.allclose(I_num[1:], I_ref[1:], rtol=2e-3, atol=2e-6), "RL integral test failed"

    print("fracops.py basic self-tests passed.")