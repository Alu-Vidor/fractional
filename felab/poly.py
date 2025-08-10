"""
felab/poly.py

Сырьевые базисные функции p_k(x) и их дробные производные (Капуто).
По умолчанию используем семейство:
    p_k(x) = x * T_k( 2x/T - 1 ),   k = 0,1,2,...
где T_k — полином Чебышёва I рода. Фактор x гарантирует p_k(0)=0.

Также доступен вариант на базе полиномов Лежандра (Jacobi(0,0)):
    p_k(x) = x * P_k( 2x/T - 1 )

Основные возможности:
- генерация мономиальных коэффициентов p_k(x) = Σ_{j=0}^{deg} c_j x^j
  (c_0 = 0 по построению);
- быстрое вычисление значений p_k(x) по этим коэффициентам;
- вычисление дробной производной Капуто порядка ν∈(0,1):
      D_C^ν x^m = Γ(m+1)/Γ(m+1-ν) * x^{m-ν},   для m≥1
  и D_C^ν(1) = 0.
- построение callable-функций для p_k и D_C^ν p_k.

Зависимости: numpy, mpmath (для Γ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import Legendre


Array = np.ndarray


# ----------------------------- ВСПОМОГАТЕЛЬНОЕ ----------------------------- #
def _binom(n: int, k: int) -> float:
    """Биномиальный коэффициент (вещественный)."""
    if k < 0 or k > n:
        return 0.0
    # быстрая и точная реализация для целых n
    from math import comb
    return float(comb(n, k))


def _compose_y_poly_to_x_coeffs(a: Array, T: float) -> Array:
    """
    Преобразовать полином по y в полином по x, где y = 2x/T - 1.
    На входе a[n] — коэффициент при y^n.
    Возвращает b[j] — коэффициент при x^j.
    """
    a = np.asarray(a, dtype=float)
    deg_y = a.size - 1
    # Максимальная степень по x после подстановки (ax+b)^n — n
    b = np.zeros(deg_y + 1, dtype=float)
    # y^n = sum_{j=0}^n C(n,j) (2/T)^j x^j * (-1)^{n-j}
    for n in range(deg_y + 1):
        an = a[n]
        if an == 0.0:
            continue
        for j in range(n + 1):
            b[j] += an * _binom(n, j) * (2.0 / T) ** j * ((-1.0) ** (n - j))
    return b


def _multiply_by_x(coeffs: Array) -> Array:
    """Умножить полином на x: сдвиг коэффициентов на 1 степень."""
    coeffs = np.asarray(coeffs, dtype=float)
    out = np.zeros(coeffs.size + 1, dtype=float)
    out[1:] = coeffs
    return out


def eval_poly_coeffs(coeffs: Array, x: Array | float) -> Array:
    """
    Вычислить Σ c_j x^j для массива/скаляра x (стабильная схема Горнера).
    """
    # Используем numpy.polynomial.Polynomial
    p = Poly(coeffs)
    return p(x)


# ------------------------------ СЫРЬЕВЫЕ p_k ------------------------------- #
def chebyshev_raw_coeffs(k: int, T: float) -> Array:
    """
    Вернуть мономиальные коэффициенты p_k(x) = x * T_k(2x/T - 1)
    в виде массива c[0..k+1], где p_k(x)=Σ c_j x^j и c_0=0.
    """
    # T_k(y) в базисе Чебышёва
    c_cheb = np.zeros(k + 1, dtype=float)
    c_cheb[-1] = 1.0
    Tk = Chebyshev(c_cheb, domain=[-1, 1])
    # Переводим в обычный полином по y: Tk(y) = Σ a_n y^n
    Tk_poly: Poly = Tk.convert(kind=Poly)  # type: ignore
    a = Tk_poly.coef  # коэффициенты по y
    # Подставляем y = 2x/T - 1 → получаем coeffs по x
    b = _compose_y_poly_to_x_coeffs(a, T)
    # Умножаем на x
    coeffs = _multiply_by_x(b)
    # Нормируем численно очень малые значения к нулю
    coeffs[np.isclose(coeffs, 0.0, atol=1e-16, rtol=0.0)] = 0.0
    return coeffs


def legendre_raw_coeffs(k: int, T: float) -> Array:
    """
    Вернуть мономиальные коэффициенты p_k(x) = x * P_k(2x/T - 1),
    где P_k — полином Лежандра (Jacobi(0,0)).
    """
    c_leg = np.zeros(k + 1, dtype=float)
    c_leg[-1] = 1.0
    Lk = Legendre(c_leg, domain=[-1, 1])
    Lk_poly: Poly = Lk.convert(kind=Poly)  # type: ignore
    a = Lk_poly.coef  # коэффициенты по y
    b = _compose_y_poly_to_x_coeffs(a, T)
    coeffs = _multiply_by_x(b)
    coeffs[np.isclose(coeffs, 0.0, atol=1e-16, rtol=0.0)] = 0.0
    return coeffs


def raw_coeffs(k: int, T: float, family: Literal["chebyshev", "jacobi"]) -> Array:
    """
    Унифицированный доступ к коэффициентам p_k.
    family="chebyshev" → x*T_k(2x/T-1)
    family="jacobi"    → x*P_k(2x/T-1) (Legendre P_k)
    """
    if family == "chebyshev":
        return chebyshev_raw_coeffs(k, T)
    elif family == "jacobi":
        return legendre_raw_coeffs(k, T)
    else:
        raise ValueError(f"Неизвестное семейство raw_family='{family}'")


# --------------------- ДРОБНАЯ ПРОИЗВОДНАЯ КАПУТО -------------------------- #
def caputo_frac_deriv_eval(coeffs: Array, nu: float, x: Array | float) -> Array:
    """
    Вычислить D_C^ν p(x), где p(x)=Σ_{j=0}^M c_j x^j и 0<ν<1.

    Формула (для m≥1):
      D_C^ν x^m = Γ(m+1)/Γ(m+1-ν) * x^{m-ν}
    и D_C^ν(1) = 0.

    ВАЖНО: при x=0 задаём 0**a := 0 для a>0 и 1 для a≈0 (с допуском).
    """
    if not (0.0 < nu < 1.0):
        raise ValueError("Порядок ν должен быть в (0,1) для Caputo в данной реализации.")

    x_arr = np.asarray(x, dtype=float)
    # Начинаем с нулевого массива результата
    res = np.zeros_like(x_arr, dtype=float)

    # m = степень монома. c_0* x^0 → производная = 0 (Caputo)
    M = len(coeffs) - 1
    # Предварительно вычислим гамма-коэффициенты
    gammas = [float(mp.gamma(m + 1) / mp.gamma(m + 1 - nu)) if m >= 1 else 0.0 for m in range(M + 1)]

    # Обработка x==0: для каждой степени m создаёт вклад ~ x^{m-ν}
    # Если m-ν > 0 → вклад в 0-ой точке нулевой; если ≈0 → 1.
    # Для векторной реализации: вычисляем через обычную степень с защитой.
    # Численно безопасно: np.where(x>0, x**(m-nu), (1.0 if |m-ν|<tol else 0.0))
    tol0 = 1e-14

    # Суммируем вклады по m≥1
    for m in range(1, M + 1):
        c = coeffs[m]
        if c == 0.0:
            continue
        pow_exp = m - nu
        contrib = np.where(
            x_arr > 0.0,
            (x_arr ** pow_exp),
            (1.0 if abs(pow_exp) < tol0 else 0.0)
        )
        res = res + (c * gammas[m]) * contrib

    # Если вход был скаляром — вернуть скаляр
    if np.isscalar(x):
        return float(res)  # type: ignore
    return res


def caputo_frac_deriv_callable(coeffs: Array, nu: float) -> Callable[[Array | float], Array]:
    """
    Вернуть функцию x ↦ D_C^ν p(x) для p по мономиальным коэффициентам.
    """
    def _f(x: Array | float) -> Array:
        return caputo_frac_deriv_eval(coeffs, nu, x)
    return _f


# ------------------------- УДОБНЫЕ КОНСТРУКТОРЫ ---------------------------- #
@dataclass
class RawPoly:
    """
    Контейнер сырьевого базиса:
      - coeffs: мономиальные коэффициенты p_k(x)
      - eval(x): значения p_k(x)
      - dcaputo(nu)(x): значения D_C^ν p_k(x)
    """
    k: int
    T: float
    family: Literal["chebyshev", "jacobi"]
    coeffs: Array

    def eval(self, x: Array | float) -> Array:
        return eval_poly_coeffs(self.coeffs, x)

    def dcaputo(self, nu: float) -> Callable[[Array | float], Array]:
        return caputo_frac_deriv_callable(self.coeffs, nu)


def build_raw_family(N: int,
                     T: float,
                     family: Literal["chebyshev", "jacobi"] = "chebyshev") -> List[RawPoly]:
    """
    Построить список сырьевых функций p_k, k=0..N-1, для выбранного семейства.
    Возвращает список RawPoly с коэффициентами и callables.

    Примечания:
    - p_k(0)=0 по построению.
    - Степень p_k равна k+1.
    """
    out: List[RawPoly] = []
    for k in range(N):
        coeffs = raw_coeffs(k, T, family)
        out.append(RawPoly(k=k, T=T, family=family, coeffs=coeffs))
    return out


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Небольшие проверки корректности
    T = 1.0
    N = 5
    fam = "chebyshev"

    raws = build_raw_family(N, T, fam)  # p_k = x*T_k(2x-1)
    xs = np.linspace(0, T, 6)

    # 1) Проверяем p_k(0)=0
    for r in raws:
        v0 = r.eval(0.0)
        assert abs(v0) < 1e-14, f"p_{r.k}(0) != 0"

    # 2) Сравним прямую оценку через Poly с определением p_k(x)
    for r in raws:
        # «истинное» значение через определение: x * T_k(2x/T - 1)
        c_cheb = np.zeros(r.k + 1, dtype=float); c_cheb[-1] = 1.0
        Tk = Chebyshev(c_cheb, domain=[-1, 1])
        y = 2*xs/T - 1
        truth = xs * Tk(y)
        approx = r.eval(xs)
        assert np.allclose(approx, truth, atol=1e-12), f"Mismatch for k={r.k}"

    # 3) Проверяем Caputo-производную для простого случая p_0(x) = x*T_0 = x
    #    D_C^ν x = Γ(2)/Γ(2-ν) * x^{1-ν}
    nu = 0.4
    r0 = raws[0]
    val = r0.dcaputo(nu)(xs[1:])  # избегаем x=0, чтобы сравнение было тривиальным
    ref = (mp.gamma(2)/mp.gamma(2-nu)) * (xs[1:] ** (1-nu))
    assert np.allclose(val, ref, atol=1e-12), "Caputo derivative check for x failed"

    print("poly.py basic self-tests passed.")