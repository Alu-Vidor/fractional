"""
felab/basis.py

Сборка операторно-энергетического базиса 𝔅_N(ε) = { φ₀, ψ₁, …, ψ_N }.

Идея:
- φ₀(x; ε) = E_α( -a(0) x^α / ε^α ) — атом начального слоя.
- {ψ_k} — результат модифицированного Грама–Шмидта в ε-энергии
  над сырьём p_k(x)=x*T_k(2x/T-1) (или x*P_k(2x/T-1)), ортогонализованных
  к φ₀ и взаимно в ⟨·,·⟩_{ε,α,a}.

Публичное API:
- build_basis(N, alpha, eps, a_fun, T, raw_family="chebyshev", **opts) -> Basis
- класс Basis:
    .phi0(x)                 — атом
    .psi[k](x)               — k-й базисный вектор, 1≤k≤N (список callables)
    .size                    — N
    .quad                    — квадратура, на которой строился базис
    .inner(u,v)              — ε-энергетическое ⟨u,v⟩ (частично применённая)
    .dhalf_phi0()            — численный интерполятор D_C^{α/2} φ₀(x) на узлах quad
    .grid(n, kind)           — удобная сетка на [0,T]
    .T, .alpha, .eps, .a0    — параметры задачи

Опции (необязательные ключевые аргументы build_basis):
- quad_scheme: "gauss-legendre" (по умолчанию)
- quad_n:      int, число узлов квадратуры (по умолчанию max(200, 4N+40))
- gs_reorth:   bool, повторная ортогонализация (по умолчанию True)
- gs_orth_tol: float, порог отсечения вектора (по умолчанию 1e-12)
- gs_stable:   bool, использовать устойчивую проекцию по Граму (по умолчанию True)
- gs_reg:      float, тихоновская регуляризация (по умолчанию 0.0)

Зависимости: numpy, felab.poly, felab.atom, felab.gram_schmidt, felab.energy, felab.quadrature
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from .poly import build_raw_family
from .atom import phi0_factory
from .gram_schmidt import orthonormalize_energy
from .energy import inner_energy
from .quadrature import build_quadrature_on_0T, Quadrature


Array = np.ndarray
Func = Callable[[Array | float], Array]


# -------------------------------- БАЗИС ------------------------------------ #
@dataclass(frozen=True)
class Basis:
    """
    Хранилище энергетического базиса и удобные методы.
    """
    T: float
    alpha: float
    eps: float
    a0: float
    a_fun: Callable[[Array], Array]
    quad: Quadrature
    phi0: Func
    psi: List[Func]  # длины N
    _inner: Callable[[Func, Func], float]

    @property
    def size(self) -> int:
        return len(self.psi)

    def inner(self, u: Func, v: Func) -> float:
        return self._inner(u, v)

    def dhalf_phi0(self) -> Callable[[Array | float], Array]:
        """
        Численный интерполятор D_C^{α/2} φ₀(x) на узлах квадратуры (см. atom.Phi0.dhalf).
        """
        from .atom import Phi0  # тип для mypy
        if not hasattr(self.phi0, "__self__") or not isinstance(self.phi0.__self__, object):
            # Если вдруг передали «голую» функцию, переупакуем в фабрику для dhalf.
            # Более надёжно: пересоберём объект Phi0 и вызовем dhalf.
            p = phi0_factory(self.a0, self.eps, self.alpha)
            return p.dhalf(self.quad)
        # В обычной сборке phi0 — bound-method Phi0.__call__, так что извлечём объект:
        obj = self.phi0.__self__  # type: ignore[attr-defined]
        try:
            return obj.dhalf(self.quad)  # type: ignore[no-any-return]
        except Exception:
            # Fallback
            p = phi0_factory(self.a0, self.eps, self.alpha)
            return p.dhalf(self.quad)

    def grid(self, n: int, kind: Literal["uniform", "chebyshev"] = "uniform") -> Array:
        """
        Удобная сетка на [0, T] для визуализации.
        """
        if n <= 1:
            return np.array([0.0, self.T])
        if kind == "uniform":
            return np.linspace(0.0, self.T, int(n), dtype=float)
        elif kind == "chebyshev":
            k = np.arange(n, dtype=float)
            x_ref = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
            x_ref.sort()
            return 0.5 * (x_ref + 1.0) * self.T
        else:
            raise ValueError("Unknown kind for grid()")


# ---------------------------- ВСПОМОГАТЕЛЬНОЕ ------------------------------ #
def _as_callable_vec(f: Callable[[float], float] | Callable[[Array], Array]) -> Callable[[Array], Array]:
    """
    Обёртка для векторизуемых функций a(x), f(x).
    """
    def g(x: Array) -> Array:
        xv = np.asarray(x, dtype=float)
        out = f(xv)  # type: ignore
        if np.isscalar(out):
            return np.full_like(xv, float(out))
        return np.asarray(out, dtype=float)
    return g


# -------------------------------- СБОРКА ----------------------------------- #
def build_basis(
    N: int,
    alpha: float,
    eps: float,
    a_fun: Callable[[float], float] | Callable[[Array], Array],
    T: float,
    raw_family: Literal["chebyshev", "jacobi"] = "chebyshev",
    *,
    quad_scheme: Literal["gauss-legendre"] = "gauss-legendre",
    quad_n: Optional[int] = None,
    gs_reorth: bool = True,
    gs_orth_tol: float = 1e-12,
    gs_stable: bool = True,
    gs_reg: float = 0.0,
) -> Basis:
    """
    Построить базис 𝔅_N(ε) = {φ₀, ψ₁..ψ_N}.

    Параметры
    ---------
    N          : число функций ψ_k (без учёта φ₀).
    alpha      : порядок 0<α<1.
    eps        : малый параметр ε>0.
    a_fun      : функция коэффициента a(x) (ожидается a(x) ≥ a0 > 0).
    T          : длина интервала [0, T].
    raw_family : "chebyshev" | "jacobi" — исходное семейство p_k.

    Опции
    -----
    quad_scheme: схема квадратуры на [0,T] для ⟨·,·⟩ (сейчас "gauss-legendre").
    quad_n     : число узлов квадратуры (по умолчанию max(200, 4N+40)).
    gs_*       : настройки Грама–Шмидта (см. описание модуля gram_schmidt.py).

    Возвращает
    ----------
    Basis
    """
    if not (N > 0):
        raise ValueError("N должно быть > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должно быть в (0,1)")
    if not (eps > 0.0):
        raise ValueError("eps должно быть > 0")
    if not (T > 0.0):
        raise ValueError("T должно быть > 0")

    a_vec = _as_callable_vec(a_fun)
    a0 = float(a_vec(np.array([0.0]))[0])
    if not (a0 > 0.0):
        raise ValueError("a(0) должно быть > 0 для корректности энергетической формы.")

    # --- Квадратура для энергетической формы ---
    if quad_n is None:
        quad_n = max(200, 4 * N + 40)
    quad = build_quadrature_on_0T(quad_scheme, T, int(quad_n))

    # --- Атом φ₀ ---
    p0 = phi0_factory(a0=a0, eps=eps, alpha=alpha)
    phi0_fun = p0.__call__  # bound method (Basis.dhalf_phi0 опирается на это)

    # --- Частично применённая ε-энергия ---
    inner = partial(inner_energy, alpha=alpha, eps=eps, a_fun=a_vec, quad=quad)

    # --- Сырьё p_k ---
    raws = build_raw_family(N, T, raw_family)
    raw_funcs: List[Func] = [r.eval for r in raws]  # callables x↦p_k(x)

    # --- Ортонормировка в энергии (операторно-индуцированный MGS) ---
    psi_funcs = orthonormalize_energy(
        phi0=phi0_fun,
        raw_funcs=raw_funcs,
        inner=inner,
        reorth=gs_reorth,
        orth_tol=gs_orth_tol,
        stable_projection=gs_stable,
        regularization=gs_reg,
        enforce_zero_at=0.0,
        enforce_zero=True,
        max_zero_iters=3,
    )

    # Если из-за численной линзависимости получили меньше N функций — это допустимо,
    # но предупредим пользователя (здесь — мягко: просто логируем через print).
    if len(psi_funcs) < N:
        # Не поднимаем исключение; при желании можно усилить сырьё или увеличить quad_n.
        print(f"[felab.basis] Предупреждение: получено только {len(psi_funcs)} функций ψ (запрошено N={N}).")

    return Basis(
        T=T,
        alpha=alpha,
        eps=eps,
        a0=a0,
        a_fun=a_vec,
        quad=quad,
        phi0=phi0_fun,
        psi=psi_funcs,
        _inner=inner,
    )


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Простая проверка на синтетической задаче a(x)=1, φ₀ известен,
    # убеждаемся, что базис строится и ортонормирован (примерно).
    T = 1.0
    alpha = 0.6
    eps = 1e-3
    a = lambda x: 1.0

    B = build_basis(N=16, alpha=alpha, eps=eps, a_fun=a, T=T, raw_family="chebyshev", quad_n=256)

    # Проверка ⟨ψ_i, ψ_j⟩ ≈ δ_ij, ⟨ψ_i, φ₀⟩ ≈ 0
    tol_orth = 5e-10
    for i, qi in enumerate(B.psi):
        nrm = B.inner(qi, qi)
        assert abs(nrm - 1.0) < 5e-8, f"Нормировка ψ_{i+1} некорректна: {nrm}"
        # ортогональность между ψ
        for j in range(i):
            val = B.inner(qi, B.psi[j])
            assert abs(val) < tol_orth, f"ψ_{i+1} не ортогональна ψ_{j+1}: {val}"
        # ортогональность к φ0
        val0 = B.inner(qi, B.phi0)
        assert abs(val0) < tol_orth, f"ψ_{i+1} не ортогональна φ0: {val0}"

    # Значения в нуле
    for i, qi in enumerate(B.psi):
        z = qi(0.0)
        z = float(z if not isinstance(z, np.ndarray) else z.item())
        assert abs(z) < 1e-12, f"ψ_{i+1}(0) ≠ 0"

    print("basis.py basic self-tests passed.")