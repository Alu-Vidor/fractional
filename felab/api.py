"""
felab/api.py

Высокоуровневое API FELAB.

Главная точка входа:
    solve(a, f, alpha, eps, T, N, ...)

Также предоставляются:
    project_L2(basis, g)         — L2-проекция функции g на span{ψ_k}
    evaluate(solution, x)        — удобный доступ к u_N(x) из Solution

Класс-обёртка результата:
    Solution:
        .basis        — построенный энергетический базис
        .coeffs       — найденные коэффициенты c ∈ R^N
        .u0           — начальное значение u(0)
        .backend_info — метаданные решателя
        .evaluate(x)  — u_N(x)
        .grid(n)      — удобная сетка на [0, T]
        .diagnostics()— простая диагностика: нормы, остаток r=f-Lu_N на узлах квадратуры
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .basis import Basis, build_basis
from .assemble import assemble_system, build_solution_callable, l2_inner
from .solvers import solve_system, SolveOptions
from .quadrature import Quadrature


Array = np.ndarray
Func = Callable[[Array | float], Array]


# ------------------------------- РЕЗУЛЬТАТ --------------------------------- #
@dataclass(frozen=True)
class Solution:
    basis: Basis
    coeffs: Array
    u0: float
    backend_info: Dict

    # --- интерфейс ---
    def evaluate(self, x: Array | float) -> Array:
        uN = build_solution_callable(self.basis, self.coeffs, self.u0)
        return uN(x)

    def grid(self, n: int, kind: str = "uniform") -> Array:
        return self.basis.grid(n, kind=kind)  # type: ignore[arg-type]

    def diagnostics(self) -> Dict[str, float]:
        """
        Простая диагностика на узлах квадратуры:
          - ||r||_{L2}, где r = f - L u_N (если f известна из сборки — недоступно здесь)
          - ||u_N||_{L2}, ||u_N||_{ε,α,a}
          - кондиционирование матрицы жёсткости (если было сохранено в backend_info)
        Примечание: здесь считаем только нормы u_N; остаток r вне solve() мы не знаем.
        """
        B = self.basis
        quad: Quadrature = B.quad
        x, w = quad.nodes, quad.weights
        uN = self.evaluate(x)
        l2_norm = float(np.sqrt(np.dot(w, uN * uN)))
        # Энергетическая норма через inner(u,u)
        e_norm = float(B.inner(lambda t: np.interp(t, x, uN),  # интерполируем u_N на узлах inner (они те же)
                               lambda t: np.interp(t, x, uN)))
        out = {
            "l2_norm_uN": l2_norm,
            "energy_norm_uN": e_norm,
        }
        # доб. диагностическую инфу из решателя, если есть
        for k in ("backend", "resid", "cond_est"):
            if k in self.backend_info and self.backend_info[k] is not None:
                out[f"solver_{k}"] = self.backend_info[k]  # type: ignore[assignment]
        return out


# ------------------------------- SOLVE ------------------------------------- #
def solve(
    *,
    a: Callable[[float], float] | Callable[[Array], Array],
    f: Callable[[float], float] | Callable[[Array], Array],
    alpha: float,
    eps: float,
    T: float,
    N: int,
    raw_family: str = "chebyshev",
    quad_scheme: str = "gauss-legendre",
    quad_n: Optional[int] = None,
    solver: Optional[SolveOptions] = None,
    u0: float = 0.0,
) -> Solution:
    """
    Решить ε-фракционную задачу в энергетическом базисе FELAB.

    Параметры
    ---------
    a, f : callable
        Коэффициент a(x) и правая часть f(x). Могут быть скалярными или векторизованными.
    alpha : float (0<α<1), eps>0, T>0
        Параметры оператора и интервала.
    N : int
        Число функций ψ_k (без учёта φ₀).
    raw_family : {"chebyshev","jacobi"}
        Семейство сырьевых функций для порождения базиса.
    quad_scheme : {"gauss-legendre"}
        Схема квадратуры для энергетической формы (сейчас поддерживается GL).
    quad_n : Optional[int]
        Число узлов квадратуры (по умолчанию max(200, 4N+40)).
    solver : SolveOptions | None
        Параметры решателя линсистемы.
    u0 : float
        Начальное значение u(0)=u0.

    Возвращает
    ----------
    Solution
        Объект с методами evaluate(), grid() и diagnostics().
    """
    # 1) Построить базис
    B = build_basis(
        N=N,
        alpha=alpha,
        eps=eps,
        a_fun=a,
        T=T,
        raw_family=raw_family,  # type: ignore[arg-type]
        quad_scheme="gauss-legendre",  # пока фиксируем
        quad_n=quad_n,
        gs_reorth=True,
        gs_orth_tol=1e-12,
        gs_stable=True,
        gs_reg=0.0,
    )

    # 2) Собрать систему K c = b
    K, b, extras = assemble_system(B, f_fun=f, u0=u0)

    # 3) Решить систему
    opts = solver or SolveOptions()
    x, info = solve_system(K, b, options=opts, spd_hint=extras.SPD_hint)

    # 4) Упаковать результат
    return Solution(basis=B, coeffs=x, u0=u0, backend_info=info)


# ------------------------------- ПРОЕКЦИЯ ---------------------------------- #
def project_L2(
    basis: Basis,
    g: Callable[[float], float] | Callable[[Array], Array],
) -> Tuple[Array, Dict]:
    """
    L2-проекция функции g на span{ψ_k}:
        найти c, минимизирующую || g - Σ c_k ψ_k ||_{L2}.

    Система нормальных уравнений:
        M c = b,
    где M_{ij} = (ψ_j, ψ_i)_{L2}, b_i = (g, ψ_i)_{L2}.

    Возвращает (c, info), где info содержит "cond_est" и "backend".
    """
    psi = basis.psi
    N = len(psi)
    if N == 0:
        raise ValueError("В базисе нет функций ψ (N=0).")
    # Матрица масс M и правая часть b
    M = np.empty((N, N), dtype=float)
    b = np.empty(N, dtype=float)
    for i in range(N):
        for j in range(i, N):
            Mij = l2_inner(basis, psi[j], psi[i])
            M[i, j] = Mij
            M[j, i] = Mij
        b[i] = l2_inner(basis, g, psi[i])

    # Решим плотным Холецким (M — SPD)
    try:
        L = np.linalg.cholesky(M)
        y = np.linalg.solve(L, b)
        c = np.linalg.solve(L.T, y)
        backend = "cholesky"
    except np.linalg.LinAlgError:
        c = np.linalg.solve(M, b)
        backend = "lu"

    # Диагностика
    try:
        s = np.linalg.svd(M, compute_uv=False)
        cond_est = float(s.max() / s.min()) if s.min() > 0 else np.inf
    except Exception:
        cond_est = np.nan

    return c, {"backend": backend, "cond_est": cond_est}


# ------------------------------- УДОБСТВА ---------------------------------- #
def evaluate(solution: Solution, x: Array | float) -> Array:
    """Синоним Solution.evaluate(x)."""
    return solution.evaluate(x)


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Мини e2e тест
    alpha = 0.6
    eps = 1e-3
    T = 1.0
    N = 12
    a = lambda x: 1.0
    u0 = 0.7

    # Случай с «синтетической истиной»: u_true = u0*φ0 + p(x), p = x + 0.25 x^2
    from .basis import build_basis as _build
    B = _build(N=N, alpha=alpha, eps=eps, a_fun=a, T=T, raw_family="chebyshev", quad_n=256)
    lam = -B.a0 / (eps ** alpha)

    import mpmath as mp
    def u_true(x):
        xv = np.asarray(x, dtype=float)
        return u0 * B.phi0(xv) + xv + 0.25 * xv**2

    def Dcap_alpha_poly(x):
        xv = np.asarray(x, dtype=float)
        return (mp.gamma(2)/mp.gamma(2-alpha)) * xv**(1-alpha) + \
               0.25 * (mp.gamma(3)/mp.gamma(3-alpha)) * xv**(2-alpha)

    def f_fun(x):
        xv = np.asarray(x, dtype=float)
        return (eps**alpha) * (lam * B.phi0(xv) + Dcap_alpha_poly(xv)) + u_true(xv)

    sol = solve(a=a, f=f_fun, alpha=alpha, eps=eps, T=T, N=N, u0=u0, quad_n=256)
    X = sol.grid(400)
    err = np.max(np.abs(sol.evaluate(X) - u_true(X)))
    assert err < 2e-2, f"API solve e2e error too large: {err:.3e}"

    # Проверим L2-проекцию (проецируем p(x)=x+x^2/4 из примера на span{ψ})
    c_proj, info_proj = project_L2(sol.basis, lambda x: np.asarray(x, dtype=float) + 0.25*np.asarray(x, dtype=float)**2)
    assert np.isfinite(info_proj["cond_est"]), "Projection cond estimate failed"
    print("api.py basic self-tests passed.")