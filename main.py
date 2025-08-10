# main.py
# Основной эксперимент для FELAB:
# - выбор тестовой задачи (models.py)
# - сборка и решение методом FELAB (api.solve)
# - метрики точности и устойчивости (diagnostics)
# - (опц.) серия экспериментов по сетке параметров и сохранение результатов в CSV
#
# Запуск (примеры):
#   python main.py --model poly_const --N 24 --alpha 0.6 --eps 1e-3 --quad-n 256 --plot
#   python main.py --model phi0_only --sweepN 8,12,16,24,32 --eps 1e-3
#   python main.py --model poly_var --N 24 --sweepEps 1e-2,5e-3,1e-3 --quad-n 256
#
# Требования: numpy, mpmath; matplotlib (опционально для графиков)

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from felab.api import solve
from felab.models import (
    problem_poly_const_a,
    problem_poly_var_a,
    problem_phi0_only_const_a,
    problem_phi0_plus_poly_var_a,
    Problem,
)
from felab.diagnostics import residual_norms, condition_number
from felab.utils import relative_error_L2, relative_error_Linf


# ------------------------------- CLI ПАРСЕР -------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FELAB: основной эксперимент")
    p.add_argument(
        "--model",
        type=str,
        default="poly_const",
        choices=["poly_const", "poly_var", "phi0_only", "phi0_poly_var"],
        help="Выбор тестовой задачи",
    )
    p.add_argument("--alpha", type=float, default=None, help="Порядок α в (0,1)")
    p.add_argument("--eps", type=float, default=None, help="Малый параметр ε>0")
    p.add_argument("--T", type=float, default=None, help="Длина интервала [0,T]")
    p.add_argument("--u0", type=float, default=None, help="Начальное условие u(0)=u0")

    p.add_argument("--N", type=int, default=24, help="Размерность подпространства ψ (без φ0)")
    p.add_argument("--quad-n", type=int, default=None, help="Число узлов квадратуры для ⟨·,·⟩")
    p.add_argument("--raw-family", type=str, default="chebyshev", choices=["chebyshev", "jacobi"])

    p.add_argument("--grid", type=int, default=1000, help="Размер визуализационной сетки для ошибок")
    p.add_argument("--plot", action="store_true", help="Построить графики (если доступен matplotlib)")

    p.add_argument("--sweepN", type=str, default=None, help="Список N через запятую (например: 8,12,16,24)")
    p.add_argument("--sweepEps", type=str, default=None, help="Список ε через запятую (например: 1e-2,5e-3,1e-3)")

    p.add_argument("--out", type=str, default="results.csv", help="Файл для записи результатов серии")

    return p.parse_args()


# ------------------------------- ПОМОЩНИКИ -------------------------------- #
def get_problem(name: str, alpha: Optional[float], eps: Optional[float],
                T: Optional[float], u0: Optional[float]) -> Problem:
    # Базовая задача по имени
    if name == "poly_const":
        prob = problem_poly_const_a()
    elif name == "poly_var":
        prob = problem_poly_var_a()
    elif name == "phi0_only":
        prob = problem_phi0_only_const_a()
    elif name == "phi0_poly_var":
        prob = problem_phi0_plus_poly_var_a()
    else:
        raise ValueError(f"Неизвестная модель: {name}")

    # Переопределения параметров, если заданы
    a = prob.a
    f = prob.f
    _alpha = prob.alpha if alpha is None else alpha
    _eps = prob.eps if eps is None else eps
    _T = prob.T if T is None else T
    _u0 = prob.u0 if u0 is None else u0

    # Собираем новый Problem с теми же a,f, но обновлёнными метаданными
    return Problem(a=a, f=f, u_true=prob.u_true, u0=_u0, alpha=_alpha, eps=_eps, T=_T, meta=prob.meta)


@dataclass
class RunResult:
    model: str
    N: int
    alpha: float
    eps: float
    T: float
    u0: float
    relL2: float
    relLinf: float
    resL2: float
    resLinfQuad: float
    vstar: float
    condK: float
    solver_backend: str
    solver_resid: Optional[float]
    solver_cond_est: Optional[float]


def run_once(
    prob: Problem,
    N: int,
    *,
    quad_n: Optional[int],
    raw_family: str,
    grid_points: int,
    do_plot: bool,
) -> RunResult:
    # Решаем
    sol = solve(
        a=prob.a,
        f=prob.f,
        alpha=prob.alpha,
        eps=prob.eps,
        T=prob.T,
        N=N,
        raw_family=raw_family,
        quad_n=quad_n,
        u0=prob.u0,
    )

    # Оценка точности на сетке (если u_true известна)
    X = np.linspace(0.0, prob.T, grid_points)
    u_true = sol.basis.a_fun(X) * 0.0  # заготовка (не используется) — просто показываем доступ к a(x)
    u_true = prob.u_true(X)
    uN = sol.evaluate(X)

    relL2 = float(relative_error_L2(X, np.ones_like(X) * (X[1] - X[0]), u_true, uN))
    relLinf = float(relative_error_Linf(u_true, uN))

    # Диагностика остатка r = f - L u_N
    diag = residual_norms(sol.basis, sol.coeffs, prob.u0, prob.f, return_details=True)

    # Печать результата
    binfo = sol.backend_info
    print("\n--- Результат ---")
    print(f"Модель: {prob.meta.get('type','custom')}, N={N}, α={prob.alpha}, ε={prob.eps}, T={prob.T}, u0={prob.u0}")
    print(f"решатель: {binfo.get('backend')}, resid≈{binfo.get('resid')}, cond_est≈{binfo.get('cond_est')}")
    print(f"ошибки:  relL2={relL2:.3e},  relL∞={relLinf:.3e}")
    print(f"остаток: ||r||_L2={diag['res_l2']:.3e},  max|r(узлы)|={diag['res_linf_quad']:.3e},  ||r||_V* (оценка)={diag['vstar_est']:.3e}")
    print(f"cond(K)≈{diag['cond_K']:.3e}")

    # (опц.) графики
    if do_plot:
        try:
            import matplotlib.pyplot as plt  # noqa
            plt.figure()
            plt.plot(X, u_true, label="u_true")
            plt.plot(X, uN, linestyle="--", label="u_N (FELAB)")
            plt.xlabel("x")
            plt.ylabel("u")
            plt.title(f"FELAB: {prob.meta.get('type','model')} (N={N}, α={prob.alpha}, ε={prob.eps})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[plot] пропущено: {e}")

    return RunResult(
        model=str(prob.meta.get("type", "custom")),
        N=N,
        alpha=prob.alpha,
        eps=prob.eps,
        T=prob.T,
        u0=prob.u0,
        relL2=relL2,
        relLinf=relLinf,
        resL2=float(diag["res_l2"]),
        resLinfQuad=float(diag["res_linf_quad"]),
        vstar=float(diag["vstar_est"]),
        condK=float(diag["cond_K"]),
        solver_backend=str(binfo.get("backend")),
        solver_resid=(None if binfo.get("resid") is None else float(binfo.get("resid"))),
        solver_cond_est=(None if binfo.get("cond_est") is None else float(binfo.get("cond_est"))),
    )


def write_csv(path: str, rows: List[RunResult]) -> None:
    fieldnames = [
        "model", "N", "alpha", "eps", "T", "u0",
        "relL2", "relLinf", "resL2", "resLinfQuad", "vstar", "condK",
        "solver_backend", "solver_resid", "solver_cond_est",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "model": r.model, "N": r.N, "alpha": r.alpha, "eps": r.eps, "T": r.T, "u0": r.u0,
                "relL2": f"{r.relL2:.6e}", "relLinf": f"{r.relLinf:.6e}",
                "resL2": f"{r.resL2:.6e}", "resLinfQuad": f"{r.resLinfQuad:.6e}",
                "vstar": f"{r.vstar:.6e}", "condK": f"{r.condK:.6e}",
                "solver_backend": r.solver_backend,
                "solver_resid": "" if r.solver_resid is None else f"{r.solver_resid:.6e}",
                "solver_cond_est": "" if r.solver_cond_est is None else f"{r.solver_cond_est:.6e}",
            })
    print(f"[OK] результаты записаны в {path}")


# ---------------------------------- MAIN ----------------------------------- #
def main():
    args = parse_args()

    prob = get_problem(args.model, args.alpha, args.eps, args.T, args.u0)

    # Одиночный прогон или серия?
    results: List[RunResult] = []

    sweepN = None
    sweepEps = None
    if args.sweepN:
        sweepN = [int(s.strip()) for s in args.sweepN.split(",") if s.strip()]
    if args.sweepEps:
        def _parse_eps(s: str) -> float:
            try:
                return float(s)
            except Exception:
                # поддержка формата 1e-3 как строки
                return float(eval(s))  # noqa: S307 (для локального CLI ok)
        sweepEps = [_parse_eps(s.strip()) for s in args.sweepEps.split(",") if s.strip()]

    if sweepN or sweepEps:
        Ns = sweepN if sweepN else [args.N]
        Eps = sweepEps if sweepEps else [prob.eps]
        for N in Ns:
            for eps_val in Eps:
                prob_i = get_problem(args.model, prob.alpha, eps_val, prob.T, prob.u0)
                r = run_once(
                    prob_i, N,
                    quad_n=args.quad_n, raw_family=args.raw_family,
                    grid_points=args.grid, do_plot=args.plot,
                )
                results.append(r)
        write_csv(args.out, results)
    else:
        r = run_once(
            prob, args.N,
            quad_n=args.quad_n, raw_family=args.raw_family,
            grid_points=args.grid, do_plot=args.plot,
        )
        results.append(r)

    # Краткая сводка по серии
    if len(results) > 1:
        print("\n=== Сводка по серии ===")
        # Сортируем по N затем по eps
        for r in sorted(results, key=lambda z: (z.N, z.eps)):
            print(f"N={r.N:>3}, ε={r.eps:<10g} | relL2={r.relL2:.3e}, relL∞={r.relLinf:.3e}, cond(K)≈{r.condK:.3e}")

    print("\nГотово.")


if __name__ == "__main__":
    main()