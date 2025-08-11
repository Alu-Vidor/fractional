"""
felab/solvers.py

Решатели линейных систем для FELAB.

Назначение:
- Компактное и устойчивое решение систем K c = b, где
  K — матрица жёсткости (обычно симметричная, SPD по построению),
  b — правая часть.
- Автовыбор метода в зависимости от размера и свойств:
    * 'lu'         — плотный LU (numpy)
    * 'cholesky'   — плотный Холецкий (numpy), если SPD
    * 'cg'         — итерационный Conjugate Gradient (SciPy, если доступен)
    * 'minres'     — итерационный MINRES для симм. матриц (SciPy, если доступен)
    * 'auto'       — эвристика выбора
- Простые предобуславливатели для итерационных методов:
    * 'jacobi'     — диагональный предобуславливатель
    * 'ilu'        — неполное LU (если SciPy доступен; для плотной матрицы собираем csr)

API:
    SolveOptions
    solve_system(K, b, options: SolveOptions | None = None, spd_hint: bool = True) -> (x, info)

Зависимости:
- numpy (обязательно)
- SciPy (опционально; если отсутствует — итерационные методы недоступны)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional, Tuple, List

import numpy as np


Array = np.ndarray


# --------------------------- ОПЦИИ РЕШАТЕЛЯ -------------------------------- #
@dataclass(frozen=True)
class SolveOptions:
    backend: Literal["auto", "lu", "cholesky", "cg", "minres"] = "auto"
    atol: float = 1e-12
    rtol: float = 1e-10
    maxiter: int = 20000
    preconditioner: Optional[Literal["jacobi", "ilu", "none"]] = None

    def effective_backend(self, n: int, spd_hint: bool) -> str:
        if self.backend != "auto":
            return self.backend
        # Эвристика: для маленьких n — плотные решатели; для больших — итерационные.
        if n <= 1200:
            return "cholesky" if spd_hint else "lu"
        # Для больших систем: CG если SPD, иначе MINRES.
        return "cg" if spd_hint else "minres"

    def with_backend(self, name: str) -> "SolveOptions":
        return replace(self, backend=name)


# ----------------------------- УТИЛИТЫ SPD --------------------------------- #
def _is_spd_dense(A: Array) -> bool:
    """
    Быстрая проверка SPD через попытку Холецкого (без модификации A).
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _cond_est(A: Array, p: int = 1) -> float:
    """
    Грубая оценка числа обусловленности в норме 2 через SVD (дорого O(n^3), но иногда полезно).
    Используется только в отладочной информации и не влияет на решение.
    """
    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s.size == 0 or s.min() == 0:
            return np.inf
        return float(s.max() / s.min())
    except Exception:
        return np.nan


# -------------------------- ПЛОТНЫЕ РЕШАТЕЛИ ------------------------------- #
def _solve_dense_lu(K: Array, b: Array) -> Array:
    return np.linalg.solve(K, b)


def _solve_dense_cholesky(K: Array, b: Array) -> Array:
    L = np.linalg.cholesky(K)
    # Решаем L y = b, L^T x = y
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x


# -------------------------- ИТЕРАЦИОННЫЕ (SciPy) --------------------------- #
def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def _as_linear_operator(K: Array):
    from scipy.sparse.linalg import aslinearoperator  # type: ignore
    return aslinearoperator(K)


def _build_preconditioner(K: Array, kind: Optional[str]):
    """
    Вернуть (LinearOperator) предобуславливатель M ≈ K^{-1} или None.
    """
    if kind is None or kind == "none":
        return None

    try:
        from scipy.sparse import csr_matrix  # type: ignore
        from scipy.sparse.linalg import LinearOperator  # type: ignore
    except Exception:
        return None

    n = K.shape[0]

    if kind == "jacobi":
        d = np.diag(K)
        # Защита от нулевых диагоналей
        d_safe = np.where(np.abs(d) > 1e-30, d, 1.0)
        invd = 1.0 / d_safe

        def mvec(x):
            return invd * x

        return LinearOperator((n, n), matvec=mvec)

    if kind == "ilu":
        try:
            from scipy.sparse.linalg import spilu  # type: ignore
        except Exception:
            return None
        A = csr_matrix(K)
        ilu = spilu(A)  # по умолчанию — умеренно плотный ILU

        def mvec(x):
            return ilu.solve(x)

        from scipy.sparse.linalg import LinearOperator
        return LinearOperator((n, n), matvec=mvec)

    return None


def _solve_iterative(K: Array, b: Array, backend: str, opts: SolveOptions, spd_hint: bool) -> Tuple[Array, Dict]:
    """
    Решить итерационным методом (CG/MINRES). Требует SciPy.

    В процессе решения печатаются нормы остатка для контроля сходимости.
    """
    from scipy.sparse.linalg import cg, minres  # type: ignore

    Aop = _as_linear_operator(K)
    M = _build_preconditioner(K, opts.preconditioner)
    tol = max(opts.atol, opts.rtol)

    res_hist: List[float] = []

    def _callback(xk: Array) -> None:
        r = b - K @ xk
        resid = float(np.linalg.norm(r))
        res_hist.append(resid)
        print(f"[{backend}] iter {len(res_hist)} resid={resid:.3e}")

    info: Dict = {"backend": backend, "it": None, "resid": None, "res_hist": res_hist}
    if backend == "cg":
        x, exit_code = cg(Aop, b, rtol=tol, maxiter=opts.maxiter, M=M, callback=_callback)
        info["exit_code"] = int(exit_code)
    else:
        # MINRES для симметричных (даже не SPD) матриц
        x, exit_code = minres(Aop, b, tol=tol, maxiter=opts.maxiter, M=M, shift=0.0, callback=_callback)
        info["exit_code"] = int(exit_code)

    # финальный остаток
    r = b - K @ x
    resid_final = float(np.linalg.norm(r))
    info["resid"] = resid_final
    info["it"] = len(res_hist)
    return x, info


# ------------------------------ ГЛАВНАЯ ФУНКЦИЯ ---------------------------- #
def solve_system(
    K: Array,
    b: Array,
    options: Optional[SolveOptions] = None,
    *,
    spd_hint: bool = True,
) -> Tuple[Array, Dict]:
    """
    Решение K c = b.

    Параметры
    ---------
    K : ndarray (N×N)
        Матрица жёсткости. Часто симметрична и SPD.
    b : ndarray (N,)
        Правая часть.
    options : SolveOptions | None
        Параметры решателя. Если None, используются значения по умолчанию.
    spd_hint : bool
        Подсказка о SPD. Если True и проверка провалится, метод сменится на LU/MINRES.

    Возвращает
    ---------
    (x, info):
        x    — решение,
        info — словарь с метаданными (backend, cond_est, exit_code, resid, spd_used).
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K должна быть квадратной матрицей.")
    if b.ndim != 1 or b.shape[0] != K.shape[0]:
        raise ValueError("Размерность b не совпадает с K.")

    n = K.shape[0]
    opts = options or SolveOptions()

    backend = opts.effective_backend(n, spd_hint)
    info: Dict = {
        "backend": backend,
        "spd_hint": bool(spd_hint),
        "spd_used": None,
        "exit_code": 0,
        "resid": None,
        "cond_est": None,
    }

    # Для небольших систем — можно оценить обусловленность (диагностика)
    if n <= 600:
        info["cond_est"] = _cond_est(K)

    # Ветвление по бэкенду
    if backend == "cholesky":
        # Убедимся, что SPD в действительности (иначе fallback на LU)
        if spd_hint and _is_spd_dense(K):
            x = _solve_dense_cholesky(K, b)
            r = b - K @ x
            info["spd_used"] = True
            info["resid"] = float(np.linalg.norm(r))
            return x, info
        # Fallback
        backend = "lu"
        info["backend"] = backend

    if backend == "lu":
        x = _solve_dense_lu(K, b)
        r = b - K @ x
        info["spd_used"] = False
        info["resid"] = float(np.linalg.norm(r))
        return x, info

    # Итерационные методы требуют SciPy
    if backend in ("cg", "minres"):
        if not _have_scipy():
            # Fallback на плотные решатели
            fallback = "cholesky" if (spd_hint and _is_spd_dense(K)) else "lu"
            info["backend"] = f"{backend}->fallback:{fallback}"
            if fallback == "cholesky":
                x = _solve_dense_cholesky(K, b)
            else:
                x = _solve_dense_lu(K, b)
            r = b - K @ x
            info["spd_used"] = (fallback == "cholesky")
            info["resid"] = float(np.linalg.norm(r))
            return x, info

        # Есть SciPy — решаем итерационно
        x, it_info = _solve_iterative(K, b, backend, opts, spd_hint)
        info.update(it_info)
        return x, info

    # Неизвестный режим — fallback на LU
    info["backend"] = "lu"
    x = _solve_dense_lu(K, b)
    r = b - K @ x
    info["resid"] = float(np.linalg.norm(r))
    return x, info


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Небольшие тесты
    rng = np.random.default_rng(0)
    n = 50
    # Сгенерируем SPD матрицу K = A^T A + 1e-3 I
    A = rng.standard_normal((n, n))
    K = A.T @ A + 1e-3 * np.eye(n)
    x_true = rng.standard_normal(n)
    b = K @ x_true

    # 1) Автовыбор (должен выбрать cholesky)
    x, info = solve_system(K, b, SolveOptions(backend="auto"))
    assert np.allclose(x, x_true, rtol=1e-10, atol=1e-10), f"auto failed: {info}"

    # 2) Явно cholesky
    x2, info2 = solve_system(K, b, SolveOptions(backend="cholesky"))
    assert np.allclose(x2, x_true, rtol=1e-10, atol=1e-10), f"cholesky failed: {info2}"

    # 3) LU fallback
    K_bad = K.copy()
    # Сделаем K_bad несин. положительно определённой (симметрию портить не будем)
    K_bad[0, 0] = -abs(K_bad[0, 0])
    x3, info3 = solve_system(K_bad, b, SolveOptions(backend="cholesky"), spd_hint=True)
    assert np.allclose(K_bad @ x3, b, rtol=1e-10, atol=1e-10), f"fallback LU failed: {info3}"

    # 4) Если SciPy доступен — проверим итерационные
    if _have_scipy():
        x4, info4 = solve_system(K, b, SolveOptions(backend="cg", preconditioner="jacobi"))
        assert np.linalg.norm(K @ x4 - b) < 1e-8, f"CG failed: {info4}"

        # Для не-SPD возьмём симметричную, но с плохим элементом
        Ks = 0.5 * (K_bad + K_bad.T)
        x5, info5 = solve_system(Ks, b, SolveOptions(backend="minres", preconditioner="jacobi"), spd_hint=False)
        assert np.linalg.norm(Ks @ x5 - b) < 1e-8, f"MINRES failed: {info5}"

    print("solvers.py basic self-tests passed.")