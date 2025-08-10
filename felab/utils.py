"""
felab/utils.py

Вспомогательные функции и классы, не привязанные напрямую к математике метода,
но необходимые для инфраструктуры FELAB:
- Таймеры, отладочные принты, безопасные импорты.
- Проверка параметров (валидация α, ε, T, N и т.д.).
- Интерполяция и преобразование форматов данных.
- Кэширование вычислений.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np


T = TypeVar("T")


# --------------------------- ПРОСТОЙ ПРОФИЛЬНЫЙ ТАЙМЕР --------------------- #
@contextlib.contextmanager
def timer(name: str = "", *, stream=sys.stdout):
    """
    Контекстный менеджер для измерения времени выполнения блока.

    Пример:
    >>> with timer("Сборка матрицы"):
    ...     A = assemble(...)
    [Сборка матрицы] 0.0123 s
    """
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    if name:
        print(f"[{name}] {t1 - t0:.4f} s", file=stream)
    else:
        print(f"[timer] {t1 - t0:.4f} s", file=stream)


# ----------------------------- КЭШИРОВАНИЕ --------------------------------- #
def cached(func: Callable[..., T]) -> Callable[..., T]:
    """
    Простейший декоратор-кэш на основе functools.lru_cache(maxsize=None).
    """
    return functools.lru_cache(maxsize=None)(func)


# ---------------------------- ВАЛИДАЦИЯ ПАРАМЕТРОВ ------------------------ #
def validate_alpha(alpha: float) -> float:
    """
    Проверка 0 < α < 1.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha должно быть в (0,1), получено {alpha}")
    return float(alpha)


def validate_eps(eps: float) -> float:
    """
    Проверка eps > 0.
    """
    if not (eps > 0.0):
        raise ValueError(f"eps должно быть > 0, получено {eps}")
    return float(eps)


def validate_T(T: float) -> float:
    """
    Проверка T > 0.
    """
    if not (T > 0.0):
        raise ValueError(f"T должно быть > 0, получено {T}")
    return float(T)


def validate_N(N: int) -> int:
    """
    Проверка N ≥ 1 (число функций в базисе).
    """
    if not (N >= 1):
        raise ValueError(f"N должно быть ≥ 1, получено {N}")
    return int(N)


# --------------------------- МАТЕМАТИЧЕСКИЕ МЕЛОЧИ ------------------------ #
def norm_L2(x: Sequence[float], w: Sequence[float], fvals: Sequence[float]) -> float:
    """
    ||f||_{L2} = sqrt(∑ w_i f_i^2) для узлов x_i и весов w_i.
    """
    fv = np.asarray(fvals, dtype=float)
    wv = np.asarray(w, dtype=float)
    return float(np.sqrt(np.dot(wv, fv * fv)))


def norm_Linf(fvals: Sequence[float]) -> float:
    """
    ||f||_{L∞} на дискретных узлах.
    """
    return float(np.max(np.abs(fvals)))


def relative_error_L2(
    x: Sequence[float],
    w: Sequence[float],
    fvals: Sequence[float],
    gvals: Sequence[float],
) -> float:
    """
    Относительная L2 ошибка: ||f - g|| / ||f||.
    """
    f = np.asarray(fvals, dtype=float)
    g = np.asarray(gvals, dtype=float)
    num = norm_L2(x, w, f - g)
    den = norm_L2(x, w, f)
    return float(num / den if den != 0 else np.inf)


def relative_error_Linf(
    fvals: Sequence[float],
    gvals: Sequence[float],
) -> float:
    """
    Относительная L∞ ошибка: ||f - g||∞ / ||f||∞.
    """
    f = np.asarray(fvals, dtype=float)
    g = np.asarray(gvals, dtype=float)
    num = norm_Linf(f - g)
    den = norm_Linf(f)
    return float(num / den if den != 0 else np.inf)


# --------------------------- ОТОБРАЖЕНИЕ СООБЩЕНИЙ ------------------------ #
def debug_print(*args, enabled: bool = False, **kwargs):
    """
    Печать отладочного сообщения, если enabled=True.
    """
    if enabled:
        print(*args, **kwargs)


# ---------------------------- САМООТЛАДКА ---------------------------------- #
if __name__ == "__main__":
    with timer("demo"):
        time.sleep(0.01)

    @cached
    def ftest(x):
        print("eval ftest")
        return x * 2

    print(ftest(3))
    print(ftest(3))  # кэш, не печатает "eval"

    # Проверка норм
    x = np.array([0.0, 0.5, 1.0])
    w = np.array([0.5, 1.0, 0.5])
    f = np.array([1.0, 2.0, 3.0])
    g = np.array([1.1, 2.1, 3.1])
    print("L2:", norm_L2(x, w, f))
    print("Linf:", norm_Linf(f))
    print("rel L2:", relative_error_L2(x, w, f, g))
    print("rel Linf:", relative_error_Linf(f, g))