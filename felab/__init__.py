"""
felab: FELAB — ε-фракционно-энергетический спектральный метод с атомом слоя.

Идея пакета:
- Атом начального слоя ϕ₀(x; ε) = E_α(−a(0) x^α / ε^α)
- Ортонормированный в энергетической норме операторно-индуцированный базис {ψ_k}
- Петров–Галёркин в собственной ε-энергии для ε-униформной устойчивости

Основные точки входа (экспортируются на верхний уровень):
- solve(...)        → высокоуровневое решение задачи L u = f
- build_basis(...)  → построение базиса {ϕ₀, ψ₁,…,ψ_N}
- mittag_leffler(...) → E_{α,β}(z), базовая спецфункция метода
- inner_energy(...) → энергетическое скалярное произведение ⟨·,·⟩_{ε,α,a}

Пример:
>>> from felab import solve
>>> sol = solve(a=lambda x: 1.0, f=lambda x: x, alpha=0.6, eps=1e-3, T=1.0, N=32)
>>> x = sol.grid(200)
>>> u = sol.evaluate(x)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Версия пакета
try:  # предпочтительно брать из метаданных установленного дистрибутива
    from importlib.metadata import version as _pkg_version  # type: ignore
    __version__ = _pkg_version("felab")
except Exception:
    __version__ = "0.0.0-dev"

# Настройка логгера пакета (тихий по умолчанию)
import logging as _logging

_logger = _logging.getLogger("felab")
if not _logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.WARNING)


# --- Безопасные «мягкие» импорты верхнеуровневого API ---------------------- #
# Чтобы __init__.py не падал до создания остальных модулей,
# экспортируем заглушки, если реальные реализации ещё не добавлены.

# mittag_leffler
try:
    from .mlf import mittag_leffler  # E_{α,β}(z)
except Exception:  # pragma: no cover
    def mittag_leffler(alpha: float, z: Any, beta: float = 1.0,
                       tol: float = 1e-14, max_terms: int = 1000) -> Any:
        raise RuntimeError(
            "felab.mlf.mittag_leffler ещё недоступен. "
            "Создайте модуль `felab/mlf.py` с функцией `mittag_leffler`."
        )

# inner_energy
try:
    from .energy import inner_energy  # ⟨u,v⟩_{ε,α,a}
except Exception:  # pragma: no cover
    def inner_energy(u: Callable[[float], float],
                     v: Callable[[float], float],
                     alpha: float,
                     eps: float,
                     a_fun: Callable[[float], float],
                     quad: Optional[Any] = None) -> float:
        raise RuntimeError(
            "felab.energy.inner_energy ещё недоступен. "
            "Создайте `felab/energy.py` с функцией `inner_energy`."
        )

# build_basis
try:
    from .basis import build_basis  # построение {ϕ0, ψ_k}
except Exception:  # pragma: no cover
    def build_basis(N: int,
                    alpha: float,
                    eps: float,
                    a_fun: Callable[[float], float],
                    T: float,
                    raw_family: str = "chebyshev") -> Any:
        raise RuntimeError(
            "felab.basis.build_basis ещё недоступен. "
            "Создайте `felab/basis.py` с функцией `build_basis`."
        )

# solve (высокоуровневое API)
try:
    from .api import solve  # основной вход
except Exception:  # pragma: no cover
    def solve(*args, **kwargs) -> Any:
        raise RuntimeError(
            "felab.api.solve ещё недоступен. "
            "Создайте `felab/api.py` с функцией `solve`."
        )

# Диагностика (необязательно)
try:
    from .diagnostics import condition_number, residual_norms  # noqa: F401
except Exception:  # pragma: no cover
    # Пустые заглушки для опциональных фич
    def condition_number(*_args, **_kwargs) -> float:
        raise RuntimeError(
            "felab.diagnostics.condition_number недоступен. "
            "Добавьте `felab/diagnostics.py`."
        )

    def residual_norms(*_args, **_kwargs) -> dict:
        raise RuntimeError(
            "felab.diagnostics.residual_norms недоступен. "
            "Добавьте `felab/diagnostics.py`."
        )


# Удобные алиасы верхнего уровня
__all__ = [
    "__version__",
    "solve",
    "build_basis",
    "inner_energy",
    "mittag_leffler",
    "condition_number",
    "residual_norms",
]