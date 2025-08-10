"""
felab/config.py

Глобальные и задачеспецифические настройки FELAB.
Содержит dataclass-конфиги с валидацией, значения по умолчанию
и удобные фабрики для загрузки/переопределения (в т.ч. из env).

Использование:
>>> from felab.config import FELABConfig
>>> cfg = FELABConfig.default().override(N=64, alpha=0.6).validate()
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional, Dict, Any
import os
import math
import logging


# ----------------------------- ЛОГГЕР --------------------------------------- #
_LOG = logging.getLogger("felab.config")


# --------------------------- ВСПОМОГАТЕЛЬНОЕ -------------------------------- #
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        _LOG.warning("Невозможно преобразовать %s=%r к float; используем %s", name, v, default)
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        _LOG.warning("Невозможно преобразовать %s=%r к int; используем %s", name, v, default)
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


# ----------------------------- КОНФИГИ ------------------------------------- #
@dataclass(frozen=True)
class QuadratureConfig:
    """
    Настройки квадратур/интегрирования.
    """
    scheme: Literal["gauss-jacobi", "gauss-legendre", "adaptive-mixed"] = "gauss-jacobi"
    n_global: int = 200                    # число узлов глобальной квадратуры
    use_layer_split: bool = True           # делить интеграл на [0, c*eps] и [c*eps, T]
    layer_c_multiplier: float = 8.0        # множитель c в [0, c*eps]
    n_layer: int = 120                     # число узлов в слое
    abs_tol: float = 1e-12
    rel_tol: float = 1e-10
    max_subdiv: int = 12                   # максимум делений при адаптиве

    def validate(self) -> "QuadratureConfig":
        if self.n_global <= 0:
            raise ValueError("QuadratureConfig.n_global должно быть > 0")
        if self.n_layer <= 0:
            raise ValueError("QuadratureConfig.n_layer должно быть > 0")
        if not (self.layer_c_multiplier > 0):
            raise ValueError("QuadratureConfig.layer_c_multiplier должно быть > 0")
        if not (self.abs_tol > 0 and self.rel_tol > 0):
            raise ValueError("QuadratureConfig: допуски должны быть > 0")
        if self.max_subdiv < 0:
            raise ValueError("QuadratureConfig.max_subdiv должно быть >= 0")
        return self

    def override(self, **kwargs: Any) -> "QuadratureConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class GramSchmidtConfig:
    """
    Настройки ортогонализации в энергетической норме.
    """
    reorthogonalize: bool = True           # повторная ортогонализация (MGS2)
    orth_tol: float = 1e-12                # допуск ортогональности
    scale_before_orth: bool = True         # нормировка сырьевых p_k перед ГШ
    stable_projection: bool = True         # проекция на подпространство ⟂ φ0 через solve с регур.
    regularization: float = 0.0            # Tikhonov для проекций (0 = выкл)
    check_values_at_zero: bool = True      # принудительно ψ_k(0)=0

    def validate(self) -> "GramSchmidtConfig":
        if self.orth_tol <= 0:
            raise ValueError("GramSchmidtConfig.orth_tol должно быть > 0")
        if self.regularization < 0:
            raise ValueError("GramSchmidtConfig.regularization должно быть >= 0")
        return self

    def override(self, **kwargs: Any) -> "GramSchmidtConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SolverConfig:
    """
    Настройки решателя линейной системы.
    """
    backend: Literal["auto", "lu", "cholesky", "cg", "minres"] = "auto"
    atol: float = 1e-12
    rtol: float = 1e-10
    maxiter: int = 20000
    preconditioner: Optional[Literal["jacobi", "ilu", "none"]] = None

    def validate(self) -> "SolverConfig":
        if self.atol <= 0 or self.rtol <= 0:
            raise ValueError("SolverConfig: atol/rtol должны быть > 0")
        if self.maxiter <= 0:
            raise ValueError("SolverConfig.maxiter должно быть > 0")
        return self

    def effective_backend(self, n: int, SPD_hint: bool) -> str:
        if self.backend != "auto":
            return self.backend
        # Эвристика выбора:
        if n <= 1200:
            return "lu"
        if SPD_hint:
            return "cg"
        return "minres"

    def override(self, **kwargs: Any) -> "SolverConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Настройки диагностики и записи результата.
    """
    compute_condition_number: bool = True
    compute_energy_norms: bool = True
    store_intermediates: bool = False
    log_level: Literal["WARNING", "INFO", "DEBUG"] = "WARNING"

    def validate(self) -> "DiagnosticsConfig":
        return self

    def override(self, **kwargs: Any) -> "DiagnosticsConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ProblemConfig:
    """
    Постановка задачи: интервалы, параметры оператора и размерность базиса.
    """
    T: float = 1.0                          # длина интервала [0, T]
    alpha: float = 0.6                      # порядок Caputo (0,1)
    eps: float = 1e-3                       # малый параметр ε
    N: int = 40                             # число функций ψ_k (без φ0)
    raw_family: Literal["chebyshev", "jacobi"] = "chebyshev"

    # функции задачи; по умолчанию – простые заглушки (должны быть переопределены в solve)
    a_fun: Callable[[float], float] = field(default=lambda x: 1.0, repr=False)
    f_fun: Callable[[float], float] = field(default=lambda x: 0.0, repr=False)
    u0: float = 1.0

    def validate(self) -> "ProblemConfig":
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("ProblemConfig.alpha должно быть в (0, 1)")
        if not (self.eps > 0.0):
            raise ValueError("ProblemConfig.eps должно быть > 0")
        if not (self.T > 0.0):
            raise ValueError("ProblemConfig.T должно быть > 0")
        if self.N <= 0:
            raise ValueError("ProblemConfig.N должно быть > 0")
        # Доп. проверки разумности:
        if self.eps > self.T:
            _LOG.info("Предупреждение: eps (%g) больше T (%g) — тонкий слой может отсутствовать", self.eps, self.T)
        return self

    def override(self, **kwargs: Any) -> "ProblemConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class FELABConfig:
    """
    Главный конфиг FELAB: агрегирует конфиги задачи, квадратур, Грам–Шмидта, решателя и диагностики.
    """
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    quadrature: QuadratureConfig = field(default_factory=QuadratureConfig)
    gram_schmidt: GramSchmidtConfig = field(default_factory=GramSchmidtConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Глобальные численные параметры метода Миттага–Леффлера
    mlf_tol: float = 1e-14
    mlf_max_terms: int = 2000

    def validate(self) -> "FELABConfig":
        self.problem.validate()
        self.quadrature.validate()
        self.gram_schmidt.validate()
        self.solver.validate()
        self.diagnostics.validate()
        if self.mlf_tol <= 0:
            raise ValueError("FELABConfig.mlf_tol должно быть > 0")
        if self.mlf_max_terms <= 0:
            raise ValueError("FELABConfig.mlf_max_terms должно быть > 0")
        return self

    def override(self, **kwargs: Any) -> "FELABConfig":
        """
        Удобное поверхностное переопределение полей верхнего уровня.
        Для вложенных конфигов передавайте соответствующие dataclass-объекты.
        """
        return replace(self, **kwargs)

    # -------------------------- ФАБРИКИ ------------------------------------ #
    @staticmethod
    def default() -> "FELABConfig":
        """
        Конфиг по умолчанию, независимый от окружения.
        """
        return FELABConfig().validate()

    @staticmethod
    def from_env() -> "FELABConfig":
        """
        Загрузка базовых параметров из переменных окружения (минимально инвазивно).
        Имена переменных:
          FELAB_ALPHA, FELAB_EPS, FELAB_T, FELAB_N,
          FELAB_Q_SCHEME, FELAB_Q_N_GLOBAL, FELAB_Q_N_LAYER,
          FELAB_MLF_TOL, FELAB_MLF_MAX_TERMS,
          FELAB_SOLVER_BACKEND, FELAB_SOLVER_ATOL, FELAB_SOLVER_RTOL, FELAB_SOLVER_MAXITER,
          FELAB_DIAG_LEVEL
        """
        pc = ProblemConfig(
            alpha=_env_float("FELAB_ALPHA", 0.6),
            eps=_env_float("FELAB_EPS", 1e-3),
            T=_env_float("FELAB_T", 1.0),
            N=_env_int("FELAB_N", 40),
            raw_family=_env_str("FELAB_RAW_FAMILY", "chebyshev"),  # "chebyshev" | "jacobi"
        )

        qc = QuadratureConfig(
            scheme=_env_str("FELAB_Q_SCHEME", "gauss-jacobi"),      # "gauss-jacobi"|"gauss-legendre"|"adaptive-mixed"
            n_global=_env_int("FELAB_Q_N_GLOBAL", 200),
            use_layer_split=_env_bool("FELAB_Q_USE_LAYER_SPLIT", True),
            layer_c_multiplier=_env_float("FELAB_Q_LAYER_C", 8.0),
            n_layer=_env_int("FELAB_Q_N_LAYER", 120),
            abs_tol=_env_float("FELAB_Q_ABS_TOL", 1e-12),
            rel_tol=_env_float("FELAB_Q_REL_TOL", 1e-10),
            max_subdiv=_env_int("FELAB_Q_MAX_SUBDIV", 12),
        )

        gsc = GramSchmidtConfig(
            reorthogonalize=_env_bool("FELAB_GS_REORTH", True),
            orth_tol=_env_float("FELAB_GS_ORTH_TOL", 1e-12),
            scale_before_orth=_env_bool("FELAB_GS_SCALE", True),
            stable_projection=_env_bool("FELAB_GS_STABLE_PROJ", True),
            regularization=_env_float("FELAB_GS_REG", 0.0),
            check_values_at_zero=_env_bool("FELAB_GS_CHECK_ZERO", True),
        )

        sc = SolverConfig(
            backend=_env_str("FELAB_SOLVER_BACKEND", "auto"),  # "auto"|"lu"|"cholesky"|"cg"|"minres"
            atol=_env_float("FELAB_SOLVER_ATOL", 1e-12),
            rtol=_env_float("FELAB_SOLVER_RTOL", 1e-10),
            maxiter=_env_int("FELAB_SOLVER_MAXITER", 20000),
            preconditioner=None if _env_str("FELAB_SOLVER_PREC", "none") == "none"
            else _env_str("FELAB_SOLVER_PREC", "none"),  # "jacobi"|"ilu"
        )

        dc = DiagnosticsConfig(
            compute_condition_number=_env_bool("FELAB_DIAG_COND", True),
            compute_energy_norms=_env_bool("FELAB_DIAG_ENERGY", True),
            store_intermediates=_env_bool("FELAB_DIAG_STORE", False),
            log_level=_env_str("FELAB_DIAG_LEVEL", "WARNING"),  # "WARNING"|"INFO"|"DEBUG"
        )

        cfg = FELABConfig(
            problem=pc,
            quadrature=qc,
            gram_schmidt=gsc,
            solver=sc,
            diagnostics=dc,
            mlf_tol=_env_float("FELAB_MLF_TOL", 1e-14),
            mlf_max_terms=_env_int("FELAB_MLF_MAX_TERMS", 2000),
        )

        # Настроим уровень логгера по DiagnosticsConfig
        try:
            logging.getLogger("felab").setLevel(getattr(logging, cfg.diagnostics.log_level))
        except Exception:
            pass

        return cfg.validate()


# ----------------------- УТИЛИТАРНЫЕ ФАБРИКИ -------------------------------- #
def make_problem_config(
    *,
    T: float = 1.0,
    alpha: float = 0.6,
    eps: float = 1e-3,
    N: int = 40,
    raw_family: Literal["chebyshev", "jacobi"] = "chebyshev",
    a_fun: Optional[Callable[[float], float]] = None,
    f_fun: Optional[Callable[[float], float]] = None,
    u0: float = 1.0,
) -> ProblemConfig:
    """
    Быстрая сборка ProblemConfig из параметров задачи.
    """
    pc = ProblemConfig(
        T=T, alpha=alpha, eps=eps, N=N, raw_family=raw_family,
        a_fun=(a_fun if a_fun is not None else (lambda x: 1.0)),
        f_fun=(f_fun if f_fun is not None else (lambda x: 0.0)),
        u0=u0,
    )
    return pc.validate()


def load_config(
    overrides: Optional[Dict[str, Any]] = None,
    *,
    from_environment: bool = False,
) -> FELABConfig:
    """
    Унифицированная точка входа: загрузка конфига с возможностью
    поверхностного переопределения полей верхнего уровня.
    Пример:
        cfg = load_config({"mlf_tol": 1e-16, "problem": make_problem_config(alpha=0.7)})
    """
    cfg = FELABConfig.from_env() if from_environment else FELABConfig.default()
    if overrides:
        # Вложенные объекты ожидаются как dataclass-экземпляры
        cfg = cfg.override(**overrides)
    return cfg.validate()