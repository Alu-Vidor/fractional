"""
felab/mesh.py

Сетки и отображения на интервале [0, T] для FELAB.
Поддерживает:
- равномерные и Чебышёвские узлы на [0, T]
- слоистую разбиение [0, c*eps] ⋃ [c*eps, T] с контролем плотности
- гладкие стретчинги (tanh/sqrt) для усиления около x=0
- конструирование объединённой сетки без дублей и с сортировкой

Зависимости: numpy (только).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Tuple
import numpy as np


Array = np.ndarray


# ----------------------------- УТИЛИТЫ ------------------------------------- #
def _ensure_sorted_unique(x: Array, rtol: float = 0.0, atol: float = 0.0) -> Array:
    """Отсортировать и удалить почти дублирующиеся точки (по абсолютному/относит. допускам)."""
    x = np.asarray(x, dtype=float)
    x.sort()
    if x.size <= 1:
        return x
    keep = [0]
    for i in range(1, x.size):
        xi, xk = x[i], x[keep[-1]]
        if not np.isclose(xi, xk, rtol=rtol, atol=atol):
            keep.append(i)
    return x[np.array(keep, dtype=int)]


def _clip_0T(x: Array, T: float) -> Array:
    """Жёстко обрезать погрешности округления за пределами [0, T]."""
    return np.clip(x, 0.0, T)


# ------------------------- ОТОБРАЖЕНИЯ/МАППИНГИ --------------------------- #
@dataclass(frozen=True)
class IntervalMap:
    """
    Гладкое отображение φ: [0,1] → [a,b], заданное формулой и её производной.
    Используется для построения неравномерных сеток по параметру s∈[0,1].
    """
    a: float
    b: float
    kind: Literal["linear", "tanh", "sqrt"] = "linear"
    strength: float = 4.0  # для 'tanh'
    power: float = 0.5     # для 'sqrt': x = a + (b-a) * s**power

    def map(self, s: Array) -> Array:
        s = np.asarray(s, dtype=float)
        if self.kind == "linear":
            t = s
        elif self.kind == "tanh":
            # центрированное сгущение у левой границы (x≈a), контролируется strength
            k = float(self.strength)
            t = 0.5 * (np.tanh(k * (s - 1.0)) + 1.0)  # s∈[0,1] → t∈[≈0,1]
        elif self.kind == "sqrt":
            p = float(self.power)
            t = np.power(np.clip(s, 0.0, 1.0), p)
        else:
            raise ValueError(f"Неизвестный kind='{self.kind}'")
        return self.a + (self.b - self.a) * t

    def dmap(self, s: Array) -> Array:
        """Производная dφ/ds на [0,1] (нужна редко; для справки/квадратур)."""
        s = np.asarray(s, dtype=float)
        L = (self.b - self.a)
        if self.kind == "linear":
            return np.full_like(s, L)
        elif self.kind == "tanh":
            k = float(self.strength)
            return L * 0.5 * (1.0 - np.tanh(k * (s - 1.0))**2) * k
        elif self.kind == "sqrt":
            p = float(self.power)
            s_ = np.clip(s, 1e-15, 1.0)
            return L * p * np.power(s_, p - 1.0)
        else:
            raise ValueError(f"Неизвестный kind='{self.kind}'")


# ------------------------------- КОНФИГ ------------------------------------ #
@dataclass(frozen=True)
class MeshConfig:
    """
    Конфигурация построения сеток на [0, T].
    """
    T: float = 1.0
    n_uniform: int = 0                      # число равномерных узлов (0 = не использовать)
    n_chebyshev: int = 64                   # число Чебышёвских узлов (на всю область, 0 = выкл)
    use_layer: bool = True                  # разделять на слой и «bulk»
    layer_c_multiplier: float = 8.0         # размер слоя: x ∈ [0, c*eps]
    n_layer: int = 96                       # число узлов в слое
    n_bulk: int = 96                        # число узлов в основной части
    layer_map: Literal["linear", "tanh", "sqrt"] = "tanh"
    bulk_map: Literal["linear", "tanh", "sqrt"] = "linear"
    map_strength: float = 4.0               # параметр 'tanh'
    map_power: float = 0.5                  # параметр 'sqrt'
    deduplicate_rtol: float = 1e-12
    deduplicate_atol: float = 1e-14

    def with_T(self, T: float) -> "MeshConfig":
        return replace(self, T=T)


# --------------------------- БАЗОВЫЕ СЕТКИ --------------------------------- #
def linspace_0T(T: float, n: int, endpoint: bool = True) -> Array:
    """Равномерная сетка на [0, T]."""
    if n <= 0:
        return np.array([0.0, T]) if endpoint else np.array([0.0])
    return np.linspace(0.0, T, int(n), endpoint=endpoint, dtype=float)


def chebyshev_0T(T: float, n: int, endpoint: bool = True) -> Array:
    """
    Узлы Чебышёва I рода, сдвинутые на [0, T].
    Берём стандартные cos-узлы на [-1,1], реиндексируем так, чтобы был рост x.
    """
    if n <= 0:
        return np.array([0.0, T]) if endpoint else np.array([0.0])
    k = np.arange(n, dtype=float)
    x_ref = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))  # (Gauss-Chebyshev interior)
    # включить концы интервала, если нужно endpoint=True
    if endpoint:
        x_ref = np.concatenate(([-1.0], np.sort(x_ref), [1.0]))
    else:
        x_ref = np.sort(x_ref)
    # аффинное преобразование [-1,1] → [0, T]
    x = 0.5 * (x_ref + 1.0) * T
    return x


def jacobi_mapped_0T(T: float, n: int, alpha: float, beta: float, endpoint: bool = True) -> Array:
    """
    Узлы Гаусса–Якоби на [-1,1] → аффинно на [0, T].
    Примечание: здесь без весов; собственно квадратуры реализуются в quadrature.py.
    """
    if n <= 0:
        return np.array([0.0, T]) if endpoint else np.array([0.0])
    # Спектральная аппроксимация узлов через собственные значения матрицы Якоби (Голуб–Вельч):
    # Для простоты и минимальной зависимости оставим прокси через Чебышёва,
    # а точные узлы для интегрирования вычисляются в quadrature.py.
    # Здесь — сетка для дискретизации функций/визуализации.
    return chebyshev_0T(T, n, endpoint=endpoint)


# ---------------------------- СЛОЙНАЯ СЕТКА -------------------------------- #
def layer_interval(eps: float, c: float, T: float) -> Tuple[float, float]:
    """
    Интервал тонкого слоя у x=0: [0, min(c*eps, T)].
    Возвращает кортеж (xL, xR).
    """
    xR = min(max(c * float(eps), 0.0), float(T))
    return 0.0, xR


def param_grid(n: int, kind: Literal["uniform", "chebyshev"]) -> Array:
    """
    Возвращает s-узлы на [0,1] для параметризации отображений:
    - 'uniform': равномерно
    - 'chebyshev': сгущение у границ (узлы Чебышёва в [0,1])
    """
    if n <= 0:
        return np.array([0.0, 1.0])
    if kind == "uniform":
        return np.linspace(0.0, 1.0, int(n), dtype=float)
    elif kind == "chebyshev":
        k = np.arange(n, dtype=float)
        s = 0.5 * (1.0 + np.cos(np.pi * (2 * k + 1) / (2 * n)))  # перенос cos-узлов в [0,1]
        s.sort()
        return s
    else:
        raise ValueError(f"Неизвестный kind='{kind}'")


def mapped_segment(a: float, b: float, n: int,
                   kind: Literal["linear", "tanh", "sqrt"] = "linear",
                   grid: Literal["uniform", "chebyshev"] = "chebyshev",
                   strength: float = 4.0,
                   power: float = 0.5) -> Array:
    """
    Построить n узлов на [a,b] через отображение s↦φ(s).
    """
    s = param_grid(n, grid)
    mp = IntervalMap(a, b, kind=kind, strength=strength, power=power)
    x = mp.map(s)
    return _ensure_sorted_unique(_clip_0T(x, b), rtol=0.0, atol=0.0)


def layered_mesh(T: float,
                 eps: float,
                 c: float,
                 n_layer: int,
                 n_bulk: int,
                 layer_map: Literal["linear", "tanh", "sqrt"] = "tanh",
                 bulk_map: Literal["linear", "tanh", "sqrt"] = "linear",
                 map_strength: float = 4.0,
                 map_power: float = 0.5) -> Array:
    """
    Объединённая слоистая сетка:
      слой: [0, min(c*eps, T)], сгущённый
      основная часть: [min(c*eps, T), T], с выбранным стретчингом
    Концы 0 и T гарантированно включены.
    """
    xL, xR = layer_interval(eps, c, T)
    if xR <= 0.0:  # слой пуст — возвращаем только bulk
        bulk = mapped_segment(0.0, T, n_bulk, kind=bulk_map,
                              strength=map_strength, power=map_power)
        return _ensure_sorted_unique(np.concatenate(([0.0], bulk, [T])))
    layer = mapped_segment(0.0, xR, n_layer, kind=layer_map,
                           strength=map_strength, power=map_power)
    bulk = mapped_segment(xR, T, n_bulk, kind=bulk_map,
                          strength=map_strength, power=map_power)
    x = np.concatenate((layer, bulk))
    x = _ensure_sorted_unique(x, rtol=1e-15, atol=1e-15)
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
    if x[-1] < T:
        x = np.append(x, T)
    return x


# ------------------------------- КЛАСС СЕТКИ ------------------------------- #
@dataclass(frozen=True)
class Mesh1D:
    """
    Носитель узлов на [0, T] с удобными фабриками построения.
    """
    T: float
    nodes: Array

    # ------------- Фабрики ------------- #
    @staticmethod
    def from_config(cfg: MeshConfig, eps: float) -> "Mesh1D":
        parts = []

        if cfg.n_uniform > 0:
            parts.append(linspace_0T(cfg.T, cfg.n_uniform, endpoint=True))

        if cfg.n_chebyshev > 0:
            parts.append(chebyshev_0T(cfg.T, cfg.n_chebyshev, endpoint=True))

        if cfg.use_layer:
            parts.append(
                layered_mesh(
                    T=cfg.T,
                    eps=eps,
                    c=cfg.layer_c_multiplier,
                    n_layer=cfg.n_layer,
                    n_bulk=cfg.n_bulk,
                    layer_map=cfg.layer_map,
                    bulk_map=cfg.bulk_map,
                    map_strength=cfg.map_strength,
                    map_power=cfg.map_power,
                )
            )

        if not parts:  # дефолт
            parts.append(chebyshev_0T(cfg.T, 64, endpoint=True))

        x = np.concatenate(parts)
        x = _ensure_sorted_unique(x, rtol=cfg.deduplicate_rtol, atol=cfg.deduplicate_atol)
        x = _clip_0T(x, cfg.T)
        return Mesh1D(T=cfg.T, nodes=x)

    @staticmethod
    def uniform(T: float, n: int) -> "Mesh1D":
        return Mesh1D(T, linspace_0T(T, n, endpoint=True))

    @staticmethod
    def chebyshev(T: float, n: int) -> "Mesh1D":
        return Mesh1D(T, chebyshev_0T(T, n, endpoint=True))

    @staticmethod
    def layered(T: float, eps: float, c: float, n_layer: int, n_bulk: int,
                layer_map: str = "tanh", bulk_map: str = "linear",
                strength: float = 4.0, power: float = 0.5) -> "Mesh1D":
        x = layered_mesh(T, eps, c, n_layer, n_bulk,
                         layer_map=layer_map, bulk_map=bulk_map,
                         map_strength=strength, map_power=power)
        return Mesh1D(T, x)

    # ------------- Операции ------------- #
    def with_extra_points(self, pts: Array, rtol: float = 1e-12, atol: float = 1e-14) -> "Mesh1D":
        x = _ensure_sorted_unique(np.concatenate((self.nodes, np.asarray(pts, dtype=float))),
                                  rtol=rtol, atol=atol)
        x = _clip_0T(x, self.T)
        return Mesh1D(self.T, x)

    def restrict(self, a: float, b: float) -> "Mesh1D":
        a = max(0.0, float(a))
        b = min(float(b), self.T)
        if a >= b:
            raise ValueError("restrict: пустой интервал")
        mask = (self.nodes >= a) & (self.nodes <= b)
        x = self.nodes[mask]
        if x.size == 0 or x[0] > a:
            x = np.insert(x, 0, a)
        if x[-1] < b:
            x = np.append(x, b)
        return Mesh1D(b - a, x - a)

    def refine_near_zero(self, factor: float = 0.5) -> "Mesh1D":
        """
        Добавить узлы ближе к нулю: промежуточные точки между [0, x_k] для первых K индексов.
        factor∈(0,1): доля к расстоянию до ближайшего соседа (эвристика).
        """
        x = self.nodes
        if x.size < 3:
            return self
        # Возьмём первые ~√n интервалов у нуля:
        K = max(2, int(np.sqrt(x.size)))
        mids = x[1:K] * factor
        return self.with_extra_points(mids)

    # ------------- Вывод/сэмплинг ------------- #
    def grid(self, n: int, kind: Literal["uniform", "chebyshev"] = "uniform") -> Array:
        """Удобная сетка для визуализации/оценки функций на [0, T]."""
        if kind == "uniform":
            return linspace_0T(self.T, n, endpoint=True)
        elif kind == "chebyshev":
            return chebyshev_0T(self.T, n, endpoint=True)
        else:
            raise ValueError(f"Неизвестный kind='{kind}'")

    def as_segments(self) -> Array:
        """
        Возвращает массив длин сегментов Δx_i между соседними узлами.
        """
        return np.diff(self.nodes)

    # ------------- Представление ------------- #
    def __len__(self) -> int:
        return int(self.nodes.size)

    def __repr__(self) -> str:
        return f"Mesh1D(T={self.T}, n={len(self)}, nodes[0]={self.nodes[0]:.3e}, nodes[-1]={self.nodes[-1]:.3e})"