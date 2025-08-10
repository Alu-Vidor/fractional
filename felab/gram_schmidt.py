"""
felab/gram_schmidt.py

Модифицированный Грам–Шмидт в ε-энергии для построения операторно-индуцированного
ортонормированного базиса {ψ_k} при заданном "атоме" φ0.

Особенности:
- Ортогонализация в заданном скалярном произведении ⟨·,·⟩ (обычно — ε-энергия).
- Проекция на подпространство, ортогональное φ0, затем последовательный MGS по ψ.
- Повторная ортогонализация (MGS2) для снижения утечек ортогональности.
- Необязательное "стабильное" снятие проекций через решение малых СЛАУ по Граму.
- Опциональное навязывание ψ_k(0)=0: после шага ортогонализации делаем коррекцию
  вдоль φ0, потом ре-ортогонализация к {φ0, ψ_1..ψ_{k-1}} до допустимости.

API:
    orthonormalize_energy(phi0, raw_funcs, inner, *, reorth=True, orth_tol=1e-12,
                          stable_projection=True, regularization=0.0,
                          enforce_zero_at=0.0, enforce_zero=True, max_zero_iters=3)
        -> list[Callable]

Где:
- phi0: callable x↦φ0(x)
- raw_funcs: список исходных функций (например, p_k), каждая: x↦p_k(x)
- inner(u,v): функция скалярного произведения (обычно felab.energy.inner_energy с частично
  зафиксированными параметрами), принимает два callable и возвращает float.

Зависимости: numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np


Array = np.ndarray
Func = Callable[[Array | float], Array | float]


# --------------------------- ЛИНЕЙНЫЕ КОМБИНАЦИИ --------------------------- #
@dataclass
class LinComb:
    """
    Представление функции как линейной комбинации компонент:
        f(x) = sum_i c_i * g_i(x)
    Удобно для стабильно накапливаемых преобразований.
    """
    comps: List[Func]
    coeffs: np.ndarray  # shape (m,)

    def __call__(self, x: Array | float) -> Array:
        # Векторизованная оценка суммы
        vals = None
        for gi, ci in zip(self.comps, self.coeffs):
            if ci == 0.0:
                continue
            vi = gi(x)
            vi = np.asarray(vi, dtype=float)
            vals = vi * ci if vals is None else vals + ci * vi
        if vals is None:
            # Нулевая функция
            xv = np.asarray(x, dtype=float)
            return np.zeros_like(xv)
        return vals

    @staticmethod
    def from_single(g: Func, scale: float = 1.0) -> "LinComb":
        return LinComb([g], np.array([float(scale)], dtype=float))

    def add_scaled(self, g: Func, c: float) -> None:
        self.comps.append(g)
        self.coeffs = np.append(self.coeffs, float(c))

    def scale_inplace(self, s: float) -> None:
        self.coeffs *= float(s)

    def copy(self) -> "LinComb":
        return LinComb(self.comps.copy(), self.coeffs.copy())


# ----------------------------- ПРОЕКТОРЫ ----------------------------------- #
def _project_sequential(f: LinComb, basis: Sequence[Func], inner: Callable[[Func, Func], float]) -> LinComb:
    """
    Снять проекции f на заданный (уже ортонормированный!) basis через MGS:
        f ← f - Σ ⟨f, q_i⟩ q_i
    Если basis ортонормирован, это наиболее устойчивый вариант.
    """
    out = f.copy()
    for q in basis:
        coef = inner(out, q)  # ⟨out, q⟩
        if coef != 0.0:
            out.add_scaled(q, -coef)
    return out


def _project_stable(f: LinComb, basis: Sequence[Func], inner: Callable[[Func, Func], float],
                    ridge: float = 0.0) -> LinComb:
    """
    Снять проекции f на span{basis} через решение небольшого Грам-ЛС:
        G c = b,  G_{ij} = ⟨q_i,q_j⟩,  b_i = ⟨f, q_i⟩
        f ← f - Σ c_i q_i
    ridge >= 0 добавляет λI к G (Тихонов), если нужно.
    """
    m = len(basis)
    if m == 0:
        return f.copy()
    G = np.empty((m, m), dtype=float)
    b = np.empty(m, dtype=float)
    for i, qi in enumerate(basis):
        b[i] = inner(f, qi)
        for j in range(i, m):
            Gij = inner(qi, basis[j])
            G[i, j] = Gij
            G[j, i] = Gij
    if ridge > 0.0:
        G = G + ridge * np.eye(m, dtype=float)

    try:
        c = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        # fallback на псевдообратную
        c = np.linalg.lstsq(G, b, rcond=None)[0]

    out = f.copy()
    for qi, ci in zip(basis, c):
        if ci != 0.0:
            out.add_scaled(qi, -ci)
    return out


# --------------------------- НОРМИРОВКА / НУЛИ ----------------------------- #
def _norm(inner: Callable[[Func, Func], float], f: Func) -> float:
    val = inner(f, f)
    return float(np.sqrt(max(val, 0.0)))


def _value_at_zero(f: Func) -> float:
    # Унифицированная проверка значения в нуле
    v = f(0.0)
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.size == 1 else float(v[0])
    return float(v)


def _enforce_zero_at(f: LinComb, phi0: Func, x0: float = 0.0) -> LinComb:
    """
    Навязать f(x0)=0, скорректировав вдоль φ0:
        f ← f - f(x0)/φ0(x0) * φ0
    (если φ0(x0)≈1, как в FELAB, это просто f←f - f(0)*φ0)
    """
    denom = phi0(x0)
    denom = float(denom if not isinstance(denom, np.ndarray) else denom.item())
    if abs(denom) < 1e-15:
        # Если вдруг φ0(0)=0 (в FELAB не так), пропускаем
        return f
    val = _value_at_zero(f)
    if abs(val) < 1e-15:
        return f
    g = f.copy()
    g.add_scaled(phi0, -val / denom)
    return g


# ----------------------------- ОСНОВНАЯ ФУНКЦИЯ ---------------------------- #
def orthonormalize_energy(
    phi0: Func,
    raw_funcs: Sequence[Func],
    inner: Callable[[Func, Func], float],
    *,
    reorth: bool = True,
    orth_tol: float = 1e-12,
    stable_projection: bool = True,
    regularization: float = 0.0,
    enforce_zero_at: float = 0.0,
    enforce_zero: bool = True,
    max_zero_iters: int = 3,
) -> List[Func]:
    """
    Построить операторно-ортонормированный базис {ψ_k} из сырьевых функций,
    устранить компоненту вдоль φ0 и выполнить MGS по уже построенным ψ.

    Параметры
    ---------
    phi0 : callable
        Атом слоя φ0 (не включается в возвращаемый список).
    raw_funcs : последовательность callable
        Сырьё p_k (обычно p_k(0)=0).
    inner : callable
        Скалярное произведение ⟨u,v⟩ (например, частичная обёртка над felab.energy.inner_energy).

    Опции
    -----
    reorth : bool
        Выполнять повторную ортогонализацию (MGS2).
    orth_tol : float
        Допуск на норму при нормировке; слишком малые векторы отбрасываются.
    stable_projection : bool
        Если True, снятие проекций на {φ0, ψ_1..ψ_{k-1}} через решение Грам-СЛАУ.
        Если False, используется последовательный MGS (требует уже ортонормированных ψ).
    regularization : float
        Тихоновская регуляризация (λI) при stable_projection.
    enforce_zero_at : float
        Точка, в которой навязывается условие ψ_k(enforce_zero_at)=0 (обычно 0.0).
    enforce_zero : bool
        Навязывать ли ψ_k(0)=0 (через коррекцию вдоль φ0 с последующей ре-ортогонализацией).
    max_zero_iters : int
        Число итераций «коррекция вдоль φ0 → ре-ортогонализация» для достижения нуля в точке.

    Возвращает
    ----------
    list[callable]
        Ортонормированный список функций ψ_k.
    """
    Q: List[Func] = []        # накопленный ортонормированный базис ψ
    P0 = [phi0]               # подпространство, от которого проектируемся изначально

    for pk in raw_funcs:
        f = LinComb.from_single(pk)

        # 1) Снятие компоненты вдоль φ0
        if stable_projection:
            f = _project_stable(f, P0, inner, ridge=regularization)
        else:
            # Последовательно: нормировка φ0 не требуется для корректности,
            # но если φ0 не нормирован, коэффициент = ⟨f,φ0⟩/⟨φ0,φ0⟩.
            # Это эквивалентно stable-проекции с m=1.
            num = inner(f, phi0)
            den = inner(phi0, phi0)
            if abs(den) > 0:
                f.add_scaled(phi0, -num / den)

        # 2) Снятие компонент на уже построенный Q
        if len(Q) > 0:
            if stable_projection:
                f = _project_stable(f, Q, inner, ridge=regularization)
            else:
                f = _project_sequential(f, Q, inner)

        # 3) Повторная ортогонализация при необходимости
        if reorth and len(Q) > 0:
            if stable_projection:
                f = _project_stable(f, Q, inner, ridge=regularization)
            else:
                f = _project_sequential(f, Q, inner)

        # 4) Необязательное навязывание значения в нуле
        if enforce_zero:
            # Несколько итераций: коррекция вдоль φ0 → ре-ортогонализация к {φ0} и Q
            it = 0
            while it < max_zero_iters:
                val0 = _value_at_zero(f)
                if abs(val0) < 1e-14:
                    break
                f = _enforce_zero_at(f, phi0, x0=enforce_zero_at)
                # Ре-ортогонализация к φ0 и Q, т.к. коррекция нарушила ⟨·,·⟩-орт.
                if stable_projection:
                    f = _project_stable(f, P0, inner, ridge=regularization)
                    if len(Q) > 0:
                        f = _project_stable(f, Q, inner, ridge=regularization)
                else:
                    # Сначала φ0
                    num = inner(f, phi0)
                    den = inner(phi0, phi0)
                    if abs(den) > 0:
                        f.add_scaled(phi0, -num / den)
                    # Затем Q (последовательно)
                    if len(Q) > 0:
                        f = _project_sequential(f, Q, inner)
                it += 1

        # 5) Нормировка
        nrm = _norm(inner, f)
        if not np.isfinite(nrm) or nrm < orth_tol:
            # Слишком слабый (линейно зависимый) вектор — пропускаем
            continue
        f.scale_inplace(1.0 / nrm)

        # 6) Добавляем в базис
        Q.append(f)

    return Q


# ------------------------------- САМООТЛАДКА ------------------------------- #
if __name__ == "__main__":
    # Мини-тест: возьмём искусственное ⟨·,·⟩ = L2 на [0,1] с численной квадратурой
    import numpy as _np
    from typing import Tuple

    # Простейшая квадратура для теста
    X = _np.linspace(0.0, 1.0, 1001)
    W = _np.ones_like(X) * (X[1] - X[0])

    def inner_L2(u: Func, v: Func) -> float:
        uu = _np.asarray(u(X), dtype=float)
        vv = _np.asarray(v(X), dtype=float)
        return float(_np.dot(W, uu * vv))

    # φ0 ~ 1 (для теста); в реальном FELAB φ0(0)=1, быстро спадает
    phi0 = lambda x: _np.ones_like(_np.asarray(x, dtype=float))

    # Сырьё: p_k = x*T_k(2x-1) (но здесь — просто x, x^2, x^3 для простоты)
    raws = [
        lambda x: _np.asarray(x, dtype=float),
        lambda x: _np.asarray(x, dtype=float) ** 2,
        lambda x: _np.asarray(x, dtype=float) ** 3,
        lambda x: _np.asarray(x, dtype=float) ** 4,
    ]

    Q = orthonormalize_energy(phi0, raws, inner_L2,
                              reorth=True, stable_projection=True,
                              enforce_zero=True, orth_tol=1e-12)

    # Проверим ортонормальность и значение в нуле
    for i, qi in enumerate(Q):
        nrm = inner_L2(qi, qi)
        assert abs(nrm - 1.0) < 1e-10, "Неправильная нормировка"
        z = qi(0.0)
        assert abs(float(z)) < 1e-12, "Не навязано значение 0 в x=0"
        for j in range(i):
            val = inner_L2(qi, Q[j])
            assert abs(val) < 1e-10, "Нарушена ортогональность"

    print("gram_schmidt.py basic self-tests passed.")