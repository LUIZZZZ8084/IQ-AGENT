"""Estatística usada para separar vantagem real de sorte."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

import numpy as np


def z_unilateral(confianca: float) -> float:
    return NormalDist().inv_cdf(confianca)


def breakeven(payout: float) -> float:
    """Taxa de acerto mínima para não perder dinheiro. Payout 0.85 -> 54,05%."""
    return 1.0 / (1.0 + payout)


def valor_esperado(p: float, payout: float) -> float:
    """Lucro médio por unidade apostada, com taxa de acerto p."""
    return p * payout - (1.0 - p)


def kelly(p: float, payout: float) -> float:
    """Fração de Kelly para aposta binária (0 se não houver vantagem)."""
    if payout <= 0:
        return 0.0
    return max(0.0, (p * (1.0 + payout) - 1.0) / payout)


def wilson(vitorias: float, n: float, z: float) -> tuple[float, float]:
    """Intervalo de Wilson para a taxa de acerto. Devolve (inferior, superior)."""
    if n <= 0:
        return 0.0, 1.0
    p = vitorias / n
    denom = 1.0 + z * z / n
    centro = p + z * z / (2.0 * n)
    margem = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (centro - margem) / denom), min(1.0, (centro + margem) / denom)


def wilson_inferior_vetor(vitorias: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """Limite inferior de Wilson para vários pares (vitórias, n). n = 0 devolve 0."""
    vitorias = np.asarray(vitorias, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    seguro = np.where(n > 0, n, 1.0)
    p = vitorias / seguro
    denom = 1.0 + z * z / seguro
    centro = p + z * z / (2.0 * seguro)
    margem = z * np.sqrt(p * (1.0 - p) / seguro + z * z / (4.0 * seguro * seguro))
    return np.where(n > 0, np.maximum(0.0, (centro - margem) / denom), 0.0)


def p_valor_binomial(vitorias: int, n: int, p0: float) -> float:
    """P(acertar `vitorias` ou mais em n tentativas) se a taxa real fosse p0. Cálculo exato."""
    vitorias, n = int(vitorias), int(n)
    if n <= 0 or vitorias <= 0:
        return 1.0
    if vitorias > n:
        return 0.0
    if not 0.0 < p0 < 1.0:
        return 1.0 if p0 >= 1.0 else 0.0
    log_p, log_q = math.log(p0), math.log1p(-p0)
    primeiro = (math.lgamma(n + 1) - math.lgamma(vitorias + 1) - math.lgamma(n - vitorias + 1)
                + vitorias * log_p + (n - vitorias) * log_q)
    ks = np.arange(vitorias, n, dtype=np.float64)
    passos = np.log(n - ks) - np.log(ks + 1.0) + (log_p - log_q)
    logs = primeiro + np.concatenate(([0.0], np.cumsum(passos)))
    topo = float(logs.max())
    return float(min(1.0, math.exp(topo) * np.exp(logs - topo).sum()))


def benjamini_hochberg(p_valores: Sequence[float]) -> np.ndarray:
    """q-valores (controle da taxa de falsas descobertas)."""
    p = np.asarray(p_valores, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    ordem = np.argsort(p)
    q = p[ordem] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    saida = np.empty(m, dtype=np.float64)
    saida[ordem] = np.minimum(q, 1.0)
    return saida
