"""Mercado sintético para testar o sistema sem depender da corretora.

Serve para duas verificações: num passeio aleatório puro o sistema NÃO pode aprovar
nada; quando um padrão é plantado de propósito, o sistema precisa encontrá-lo.
"""
from __future__ import annotations

import time

import numpy as np

from .velas import Velas


def gerar_velas(minutos: int, semente: int, fim_ts: int | None = None, vol: float = 0.00012,
                preco: float = 1.10, reversao_apos: int = 0, forca: float = 0.0) -> Velas:
    """Gera velas de 1 minuto.

    Com `reversao_apos` = 3 e `forca` = 0.2, depois de 3 velas seguidas da mesma cor a
    vela seguinte é forçada a virar em 20% das vezes (ou seja, vira em ~60% no total).
    """
    rng = np.random.default_rng(semente)
    ret = rng.normal(0.0, vol, minutos)
    if reversao_apos and forca > 0:
        sorteio = rng.random(minutos)
        seq, ultima = 0, 0
        for t in range(minutos):
            if seq >= reversao_apos and sorteio[t] < forca:
                ret[t] = -ultima * abs(ret[t])
            cor = 1 if ret[t] > 0 else -1 if ret[t] < 0 else 0
            if cor != 0 and cor == ultima:
                seq += 1
            else:
                seq, ultima = (1, cor) if cor != 0 else (0, 0)
    close = np.round(preco + np.cumsum(ret), 6)
    open_ = np.r_[round(preco, 6), close[:-1]]
    pavio = np.abs(rng.normal(0.0, vol / 2, (2, minutos)))
    high = np.round(np.maximum(open_, close) + pavio[0], 6)
    low = np.round(np.minimum(open_, close) - pavio[1], 6)
    fim = int(fim_ts if fim_ts is not None else time.time())
    fim -= fim % 60
    ts = fim - 60 * np.arange(minutos, 0, -1, dtype=np.int64)
    volume = rng.integers(50, 500, minutos).astype(np.float64)
    return Velas(ts, open_, high, low, close, volume, 60)
