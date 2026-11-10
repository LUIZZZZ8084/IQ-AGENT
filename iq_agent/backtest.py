"""Motor de backtest para opções de expiração fixa (turbo)."""
from __future__ import annotations

import numpy as np

from .velas import Velas


def desfechos(v: Velas, k: int, slippage_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resultado de operar após cada vela.

    Um sinal na vela t entra na abertura da vela t+1 e expira no fechamento da vela t+k.
    Devolve (res_call, res_put, valido); res vale +1 (ganhou), 0 (empate, devolve o valor)
    ou -1 (perdeu). `valido` é False onde faltam velas ou há buraco nos dados.

    `slippage_frac` piora a entrada em uma fração do corpo mediano das velas, para
    compensar o atraso real entre o sinal e a execução.
    """
    n = len(v)
    res_call = np.zeros(n, dtype=np.int8)
    res_put = np.zeros(n, dtype=np.int8)
    valido = np.zeros(n, dtype=bool)
    if k < 1:
        raise ValueError("k precisa ser pelo menos 1.")
    if n <= k:
        return res_call, res_put, valido
    idx = np.arange(n - k)
    entrada = v.open[idx + 1]
    saida = v.close[idx + k]
    desliz = slippage_frac * float(np.median(np.abs(v.close - v.open))) if slippage_frac else 0.0
    res_call[: n - k] = np.sign(saida - (entrada + desliz)).astype(np.int8)
    res_put[: n - k] = np.sign((entrada - desliz) - saida).astype(np.int8)
    valido[: n - k] = (v.ts[idx + k] - v.ts[idx]) == k * v.tf
    return res_call, res_put, valido


def sem_sobreposicao(sinal: np.ndarray, k: int) -> np.ndarray:
    """Remove sinais que cairiam enquanto a operação anterior ainda está aberta."""
    if k <= 1:
        return sinal
    saida = np.zeros_like(sinal)
    livre = 0
    for i in np.flatnonzero(sinal).tolist():
        if i >= livre:
            saida[i] = sinal[i]
            livre = i + k
    return saida


def aplicar(sinal: np.ndarray, res_call: np.ndarray, res_put: np.ndarray, valido: np.ndarray,
            k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (sinal efetivo, ganhou, perdeu, empatou) como arrays alinhados às velas."""
    efetivo = np.where(valido, sinal, 0).astype(np.int8)
    efetivo = sem_sobreposicao(efetivo, k)
    resultado = np.where(efetivo > 0, res_call, res_put)
    opera = efetivo != 0
    return efetivo, opera & (resultado > 0), opera & (resultado < 0), opera & (resultado == 0)
