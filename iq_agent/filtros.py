"""Filtros aplicados sobre os sinais: sessão do dia (UTC) e regime de volatilidade."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicadores as ind
from .velas import Velas

# Faixas de hora em UTC: [início, fim)
SESSOES: dict[str, tuple[int, int]] = {
    "todas": (0, 24),
    "asia": (0, 7),
    "londres": (7, 12),
    "ny": (12, 17),
    "tarde_noite": (17, 24),
}
VOLS = ("qualquer", "alta", "baixa")
JANELA_VOL = 120  # velas usadas como referência de volatilidade "normal"


def hora_utc(ts: np.ndarray | float) -> np.ndarray | int:
    return (np.asarray(ts, dtype=np.int64) // 3600) % 24


def descrever_sessao(sessao: str) -> str:
    """Faixa de horário da sessão no relógio do computador (ex.: '09h–14h')."""
    if sessao == "todas":
        return "qualquer horário"
    from datetime import datetime

    inicio, fim = SESSOES[sessao]
    desloc = round(datetime.now().astimezone().utcoffset().total_seconds() / 3600)
    return f"{(inicio + desloc) % 24:02d}h–{(fim + desloc) % 24:02d}h"


def sessao_inclui(sessao: str, ts_entrada: float) -> bool:
    inicio, fim = SESSOES[sessao]
    return inicio <= int(hora_utc(ts_entrada)) < fim


def mascara_sessao(v: Velas, sessao: str) -> np.ndarray:
    """True onde a ENTRADA (abertura da vela seguinte ao sinal) cai dentro da sessão."""
    inicio, fim = SESSOES[sessao]
    hora = hora_utc(v.ts + v.tf)
    return (hora >= inicio) & (hora < fim)


def razao_vol(v: Velas) -> np.ndarray:
    """ATR atual dividido pela mediana do ATR nas últimas JANELA_VOL velas."""
    a = ind.atr(v.high, v.low, v.close, 14)
    mediana = pd.Series(a).rolling(JANELA_VOL, min_periods=JANELA_VOL // 2).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / mediana


def mascara_vol(razao: np.ndarray, vol: str) -> np.ndarray:
    if vol == "qualquer":
        return np.ones(len(razao), dtype=bool)
    if vol == "alta":
        return razao > 1.0
    if vol == "baixa":
        return razao <= 1.0
    raise ValueError(f"Regime de volatilidade desconhecido: {vol}")
