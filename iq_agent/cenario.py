"""Leitura do cenário atual de um ativo a partir das velas reais mais recentes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import indicadores as ind
from .filtros import razao_vol
from .velas import Velas


@dataclass(frozen=True)
class Cenario:
    ativo: str
    ts: int  # fechamento da última vela usada
    preco: float
    tendencia: str  # "alta", "baixa" ou "lateral" (médias de 20 e 50 no gráfico de 5 min)
    forca: float  # distância entre as médias, em múltiplos da volatilidade
    volatilidade: str  # "alta", "normal" ou "baixa" (comparada às últimas 2 horas)
    razao_vol: float
    rsi: float
    variacao_1h: float  # variação percentual na última hora
    sequencia: int  # velas seguidas da mesma cor no M1 (+ alta, - baixa)

    @property
    def leitura(self) -> str:
        partes = [{"alta": "tendência de alta", "baixa": "tendência de baixa",
                   "lateral": "mercado lateral"}[self.tendencia],
                  f"volatilidade {self.volatilidade}"]
        if self.rsi >= 70:
            partes.append("sobrecomprado no curto prazo")
        elif self.rsi <= 30:
            partes.append("sobrevendido no curto prazo")
        if abs(self.sequencia) >= 4:
            partes.append(f"{abs(self.sequencia)} velas seguidas de {'alta' if self.sequencia > 0 else 'baixa'}")
        return "; ".join(partes)


def ler(ativo: str, v1: Velas) -> Cenario | None:
    """Resume o momento do ativo. Devolve None se houver poucas velas."""
    v5 = v1.reamostrar(300)
    if len(v5) < 60 or len(v1) < 130:
        return None
    media20, media50 = ind.ema(v5.close, 20), ind.ema(v5.close, 50)
    atr5 = ind.atr(v5.high, v5.low, v5.close, 14)
    if not np.isfinite(atr5[-1]) or atr5[-1] <= 0 or not np.isfinite(media50[-1]):
        return None
    forca = float((media20[-1] - media50[-1]) / atr5[-1])
    tendencia = "alta" if forca > 0.5 else "baixa" if forca < -0.5 else "lateral"
    razao = float(razao_vol(v1)[-1])
    if not np.isfinite(razao):
        razao = 1.0
    volatilidade = "alta" if razao > 1.25 else "baixa" if razao < 0.8 else "normal"
    cor = np.sign(v1.close - v1.open)
    seq = 0
    for c in cor[::-1]:
        if c == 0 or (seq and c != np.sign(seq)):
            break
        seq += int(c)
    return Cenario(
        ativo=ativo, ts=int(v1.ts[-1]) + v1.tf, preco=float(v1.close[-1]), tendencia=tendencia, forca=forca,
        volatilidade=volatilidade, razao_vol=razao, rsi=float(ind.rsi(v1.close, 14)[-1]),
        variacao_1h=float(v1.close[-1] / v1.close[-61] - 1.0), sequencia=seq,
    )
