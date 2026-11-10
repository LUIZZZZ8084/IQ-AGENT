"""Série de velas (candles) em arrays numpy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Velas:
    """Velas ordenadas por tempo. `ts` é a abertura de cada vela, em segundos UTC."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    tf: int = 60  # duração de cada vela, em segundos

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def vazia(cls, tf: int = 60) -> "Velas":
        z = np.empty(0, dtype=np.float64)
        return cls(np.empty(0, dtype=np.int64), z, z.copy(), z.copy(), z.copy(), z.copy(), tf)

    @classmethod
    def de_linhas(cls, linhas: Iterable[tuple], tf: int = 60) -> "Velas":
        """Cria a partir de tuplas (ts, open, high, low, close, volume). Ordena e remove repetidas."""
        dados = np.array(list(linhas), dtype=np.float64)
        if dados.size == 0:
            return cls.vazia(tf)
        dados = dados.reshape(-1, 6)
        ts = dados[:, 0].astype(np.int64)
        ordem = np.argsort(ts, kind="stable")
        ts, dados = ts[ordem], dados[ordem]
        # mantém a última ocorrência de cada timestamp
        manter = np.r_[ts[1:] != ts[:-1], True]
        ts, dados = ts[manter], dados[manter]
        return cls(ts, dados[:, 1].copy(), dados[:, 2].copy(), dados[:, 3].copy(), dados[:, 4].copy(),
                   dados[:, 5].copy(), tf)

    def fatia(self, inicio: int | None = None, fim: int | None = None) -> "Velas":
        s = slice(inicio, fim)
        return Velas(self.ts[s], self.open[s], self.high[s], self.low[s], self.close[s], self.volume[s], self.tf)

    def fechadas_ate(self, instante: float) -> "Velas":
        """Só as velas que já terminaram no instante dado (descarta a vela em formação)."""
        n = int(np.searchsorted(self.ts + self.tf, instante, side="right"))
        return self.fatia(0, n)

    def reamostrar(self, tf: int) -> "Velas":
        """Agrupa em velas maiores (ex.: 60s -> 300s). Grupos incompletos são descartados."""
        if tf == self.tf:
            return self
        if tf % self.tf:
            raise ValueError(f"Não dá para reamostrar de {self.tf}s para {tf}s.")
        if len(self) == 0:
            return Velas.vazia(tf)
        fator = tf // self.tf
        grupo = self.ts // tf
        inicio = np.flatnonzero(np.r_[True, grupo[1:] != grupo[:-1]])
        contagem = np.diff(np.r_[inicio, len(self.ts)])
        completos = contagem == fator
        return Velas(
            ts=(grupo[inicio] * tf)[completos],
            open=self.open[inicio][completos],
            high=np.maximum.reduceat(self.high, inicio)[completos],
            low=np.minimum.reduceat(self.low, inicio)[completos],
            close=self.close[inicio + contagem - 1][completos],
            volume=np.add.reduceat(self.volume, inicio)[completos],
            tf=tf,
        )
