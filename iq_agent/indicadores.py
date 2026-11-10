"""Indicadores técnicos. Todos são causais: o valor na vela t só usa dados até a vela t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()

def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Índice de Força Relativa (suavização de Wilder)."""
    delta = np.diff(close, prepend=close[:1])
    alta = np.where(delta > 0, delta, 0.0)
    baixa = np.where(delta < 0, -delta, 0.0)
    media_alta, media_baixa = _wilder(alta, n), _wilder(baixa, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = media_alta / media_baixa
        saida = 100.0 - 100.0 / (1.0 + rs)
    saida = np.where(media_baixa == 0, np.where(media_alta == 0, 50.0, 100.0), saida)
    saida[np.isnan(media_alta)] = np.nan
    return saida


def bollinger(close: np.ndarray, n: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (média, banda superior, banda inferior)."""
    serie = pd.Series(close)
    media = serie.rolling(n, min_periods=n).mean().to_numpy()
    desvio = serie.rolling(n, min_periods=n).std(ddof=0).to_numpy()
    return media, media + k * desvio, media - k * desvio


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Amplitude média verdadeira (volatilidade)."""
    anterior = np.r_[close[:1], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - anterior), np.abs(low - anterior)))
    return _wilder(tr, n)


def estocastico(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14,
                d: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (%K, %D)."""
    maximo = pd.Series(high).rolling(n, min_periods=n).max().to_numpy()
    minimo = pd.Series(low).rolling(n, min_periods=n).min().to_numpy()
    faixa = maximo - minimo
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (close - minimo) / faixa
    k = np.where(faixa == 0, 50.0, k)
    k[np.isnan(maximo)] = np.nan
    return k, sma(k, d)


def macd(close: np.ndarray, rapida: int = 12, lenta: int = 26,
         sinal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (linha MACD, linha de sinal, histograma)."""
    linha = ema(close, rapida) - ema(close, lenta)
    linha_sinal = pd.Series(linha).ewm(span=sinal, adjust=False, min_periods=sinal).mean().to_numpy()
    return linha, linha_sinal, linha - linha_sinal


def anterior(x: np.ndarray, passos: int = 1) -> np.ndarray:
    """Valor de `passos` velas atrás (NaN no começo)."""
    saida = np.empty(len(x), dtype=np.float64)
    saida[:passos] = np.nan
    saida[passos:] = x[:-passos] if passos else x
    return saida
