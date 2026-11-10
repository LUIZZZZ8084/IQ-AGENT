"""Estrategias de entrada.

Cada estrategia recebe as velas e devolve um array com +1 (CALL), -1 (PUT) ou 0 (nada).
O sinal na posicao t usa apenas dados ate o fechamento da vela t; a entrada acontece
na abertura da vela seguinte.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import indicadores as ind
from .velas import Velas


def _cores(v: Velas) -> np.ndarray:
    """+1 vela de alta, -1 vela de baixa, 0 doji."""
    return np.sign(v.close - v.open).astype(np.int8)


def _tamanho_sequencia(cor: np.ndarray) -> np.ndarray:
    """Quantas velas seguidas da mesma cor terminam em cada posição (0 para doji)."""
    n = len(cor)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    mudou = np.r_[True, cor[1:] != cor[:-1]]
    inicio = np.flatnonzero(mudou)
    seq = np.arange(n) - inicio[np.cumsum(mudou) - 1] + 1
    seq[cor == 0] = 0
    return seq.astype(np.int32)


def _vazio(v: Velas) -> np.ndarray:
    return np.zeros(len(v), dtype=np.int8)


def rsi_reversao(v: Velas, periodo: int = 14, nivel: float = 30) -> np.ndarray:
    """RSI sobrevendido -> CALL; sobrecomprado -> PUT."""
    sinal = _vazio(v)
    r = ind.rsi(v.close, periodo)
    sinal[r < nivel] = 1
    sinal[r > 100 - nivel] = -1
    return sinal


def bollinger_reversao(v: Velas, periodo: int = 20, desvios: float = 2.0) -> np.ndarray:
    """Fechou abaixo da banda inferior -> CALL; acima da superior -> PUT."""
    sinal = _vazio(v)
    _, sup, inf = ind.bollinger(v.close, periodo, desvios)
    sinal[v.close < inf] = 1
    sinal[v.close > sup] = -1
    return sinal


def ema_tendencia(v: Velas, rapida: int = 9, lenta: int = 21) -> np.ndarray:
    """A favor da tendência: preço cruza a média rápida no sentido da tendência das médias."""
    sinal = _vazio(v)
    mr, ml = ind.ema(v.close, rapida), ind.ema(v.close, lenta)
    fech_ant, mr_ant = ind.anterior(v.close), ind.anterior(mr)
    sinal[(mr > ml) & (v.close > mr) & (fech_ant <= mr_ant)] = 1
    sinal[(mr < ml) & (v.close < mr) & (fech_ant >= mr_ant)] = -1
    return sinal


def sequencia(v: Velas, n: int = 3, modo: str = "reversao") -> np.ndarray:
    """Após n velas seguidas da mesma cor: aposta na virada (reversao) ou na continuação."""
    cor = _cores(v)
    ativo = _tamanho_sequencia(cor) >= n
    direcao = -cor if modo == "reversao" else cor
    return np.where(ativo, direcao, 0).astype(np.int8)


def mhi(v: Velas, entrada: int = 1, modo: str = "minoria") -> np.ndarray:
    """MHI: olha as 3 últimas velas de cada bloco de 5 e entra na 1ª, 2ª ou 3ª vela do bloco
    seguinte, apostando na cor que apareceu menos (minoria) ou mais (maioria)."""
    n = len(v)
    sinal = _vazio(v)
    if n < 5:
        return sinal
    cor = _cores(v).astype(np.int32)
    c1, c2 = np.roll(cor, 1), np.roll(cor, 2)
    seguidas = (v.ts - np.roll(v.ts, 2)) == 2 * v.tf
    fim_bloco = ((v.ts // v.tf) % 5 == 4) & (cor != 0) & (c1 != 0) & (c2 != 0) & seguidas
    fim_bloco[:2] = False
    maioria = np.sign(cor + c1 + c2)
    base = np.where(fim_bloco, -maioria if modo == "minoria" else maioria, 0)
    atraso = entrada - 1
    if atraso:
        deslocado = np.roll(base, atraso)
        deslocado[:atraso] = 0
        continuo = (v.ts - np.roll(v.ts, atraso)) == atraso * v.tf
        continuo[:atraso] = False
        base = np.where(continuo, deslocado, 0)
    return base.astype(np.int8)


def engolfo(v: Velas) -> np.ndarray:
    """Engolfo de alta -> CALL; engolfo de baixa -> PUT."""
    sinal = _vazio(v)
    if len(v) < 2:
        return sinal
    cor = _cores(v)
    cor_ant = np.roll(cor, 1)
    ab_ant, fe_ant = ind.anterior(v.open), ind.anterior(v.close)
    corpo, corpo_ant = np.abs(v.close - v.open), np.abs(fe_ant - ab_ant)
    alta = (cor_ant < 0) & (cor > 0) & (v.close >= ab_ant) & (v.open <= fe_ant) & (corpo > corpo_ant)
    baixa = (cor_ant > 0) & (cor < 0) & (v.close <= ab_ant) & (v.open >= fe_ant) & (corpo > corpo_ant)
    alta[0] = baixa[0] = False
    sinal[alta] = 1
    sinal[baixa] = -1
    return sinal


def estocastico_reversao(v: Velas, periodo: int = 14, nivel: float = 20) -> np.ndarray:
    """%K cruza %D para cima em zona sobrevendida -> CALL; o inverso -> PUT."""
    sinal = _vazio(v)
    k, d = ind.estocastico(v.high, v.low, v.close, periodo)
    k_ant, d_ant = ind.anterior(k), ind.anterior(d)
    sinal[(k < nivel) & (k > d) & (k_ant <= d_ant)] = 1
    sinal[(k > 100 - nivel) & (k < d) & (k_ant >= d_ant)] = -1
    return sinal


def candle_forca(v: Velas, mult: float = 1.5, modo: str = "continuacao") -> np.ndarray:
    """Vela com corpo bem maior que a volatilidade recente: segue (continuacao) ou vira (reversao)."""
    cor = _cores(v)
    referencia = ind.anterior(ind.atr(v.high, v.low, v.close, 14))
    forte = np.abs(v.close - v.open) > mult * referencia
    direcao = cor if modo == "continuacao" else -cor
    return np.where(forte, direcao, 0).astype(np.int8)


def macd_cruzamento(v: Velas) -> np.ndarray:
    """Histograma do MACD cruza o zero: para cima -> CALL; para baixo -> PUT."""
    sinal = _vazio(v)
    _, _, hist = ind.macd(v.close)
    hist_ant = ind.anterior(hist)
    sinal[(hist > 0) & (hist_ant <= 0)] = 1
    sinal[(hist < 0) & (hist_ant >= 0)] = -1
    return sinal


def padrao_cores(v: Velas, padrao: str = "AAB", direcao: str = "call") -> np.ndarray:
    """Detector de padrões: quando as últimas velas formam exatamente a sequência de cores
    `padrao` (A = alta, B = baixa; a última letra é a vela que acabou de fechar), entra
    na direção indicada. Qual padrão e qual direção usar é decidido pelo histórico."""
    n, k = len(v), len(padrao)
    sinal = _vazio(v)
    if n < k:
        return sinal
    cor = _cores(v)
    bate = np.ones(n, dtype=bool)
    for atras, letra in enumerate(reversed(padrao)):
        bate &= np.roll(cor, atras) == (1 if letra == "A" else -1)
    if k > 1:
        bate &= (v.ts - np.roll(v.ts, k - 1)) == (k - 1) * v.tf
        bate[: k - 1] = False
    sinal[bate] = 1 if direcao == "call" else -1
    return sinal


def _todos_padroes() -> tuple[dict, ...]:
    return tuple({"padrao": "".join(letras), "direcao": direcao}
                 for k in (3, 4)
                 for letras in itertools.product("AB", repeat=k)
                 for direcao in ("call", "put"))


@dataclass(frozen=True)
class Estrategia:
    nome: str
    func: Callable[..., np.ndarray]
    descricao: str
    grade: dict[str, list] = field(default_factory=dict)
    combinacoes: tuple[dict, ...] = ()  # usado quando os parâmetros andam em pares fixos

    def parametros(self) -> list[dict]:
        """Todas as combinações de parâmetros testadas para esta estratégia."""
        if self.combinacoes:
            return [dict(c) for c in self.combinacoes]
        if not self.grade:
            return [{}]
        chaves = list(self.grade)
        return [dict(zip(chaves, valores)) for valores in itertools.product(*(self.grade[c] for c in chaves))]


ESTRATEGIAS: dict[str, Estrategia] = {
    e.nome: e
    for e in (
        Estrategia("rsi_reversao", rsi_reversao, "RSI em extremo (reversão)",
                   grade={"periodo": [7, 14], "nivel": [20, 25, 30]}),
        Estrategia("bollinger_reversao", bollinger_reversao, "Fechamento fora das Bandas de Bollinger (reversão)",
                   grade={"periodo": [20], "desvios": [2.0, 2.5]}),
        Estrategia("ema_tendencia", ema_tendencia, "Cruzamento do preço com a média, a favor da tendência",
                   combinacoes=({"rapida": 9, "lenta": 21}, {"rapida": 20, "lenta": 50})),
        Estrategia("sequencia", sequencia, "Sequência de velas da mesma cor",
                   grade={"n": [3, 4, 5], "modo": ["reversao", "continuacao"]}),
        Estrategia("mhi", mhi, "MHI (3 últimas velas do bloco de 5)",
                   grade={"entrada": [1, 2, 3], "modo": ["minoria", "maioria"]}),
        Estrategia("engolfo", engolfo, "Padrão de engolfo"),
        Estrategia("estocastico_reversao", estocastico_reversao, "Estocástico cruzando em zona extrema",
                   grade={"periodo": [14], "nivel": [20, 10]}),
        Estrategia("candle_forca", candle_forca, "Vela de força (corpo grande)",
                   grade={"mult": [1.5, 2.0], "modo": ["continuacao", "reversao"]}),
        Estrategia("macd_cruzamento", macd_cruzamento, "Histograma do MACD cruzando o zero"),
        Estrategia("padrao_cores", padrao_cores, "Padrão de cores das últimas 3 ou 4 velas (aprendido do histórico)",
                   combinacoes=_todos_padroes()),
    )
}

NOMES_CURTOS = {
    "rsi_reversao": "RSI", "bollinger_reversao": "Bollinger", "ema_tendencia": "Média (tendência)",
    "sequencia": "Sequência", "mhi": "MHI", "engolfo": "Engolfo", "estocastico_reversao": "Estocástico",
    "candle_forca": "Vela de força", "macd_cruzamento": "MACD", "padrao_cores": "Padrão de velas",
}

# Quantas velas de histórico as estratégias precisam para os indicadores estabilizarem.
VELAS_AQUECIMENTO = 200
