"""Apoio aos testes. Tudo aqui usa mercado sintético e corretora simulada: nenhum teste
fala com a IQ Option nem toca no banco de dados real."""
import numpy as np
import pytest

from iq_agent.banco import Banco
from iq_agent.config import Config
from iq_agent.velas import Velas

FIM = 1_794_000_000  # instante fixo (múltiplo de 300), para os testes serem reprodutíveis


def velas_de_cores(cores: str, inicio: int = FIM, tf: int = 60, passo: float = 0.001) -> Velas:
    """Monta velas a partir de letras: A = alta, B = baixa, D = doji."""
    ts, o, h, l, c = [], [], [], [], []
    preco = 1.0
    for i, letra in enumerate(cores):
        abertura = preco
        fechamento = abertura + {"A": passo, "B": -passo, "D": 0.0}[letra]
        ts.append(inicio + i * tf)
        o.append(abertura)
        c.append(fechamento)
        h.append(max(abertura, fechamento) + passo / 4)
        l.append(min(abertura, fechamento) - passo / 4)
        preco = fechamento
    return Velas(np.array(ts, dtype=np.int64), np.array(o), np.array(h), np.array(l), np.array(c),
                 np.ones(len(ts)), tf)


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.raiz = tmp_path
    config.db = tmp_path / "teste.db"
    return config


@pytest.fixture
def banco():
    b = Banco(":memory:")
    yield b
    b.fechar()
