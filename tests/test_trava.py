"""Trava de instância única (dois painéis ou dois bots sobre os mesmos dados)."""
import pytest

from conftest import FIM
from iq_agent import app
from iq_agent.simulado import ClienteSimulado
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import ErroIQ
from iq_agent.trava import Trava


def test_so_um_por_vez(tmp_path):
    primeira, segunda = Trava(tmp_path / "x.lock"), Trava(tmp_path / "x.lock")
    assert primeira.adquirir() and primeira.adquirir()  # pedir de novo a própria trava não falha
    assert not segunda.adquirir()
    primeira.liberar()
    assert segunda.adquirir()
    segunda.liberar()
    segunda.liberar()  # liberar duas vezes não dá erro


def test_trava_como_bloco_e_pasta_inexistente(tmp_path):
    with Trava(tmp_path / "nova" / "y.lock") as trava:
        assert trava.adquirir()
        assert not Trava(tmp_path / "nova" / "y.lock").adquirir()
    assert Trava(tmp_path / "nova" / "y.lock").adquirir()


def test_segundo_bot_e_recusado(cfg, banco):
    cliente = ClienteSimulado({"X": gerar_velas(2000, semente=1, fim_ts=FIM)}, inicio_ts=FIM - 3600)
    em_uso = Trava(cfg.db.parent / "bot.lock")
    assert em_uso.adquirir()  # como se outro bot já estivesse rodando
    with pytest.raises(ErroIQ, match="Já existe um bot rodando"):
        app.operar(cfg, banco, cliente, lambda _: None, minutos=5)
    assert banco.ultima_pesquisa() is None  # nem começou a atualizar o mercado
    em_uso.liberar()
    app.operar(cfg, banco, cliente, lambda _: None, minutos=5)  # com a trava livre, roda
    assert Trava(cfg.db.parent / "bot.lock").adquirir()  # e solta a trava ao terminar
