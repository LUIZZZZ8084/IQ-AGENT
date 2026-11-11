"""Comandos de terminal que não precisam da corretora."""
import sqlite3

import pytest

from conftest import FIM
from iq_agent import app, cli
from iq_agent.banco import Banco
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import Ativo


@pytest.fixture
def projeto(cfg, monkeypatch):
    """Um 'projeto' temporário com velas sintéticas, no lugar do config.toml e do banco reais."""
    cfg.pesquisa.modos, cfg.pesquisa.estrategias = ["M1>1"], ["sequencia"]
    banco = Banco(cfg.db)
    banco.inserir_serie("PLANT-OTC", gerar_velas(20 * 1440, semente=50, fim_ts=FIM, reversao_apos=3, forca=0.2))
    banco.salvar_ativos([Ativo(1, "PLANT-OTC", "turbo", 0.85, True)], FIM)
    banco.fechar()
    monkeypatch.setattr(cli, "carregar", lambda: cfg)
    monkeypatch.setattr(app, "preparar_logs", lambda cfg: None)
    monkeypatch.setattr("iq_agent.pesquisa.time.time", lambda: float(FIM))
    return cfg


def test_banco_somente_leitura_nao_aceita_escrita(tmp_path):
    caminho = tmp_path / "b.db"
    banco = Banco(caminho)
    banco.inserir_velas("X", 60, [(60, 1, 1, 1, 1, 1)])
    banco.fechar()
    leitura = Banco(caminho, somente_leitura=True)
    assert leitura.intervalo_velas("X") == (60, 60, 1)
    with pytest.raises(sqlite3.OperationalError):
        leitura.inserir_velas("X", 60, [(120, 1, 1, 1, 1, 1)])
    leitura.fechar()


def test_pesquisar_em_ensaio_nao_grava_nada(projeto, capsys):
    assert cli.main(["pesquisar", "--ensaio"]) == 0
    saida = capsys.readouterr().out
    assert "Ensaio: nada é gravado" in saida and "Pesquisa concluída" in saida
    assert "As mais fortes" in saida and "PLANT-OTC M1>1 sequencia" in saida
    banco = Banco(projeto.db)
    assert banco.ultima_pesquisa() is None  # o plano do bot não mudou
    banco.fechar()
    assert not projeto.dir_modelos.exists()  # os modelos do ensaio foram para uma pasta temporária


def test_pesquisar_normal_grava_plano_e_modelos(projeto, capsys):
    assert cli.main(["pesquisar"]) == 0
    assert "Pesquisa concluída" in capsys.readouterr().out
    banco = Banco(projeto.db)
    pesquisa = banco.ultima_pesquisa()
    banco.fechar()
    assert pesquisa is not None and pesquisa["resumo"]["aprovadas"] >= 1
    if projeto.ml.ativo and pesquisa["resumo"]["hipoteses"] > 3:
        assert any(h["estrategia"] == "modelo_ml" for h in pesquisa["hipoteses"])


def test_pesquisar_sem_banco_avisa(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli, "carregar", lambda: cfg)
    monkeypatch.setattr(app, "preparar_logs", lambda cfg: None)
    assert cli.main(["pesquisar", "--ensaio"]) == 1
    assert "Não há velas no banco" in capsys.readouterr().out
