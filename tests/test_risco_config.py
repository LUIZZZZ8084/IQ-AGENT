import pytest

from iq_agent.config import Config, ErroConfig, Operacao, Risco, carregar, interpretar_modo
from iq_agent.risco import GestorRisco


def gestor(saldo=1000.0, **risco) -> GestorRisco:
    return GestorRisco(Risco(**risco), Operacao(), saldo)


def test_valor_respeita_o_teto_por_operacao():
    g = gestor()
    valor, _ = g.valor(1000.0, p_inf=0.80, payout=0.85)  # Kelly enorme, mas o teto é 1% do saldo
    assert valor == 10.0


def test_valor_usa_kelly_fracionado_quando_a_vantagem_e_pequena():
    g = gestor(risco_por_operacao=0.05, valor_maximo=1000.0)
    valor, motivo = g.valor(10_000.0, p_inf=0.56, payout=0.85)
    kelly = (0.56 * 1.85 - 1) / 0.85
    assert valor == pytest.approx(10_000 * kelly * 0.25, abs=0.01)
    assert valor < 0.05 * 10_000 and "Kelly" in motivo


def test_sem_vantagem_nao_opera():
    valor, motivo = gestor().valor(1000.0, p_inf=0.53, payout=0.85)  # breakeven é 54,05%
    assert valor == 0.0 and "vantagem" in motivo
    assert gestor().valor(1000.0, p_inf=None, payout=0.85)[0] == 0.0


def test_saldo_pequeno_demais_nao_opera():
    valor, motivo = gestor().valor(50.0, p_inf=0.7, payout=0.85)  # 1% de 50 = 0,50 < aposta mínima 1,00
    assert valor == 0.0 and "saldo" in motivo


def test_exploracao_usa_valor_fixo_minimo():
    g = gestor()
    assert g.valor(1000.0, None, 0.85, exploracao=True)[0] == 1.0
    assert g.valor(1000.0, None, 0.85, exploracao=True, aposta_min=2.0)[0] == 2.0


def test_valor_maximo_absoluto():
    g = gestor(risco_por_operacao=0.05, valor_maximo=7.0)
    assert g.valor(100_000.0, p_inf=0.9, payout=0.85)[0] == 7.0


def test_valor_nunca_aumenta_depois_de_perder():
    """Sem martingale: perder reduz o saldo e, com ele, o valor da próxima operação."""
    g = gestor(risco_por_operacao=0.02, valor_maximo=1000.0)
    saldo = 1000.0
    anterior, _ = g.valor(saldo, 0.62, 0.85)
    for _ in range(5):
        saldo -= anterior
        g.registrar("loss", -anterior, agora=0)
        atual, _ = g.valor(saldo, 0.62, 0.85)
        assert atual <= anterior
        anterior = atual


def test_para_no_limite_de_perda_diaria():
    g = gestor()
    g.registrar("loss", -30.0, agora=0)
    assert g.pode_operar(0).pode
    g.registrar("loss", -20.0, agora=0)  # total -50 = 5% de 1000
    decisao = g.pode_operar(0)
    assert not decisao.pode and decisao.encerrar_dia and "perda" in decisao.motivo


def test_para_no_limite_de_operacoes_e_na_meta():
    g = gestor(max_operacoes_dia=2)
    g.registrar("win", 1.0, 0)
    g.registrar("equal", 0.0, 0)
    assert g.pode_operar(0).encerrar_dia
    g = gestor(meta_diaria=0.01)
    g.registrar("win", 10.0, 0)
    assert g.pode_operar(0).encerrar_dia


def test_pausa_depois_de_perdas_seguidas():
    g = gestor(max_perdas_seguidas=3, pausa_apos_perdas_min=30, perda_maxima_diaria=0.5)
    for _ in range(2):
        g.registrar("loss", -1.0, agora=1000)
    assert g.pode_operar(1000).pode
    g.registrar("loss", -1.0, agora=1000)
    decisao = g.pode_operar(1000 + 60)
    assert not decisao.pode and not decisao.encerrar_dia
    assert g.pode_operar(1000 + 31 * 60).pode  # a pausa acabou


def test_aviso_de_pausa_e_o_mesmo_a_cada_minuto():
    """Com 'faltam N min' o texto mudava a cada minuto e o registro ganhava 29 linhas iguais."""
    g = gestor(max_perdas_seguidas=3, pausa_apos_perdas_min=30, perda_maxima_diaria=0.5)
    for _ in range(3):
        g.registrar("loss", -1.0, agora=1000)
    primeiro, depois = g.pode_operar(1000 + 60), g.pode_operar(1000 + 20 * 60)
    assert not primeiro.pode and primeiro.motivo == depois.motivo
    assert "pausa de 30 min após 3 perdas seguidas" in primeiro.motivo and "volto a operar às" in primeiro.motivo
    assert "faltam" not in primeiro.motivo


def test_vitoria_zera_a_contagem_de_perdas():
    g = gestor(perda_maxima_diaria=0.5)
    g.registrar("loss", -1.0, 0)
    g.registrar("loss", -1.0, 0)
    g.registrar("win", 0.85, 0)
    g.registrar("loss", -1.0, 0)
    assert g.pode_operar(1).pode


def test_estado_do_dia_e_reconstruido_do_diario():
    ops = [{"status": "win", "lucro": 8.5}, {"status": "loss", "lucro": -10.0}, {"status": "loss", "lucro": -10.0},
           {"status": "recusada", "lucro": 0.0}]
    g = GestorRisco.do_historico(Risco(), Operacao(), 988.5, ops)
    assert g.operacoes == 3 and g.lucro == pytest.approx(-11.5)
    assert g.saldo_inicial == pytest.approx(1000.0) and g.perdas_seguidas == 2


def test_interpretar_modo():
    assert interpretar_modo("M1>5") == (1, 5)
    assert interpretar_modo("m5>5") == (5, 5)
    for ruim in ("M1>15", "M5>3", "M2>2", "1>1", "M1-1"):
        with pytest.raises(ErroConfig):
            interpretar_modo(ruim)


def test_carregar_config(tmp_path):
    arquivo = tmp_path / "config.toml"
    arquivo.write_text('[risco]\nrisco_por_operacao = 0.02\nmax_operacoes_dia = 5\n[coleta]\nativos = ["EURUSD"]\n',
                       encoding="utf-8")
    cfg = carregar(arquivo)
    assert cfg.risco.risco_por_operacao == 0.02 and cfg.risco.max_operacoes_dia == 5
    assert cfg.coleta.ativos == ["EURUSD"]
    assert cfg.real.permitido is False  # a conta real nasce bloqueada
    assert isinstance(carregar(tmp_path / "nao_existe.toml"), Config)


@pytest.mark.parametrize("conteudo", [
    "[risco]\nrisco_por_operacao = 0.5\n",  # acima do teto de 5%
    "[risco]\nopcao_que_nao_existe = 1\n",
    "[secao_errada]\nx = 1\n",
    "[risco]\nmax_operacoes_dia = true\n",
    '[pesquisa]\nmodos = ["M1>30"]\n',
    "[real]\nvalor_maximo = 0\n",
])
def test_config_invalida_e_recusada(tmp_path, conteudo):
    arquivo = tmp_path / "config.toml"
    arquivo.write_text(conteudo, encoding="utf-8")
    with pytest.raises(ErroConfig):
        carregar(arquivo)


def test_config_do_repositorio_e_valida():
    cfg = carregar()
    assert cfg.real.permitido is False  # no arquivo do repositório a conta real continua bloqueada
