import numpy as np

from conftest import FIM, velas_de_cores
from iq_agent import app
from iq_agent.backtest import aplicar, desfechos, sem_sobreposicao
from iq_agent.config import Pesquisa
from iq_agent.pesquisa import avaliar_modo, finalizar, limites_folds, montar_plano
from iq_agent.sintetico import gerar_velas
from iq_agent.velas import Velas


def test_desfecho_entra_na_abertura_seguinte_e_sai_no_fechamento():
    v = velas_de_cores("ABAAB")
    call, put, valido = desfechos(v, k=1)
    # sinal na vela 0 -> opera a vela 1 (baixa): call perde, put ganha
    assert (call[0], put[0]) == (-1, 1)
    assert (call[1], put[1]) == (1, -1)
    assert valido.tolist() == [True, True, True, True, False]  # a última não tem vela seguinte


def test_desfecho_com_expiracao_de_varias_velas():
    v = velas_de_cores("AABBB")  # sinal na vela 0: entra na abertura da 1, sai no fechamento da 3
    call, put, valido = desfechos(v, k=3)
    assert v.close[3] < v.open[1]
    assert (call[0], put[0]) == (-1, 1)
    assert valido.tolist() == [True, True, False, False, False]


def test_empate_devolve_e_buraco_invalida():
    v = velas_de_cores("ADA")
    call, put, _ = desfechos(v, k=1)
    assert (call[0], put[0]) == (0, 0)  # doji: fechou no preço de entrada
    furado = Velas(np.array([0, 60, 180]), v.open, v.high, v.low, v.close, v.volume, 60)
    assert desfechos(furado, k=1)[2].tolist() == [True, False, False]


def test_slippage_transforma_ganho_minimo_em_perda():
    v = velas_de_cores("AAAA")
    assert desfechos(v, k=1, slippage_frac=0.0)[0][0] == 1
    assert desfechos(v, k=1, slippage_frac=1.5)[0][0] == -1  # entrada pior que o corpo inteiro da vela


def test_sem_sobreposicao():
    sinal = np.array([1, 1, 0, -1, 1, 0, 1, 1], dtype=np.int8)
    assert sem_sobreposicao(sinal, 1) is sinal
    assert sem_sobreposicao(sinal, 3).tolist() == [1, 0, 0, -1, 0, 0, 1, 0]


def test_aplicar_separa_ganhos_perdas_e_empates():
    v = velas_de_cores("ABADA")
    call, put, valido = desfechos(v, k=1)
    sinal = np.array([1, 1, 1, -1, 1], dtype=np.int8)
    efetivo, ganhou, perdeu, empatou = aplicar(sinal, call, put, valido, 1)
    assert efetivo.tolist() == [1, 1, 1, -1, 0]
    assert ganhou.tolist() == [False, True, False, False, False]
    assert perdeu.tolist() == [True, False, False, True, False]
    assert empatou.tolist() == [False, False, True, False, False]


def test_limites_dos_folds():
    dia = 86400
    limites = limites_folds(0, 30 * dia, 5, 3.0)
    assert limites.tolist() == [15 * dia, 18 * dia, 21 * dia, 24 * dia, 27 * dia, 30 * dia]
    assert len(limites_folds(0, 10 * dia, 5, 3.0)) == 3  # histórico curto: menos janelas
    assert limites_folds(0, 5 * dia, 5, 3.0) is None  # curto demais para treinar e testar


def test_mercado_aleatorio_nao_aprova_e_padrao_plantado_e_encontrado():
    pq = Pesquisa()
    aleatorio = avaliar_modo("ALEAT", gerar_velas(20 * 1440, semente=1, fim_ts=FIM), 1, 1, 0.85, pq,
                             ["sequencia", "rsi_reversao", "mhi", "padrao_cores"])
    plantado = avaliar_modo("PLANT", gerar_velas(20 * 1440, semente=2, fim_ts=FIM, reversao_apos=3, forca=0.2),
                            1, 1, 0.85, pq, ["sequencia", "rsi_reversao", "mhi", "padrao_cores"])
    hipoteses = aleatorio + plantado
    finalizar(hipoteses, pq)
    aprovadas = {(h.ativo, h.estrategia) for h in hipoteses if h.aprovado}
    assert ("PLANT", "sequencia") in aprovadas
    assert not any(ativo == "ALEAT" for ativo, _ in aprovadas)
    seq = next(h for h in plantado if h.estrategia == "sequencia")
    assert seq.config["params"]["modo"] == "reversao"
    assert seq.n > 1000 and 0.55 < seq.taxa < 0.65  # plantado: ~60% de reversão
    assert seq.wilson_inf > seq.breakeven and seq.q_valor < 0.01

    plano = montar_plano(hipoteses, max_exploracao=2)
    assert plano[0]["tipo"] == "aprovado" and plano[0]["ativo"] == "PLANT"
    assert all(e["tipo"] == "exploracao" for e in plano if not (e["ativo"] == "PLANT" and e["estrategia"] in
                                                               {h.estrategia for h in hipoteses if h.aprovado}))
    assert sum(e["tipo"] == "exploracao" for e in plano) <= 2


def test_pouco_historico_nao_gera_hipoteses():
    assert avaliar_modo("X", gerar_velas(300, semente=1, fim_ts=FIM), 1, 1, 0.85, Pesquisa(), ["sequencia"]) == []
    assert avaliar_modo("X", gerar_velas(4 * 1440, semente=1, fim_ts=FIM), 1, 1, 0.85, Pesquisa(), ["sequencia"]) == []


def test_autoteste_completo_passa():
    assert app.autoteste(log=lambda _: None)
