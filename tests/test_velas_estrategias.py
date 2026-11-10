import numpy as np
import pytest

from conftest import FIM, velas_de_cores
from iq_agent import indicadores as ind
from iq_agent.estrategias import ESTRATEGIAS, mhi, padrao_cores, sequencia
from iq_agent.filtros import mascara_sessao, mascara_vol, razao_vol, sessao_inclui
from iq_agent.sintetico import gerar_velas
from iq_agent.velas import Velas


def test_de_linhas_ordena_e_remove_repetidas():
    v = Velas.de_linhas([(120, 1, 2, 0, 1.5, 9), (60, 1, 1, 1, 1, 1), (120, 3, 4, 2, 3.5, 7)])
    assert v.ts.tolist() == [60, 120]
    assert v.close.tolist() == [1.0, 3.5]  # ficou a última ocorrência do timestamp repetido
    assert len(Velas.de_linhas([])) == 0


def test_fechadas_ate_descarta_a_vela_em_formacao():
    v = velas_de_cores("AAAA", inicio=600)
    assert len(v.fechadas_ate(600 + 180)) == 3  # a 4ª vela ainda está aberta
    assert len(v.fechadas_ate(600 + 240)) == 4
    assert len(v.fechadas_ate(600 + 59)) == 0


def test_reamostrar_para_5_minutos():
    v = velas_de_cores("AABAB" + "BBBBA" + "AA", inicio=FIM)  # FIM é múltiplo de 300
    v5 = v.reamostrar(300)
    assert len(v5) == 2  # o terceiro grupo está incompleto
    assert v5.ts.tolist() == [FIM, FIM + 300]
    assert v5.open[0] == v.open[0] and v5.close[0] == v.close[4]
    assert v5.high[0] == v.high[:5].max() and v5.low[1] == v.low[5:10].min()
    assert v5.volume[0] == 5


def test_reamostrar_descarta_grupo_com_buraco():
    v = velas_de_cores("AAAAA" + "AAAAA", inicio=FIM)
    furado = Velas(np.delete(v.ts, 7), np.delete(v.open, 7), np.delete(v.high, 7), np.delete(v.low, 7),
                   np.delete(v.close, 7), np.delete(v.volume, 7), 60)
    assert furado.reamostrar(300).ts.tolist() == [FIM]


def test_rsi_extremos():
    subindo = np.linspace(1.0, 2.0, 60)
    assert ind.rsi(subindo, 14)[-1] == pytest.approx(100.0)
    assert ind.rsi(subindo[::-1].copy(), 14)[-1] == pytest.approx(0.0)
    assert ind.rsi(np.ones(60), 14)[-1] == pytest.approx(50.0)
    assert np.isnan(ind.rsi(subindo, 14)[5])  # sem dados suficientes ainda


def test_sequencia_reversao_e_continuacao():
    v = velas_de_cores("BAAAB")
    assert sequencia(v, n=3, modo="reversao").tolist() == [0, 0, 0, -1, 0]
    assert sequencia(v, n=3, modo="continuacao").tolist() == [0, 0, 0, 1, 0]
    assert sequencia(velas_de_cores("AADAA"), n=3).tolist() == [0, 0, 0, 0, 0]  # doji quebra a sequência


def test_mhi_aposta_na_minoria_no_inicio_do_bloco():
    # bloco de 5 velas alinhado ao relógio; as 3 últimas são A, A, B -> maioria alta
    # no segundo bloco as 3 últimas são todas de alta: a minoria continua sendo baixa (PUT)
    v = velas_de_cores("BBAAB" + "AAAAA", inicio=FIM)
    assert mhi(v, entrada=1, modo="minoria").tolist() == [0, 0, 0, 0, -1, 0, 0, 0, 0, -1]
    assert mhi(v, entrada=1, modo="maioria")[4] == 1
    # entrada=2 emite o mesmo sinal uma vela depois (para entrar na 2ª vela do bloco)
    assert mhi(v, entrada=2, modo="minoria").tolist() == [0, 0, 0, 0, 0, -1, 0, 0, 0, 0]


def test_padrao_cores_reconhece_a_sequencia_exata():
    v = velas_de_cores("AABAABBA")
    assert padrao_cores(v, padrao="AAB", direcao="call").tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert padrao_cores(v, padrao="AAB", direcao="put").tolist() == [0, 0, -1, 0, 0, -1, 0, 0]
    assert padrao_cores(v, padrao="BBBB").sum() == 0


@pytest.mark.parametrize("nome", sorted(ESTRATEGIAS))
def test_estrategias_nao_olham_o_futuro(nome):
    """O sinal de uma vela não pode mudar quando chegam velas novas."""
    completo = gerar_velas(900, semente=11, fim_ts=FIM)
    parcial = completo.fatia(0, 600)
    est = ESTRATEGIAS[nome]
    for params in est.parametros()[:12]:
        todo = est.func(completo, **params)
        parte = est.func(parcial, **params)
        assert set(np.unique(todo)) <= {-1, 0, 1}
        assert todo[:600].tolist() == parte.tolist(), f"{nome} {params} usa dados futuros"


def test_todas_as_estrategias_geram_algum_sinal():
    v = gerar_velas(3000, semente=5, fim_ts=FIM)
    for nome, est in ESTRATEGIAS.items():
        total = sum(int(np.abs(est.func(v, **p)).sum()) for p in est.parametros())
        assert total > 0, nome


def test_filtro_de_sessao_usa_a_hora_da_entrada():
    # vela das 06:59 UTC: a entrada acontece às 07:00, já na sessão de Londres
    base = (FIM // 86400) * 86400
    v = velas_de_cores("AA", inicio=base + 6 * 3600 + 59 * 60)
    assert mascara_sessao(v, "londres").tolist() == [True, True]
    assert mascara_sessao(v, "asia").tolist() == [False, False]
    assert sessao_inclui("londres", base + 7 * 3600) and not sessao_inclui("londres", base + 12 * 3600)
    assert sessao_inclui("todas", base)


def test_filtro_de_volatilidade_ignora_aquecimento():
    razao = razao_vol(gerar_velas(400, semente=3, fim_ts=FIM))
    assert np.isnan(razao[:20]).all()
    alta, baixa = mascara_vol(razao, "alta"), mascara_vol(razao, "baixa")
    assert not (alta & baixa).any()
    assert not alta[:20].any() and not baixa[:20].any()
    assert mascara_vol(razao, "qualquer").all()
