"""Modelo de aprendizado de máquina: atributos, treino por grupo, plano e uso pelo executor.

Tudo com mercado sintético e pasta temporária: nenhum teste toca os modelos reais.
"""
import numpy as np
import pytest

from conftest import FIM
from iq_agent import ml
from iq_agent.config import Config, Ml, Pesquisa
from iq_agent.executor import Entrada, Executor, sinal_atual
from iq_agent.pesquisa import finalizar, montar_plano
from iq_agent.simulado import ClienteSimulado
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import Ativo

pytestmark = pytest.mark.skipif(not ml.disponivel(), reason="scikit-learn não instalado")

DIAS = 20


@pytest.fixture(scope="module")
def mercado():
    """Dois ativos OTC com padrão plantado e dois de mercado aberto puramente aleatórios."""
    series = {f"PLANT{i}-OTC": gerar_velas(DIAS * 1440, semente=50 + i, fim_ts=FIM, reversao_apos=3, forca=0.2)
              for i in range(2)}
    series |= {f"ALEAT{i}": gerar_velas(DIAS * 1440, semente=80 + i, fim_ts=FIM) for i in range(2)}
    return series


@pytest.fixture(scope="module")
def resultado(mercado, tmp_path_factory):
    """Roda a pesquisa do modelo uma vez para todos os testes do módulo."""
    pasta = tmp_path_factory.mktemp("ml") / "modelos"
    pq = Pesquisa(modos=["M1>1"])
    pesquisa = ml.PesquisaML(pq, Ml(), list(mercado), pasta)
    try:
        for nome, v in mercado.items():
            pesquisa.adicionar(nome, v, 0.85)
        hipoteses = pesquisa.avaliar()
    finally:
        pesquisa.fechar()
    finalizar(hipoteses, pq)
    return {h.ativo: h for h in hipoteses}, pasta, pesquisa.tmp


def test_grupos_de_ativos():
    assert ml.grupo_de("EURUSD-OTC") == "OTC" and ml.grupo_de("eurusd-otc") == "OTC"
    assert ml.grupo_de("US30:N") == "mercado" and ml.grupo_de("XAU/XAG") == "mercado"
    assert ml.rotulo_grupo("OTC") == "Grupo OTC"


def test_atributos_tem_formato_e_ficam_finitos_depois_do_aquecimento(mercado):
    v = mercado["ALEAT0"].fatia(0, 2000)
    x = ml.atributos(v)
    assert x.shape == (2000, len(ml.NOMES_ATRIBUTOS)) and x.dtype == np.float32
    assert np.isnan(x[: ml.AQUECIMENTO]).all()
    assert np.isfinite(x[ml.AQUECIMENTO:]).all()
    assert np.abs(x[ml.AQUECIMENTO:]).max() <= 20.0
    assert ml.atributos(v.fatia(0, 1)).shape == (1, len(ml.NOMES_ATRIBUTOS))


def test_atributos_nao_olham_o_futuro(mercado):
    """A linha de uma vela não pode mudar quando chegam velas novas."""
    completo = mercado["PLANT0-OTC"].fatia(0, 1500)
    parcial = completo.fatia(0, 900)
    np.testing.assert_array_equal(ml.atributos(completo)[:900], ml.atributos(parcial))


def test_sinais_por_limiar():
    p = np.array([0.39, 0.43, 0.50, 0.57, 0.61])  # longe das fronteiras exatas, onde o arredondamento decide
    assert ml.sinais(p, 0.56).tolist() == [-1, -1, 0, 1, 1]
    assert ml.sinais(p, 0.60).tolist() == [-1, 0, 0, 0, 1]


def test_modelo_acha_o_padrao_plantado_e_fica_de_fora_no_aleatorio(resultado):
    hipoteses, pasta, _ = resultado
    otc, mercado_aberto = hipoteses["Grupo OTC"], hipoteses["Grupo mercado aberto"]
    assert otc.aprovado and otc.estrategia == ml.NOME
    assert otc.n > 2000 and 0.55 < otc.taxa < 0.65  # plantado: ~60% de reversão
    assert otc.wilson_inf > otc.breakeven and otc.folds_ativos >= 3
    assert otc.config["params"]["limiar"] in ml.LIMIARES
    assert set(otc.detalhe["ativos"]) == {"PLANT0-OTC", "PLANT1-OTC"}
    assert {d["ativo"] for d in otc.detalhe["por_ativo"]} == {"PLANT0-OTC", "PLANT1-OTC"}
    assert (pasta / "OTC_1_1.pkl").exists()

    assert not mercado_aberto.aprovado and mercado_aberto.config is None
    assert mercado_aberto.n == 0 and mercado_aberto.situacao == "sem vantagem no treino"
    assert not (pasta / "mercado_1_1.pkl").exists()


def test_diagnostico_mostra_o_acerto_por_limiar_mesmo_quando_fica_de_fora(resultado):
    """Quem ficou de fora ainda informa quanto acertaria: é isso que mostra a distância até o necessário."""
    hipoteses = resultado[0]
    aleatorio = hipoteses["Grupo mercado aberto"].detalhe
    linhas = aleatorio["fora_da_amostra_por_limiar"]
    assert [d["limiar"] for d in linhas] == list(ml.LIMIARES)
    assert len(aleatorio["validacao_final_por_limiar"]) == len(ml.LIMIARES)
    amplo = ml.diagnostico_amplo(aleatorio)
    assert amplo["limiar"] == ml.LIMIARES[0] and amplo["n"] == max(d["n"] for d in linhas) > 2000
    assert 0.46 < amplo["taxa"] < 0.53  # mercado aleatório: cara ou coroa, abaixo do necessário
    # limiares mais exigentes têm poucas operações e oscilam: por isso o diagnóstico usa o mais amplo
    assert linhas[0]["n"] > 5 * linhas[2]["n"]

    # no padrão plantado o limiar amplo mistura entradas fortes e fracas: acerta menos que o limiar
    # escolhido (coluna "Acerto"), mas fica claramente acima do mercado aleatório
    plantado = ml.diagnostico_amplo(hipoteses["Grupo OTC"].detalhe)
    assert plantado["taxa"] > amplo["taxa"] + 0.02 and plantado["taxa"] < hipoteses["Grupo OTC"].taxa
    assert ml.diagnostico_amplo(None) is None and ml.diagnostico_amplo({}) is None


def test_arquivos_temporarios_sao_apagados(resultado):
    assert not resultado[2].exists()


def test_plano_transforma_o_grupo_em_entradas_por_ativo(resultado):
    otc = resultado[0]["Grupo OTC"]
    info = {"PLANT0-OTC": Ativo(1, "PLANT0-OTC", "turbo", 0.83, True),
            "PLANT1-OTC": Ativo(2, "PLANT1-OTC", "turbo", 0.87, True)}
    plano = montar_plano([otc], max_exploracao=3, info_ativos=info, max_ativos_modelo=10)
    assert [e["ativo"] for e in plano] == ["PLANT1-OTC", "PLANT0-OTC"]  # maior payout primeiro
    primeira = plano[0]
    assert primeira["tipo"] == "aprovado" and primeira["estrategia"] == ml.NOME and primeira["grupo"] == "Grupo OTC"
    assert primeira["payout"] == 0.87 and primeira["breakeven"] == pytest.approx(1 / 1.87)
    assert primeira["params"]["arquivo"].endswith("OTC_1_1.pkl") and primeira["p_inf"] == otc.wilson_inf
    assert len(montar_plano([otc], 3, info, max_ativos_modelo=1)) == 1
    fechado = info | {"PLANT1-OTC": Ativo(2, "PLANT1-OTC", "turbo", 0.87, False)}
    assert montar_plano([otc], 3, fechado, 10)[0]["ativo"] == "PLANT0-OTC"  # abertos têm prioridade


def test_sinal_ao_vivo_bate_com_o_do_historico(resultado, mercado):
    """O executor enxerga só as últimas 1.000 velas; o sinal precisa ser o mesmo do histórico inteiro."""
    otc = resultado[0]["Grupo OTC"]
    pacote = ml.carregar_modelo(otc.config["params"]["arquivo"])
    serie = mercado["PLANT0-OTC"]
    limiar = otc.config["params"]["limiar"]
    p_historico = ml.prever(pacote["modelo"], ml.atributos(serie))
    e = Entrada.de_dict({"tipo": "aprovado", "ativo": "PLANT0-OTC", "modo": "M1>1", "tf": 60, "exp_min": 1,
                         "estrategia": ml.NOME, "params": otc.config["params"], "sessao": "todas",
                         "vol": "qualquer", "breakeven": 0.54, "p_inf": 0.57})
    vistos = set()
    for t in range(5000, 5400):
        janela = serie.fatia(t - 999, t + 1)
        esperado = int(ml.sinais(p_historico[t: t + 1], limiar)[0])
        assert sinal_atual(e, janela) == esperado
        vistos.add(esperado)
    assert vistos == {-1, 0, 1}  # o trecho tem CALL, PUT e velas sem sinal


def test_executor_opera_com_o_modelo(resultado, mercado, cfg, banco):
    otc = resultado[0]["Grupo OTC"]
    plano = {"run_id": "ml", "entradas": montar_plano(
        [otc], 0, {n: Ativo(i + 1, n, "turbo", 0.85, True) for i, n in enumerate(sorted(mercado))}, 10)}
    cfg.risco.max_operacoes_dia = 12
    cliente = ClienteSimulado(mercado, inicio_ts=FIM - 1440 * 60)
    resumo = Executor(cfg, banco, cliente, plano, log=lambda _: None).rodar(minutos=600)
    ops = banco.operacoes()
    assert resumo["operacoes"] == 12 and {o["estrategia"] for o in ops} == {ml.NOME}
    assert {o["ativo"] for o in ops} <= {"PLANT0-OTC", "PLANT1-OTC"}
    assert all(o["tipo"] == "aprovado" and o["status"] in ("win", "loss", "equal") for o in ops)


def test_modelo_ausente_nao_derruba_o_bot(cfg, banco, mercado, tmp_path):
    entrada = {"tipo": "aprovado", "ativo": "PLANT0-OTC", "modo": "M1>1", "tf": 60, "exp_min": 1,
               "estrategia": ml.NOME, "params": {"limiar": 0.56, "arquivo": str(tmp_path / "modelos" / "nao_existe.pkl")},
               "sessao": "todas", "vol": "qualquer", "payout": 0.85, "breakeven": 0.54, "p_inf": 0.57}
    mensagens = []
    cliente = ClienteSimulado(mercado, inicio_ts=FIM - 1440 * 60)
    resumo = Executor(cfg, banco, cliente, {"entradas": [entrada]}, log=mensagens.append).rodar(minutos=30)
    assert resumo["operacoes"] == 0
    assert any("não encontrado em disco" in m for m in mensagens)


def test_carregar_modelo_so_aceita_a_pasta_de_modelos(tmp_path):
    import pickle

    fora = tmp_path / "qualquer.pkl"
    fora.write_bytes(pickle.dumps({"versao": ml.VERSAO, "modelo": object()}))
    assert ml.carregar_modelo(fora) is None  # fora de uma pasta "modelos": nem abre
    pasta = tmp_path / "modelos"
    pasta.mkdir()
    (pasta / "velho.pkl").write_bytes(pickle.dumps({"versao": ml.VERSAO + 1, "modelo": object()}))
    assert ml.carregar_modelo(pasta / "velho.pkl") is None  # outra versão dos atributos
    (pasta / "lixo.pkl").write_bytes(b"isto nao e um modelo")
    assert ml.carregar_modelo(pasta / "lixo.pkl") is None
    assert ml.carregar_modelo(pasta / "falta.pkl") is None and ml.carregar_modelo(None) is None
    (pasta / "ok.pkl").write_bytes(pickle.dumps({"versao": ml.VERSAO, "modelo": "x"}))
    assert ml.carregar_modelo(pasta / "ok.pkl")["modelo"] == "x"


def test_pesquisa_sem_salvar_nao_grava_plano(cfg, banco, mercado):
    from iq_agent.pesquisa import pesquisar

    cfg.pesquisa.modos, cfg.pesquisa.estrategias = ["M1>1"], ["sequencia"]
    for nome, v in mercado.items():
        banco.inserir_serie(nome, v)
    banco.salvar_ativos([Ativo(i + 1, n, "turbo", 0.85, True) for i, n in enumerate(sorted(mercado))], FIM)
    r = pesquisar(banco, cfg, log=lambda _: None, agora=FIM, salvar=False)
    assert banco.ultima_pesquisa() is None  # ensaio: o plano do bot não muda
    do_modelo = [h for h in r["hipoteses"] if h["estrategia"] == ml.NOME]
    assert {h["ativo"] for h in do_modelo} == {"Grupo OTC", "Grupo mercado aberto"}
    assert any(e["estrategia"] == ml.NOME and e["tipo"] == "aprovado" for e in r["plano"]["entradas"])
    assert (cfg.dir_modelos / "OTC_1_1.pkl").exists()


def test_modelo_desligado_na_configuracao(cfg, banco, mercado):
    from iq_agent.pesquisa import pesquisar

    cfg.ml.ativo = False
    cfg.pesquisa.modos, cfg.pesquisa.estrategias = ["M1>1"], ["sequencia"]
    banco.inserir_serie("PLANT0-OTC", mercado["PLANT0-OTC"])
    r = pesquisar(banco, cfg, log=lambda _: None, agora=FIM, salvar=False)
    assert not any(h["estrategia"] == ml.NOME for h in r["hipoteses"])
    assert not cfg.dir_modelos.exists()
