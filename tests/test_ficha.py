"""Ficha das estratégias: memória das pesquisas anteriores e placar ao vivo."""
from dataclasses import asdict

import pytest

from conftest import FIM
from iq_agent import ficha
from iq_agent.config import Config
from iq_agent.estatistica import breakeven, z_unilateral
from iq_agent.executor import Entrada, Executor
from iq_agent.pesquisa import Hipotese, montar_plano, preencher_estatisticas, usar_fichas

DIA = 86400
Z = z_unilateral(0.95)
PAYOUT = 0.85


def hipotese(ativo="EURUSD", vitorias=200, derrotas=130, aprovado=True, modo="M1>5", estrategia="engolfo"):
    h = Hipotese(ativo=ativo, modo=modo, estrategia=estrategia, payout=PAYOUT, breakeven=breakeven(PAYOUT),
                 folds=5, folds_ativos=5, vitorias=vitorias, derrotas=derrotas,
                 config={"params": {}, "sessao": "todas", "vol": "qualquer"})
    preencher_estatisticas(h, Z, Config().pesquisa)
    h.aprovado = aprovado
    h.situacao = "aprovada" if aprovado else "sem vantagem comprovada"
    return h


def gravar_pesquisa(banco, ts, hipoteses):
    registro = [asdict(h) | {"n": h.n} for h in hipoteses]
    banco.salvar_pesquisa(f"run-{ts}", ts, {}, registro, {"entradas": []})


def operacoes(banco, ativo, vitorias, derrotas, quando=FIM - DIA, modo="M1>5", estrategia="engolfo",
              conta="PRACTICE"):
    for i in range(vitorias + derrotas):
        ganhou = i < vitorias
        banco.registrar_operacao(ts_sinal=quando + i * 60, conta=conta, tipo="exploracao", ativo=ativo, modo=modo,
                                 estrategia=estrategia, config="{}", direcao="call", valor=1.0, payout=PAYOUT,
                                 status="win" if ganhou else "loss", lucro=0.85 if ganhou else -1.0)


# ------------------------------------------------------------------ memória das pesquisas
def test_aprovacao_de_um_dia_so_fica_aguardando_confirmacao(banco):
    """O caso do NZDCAD-op em 23/11: aprovada num dia e reprovada no seguinte."""
    gravar_pesquisa(banco, FIM - 2 * DIA, [hipotese(aprovado=False)])
    hoje = hipotese()
    fichas, dias = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    efeito = ficha.aplicar([hoje], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert len(dias) == 1 and efeito["confirmando"] == [hoje]
    assert not hoje.aprovado and hoje.ficha["confirmando"]
    assert "precisa se repetir em mais 1 pesquisa" in hoje.situacao


def test_aprovacao_repetida_em_outro_dia_vale(banco):
    gravar_pesquisa(banco, FIM - DIA, [hipotese()])
    hoje = hipotese()
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    efeito = ficha.aplicar([hoje], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert hoje.aprovado and not efeito["confirmando"]
    assert hoje.ficha["aprovacoes_seguidas"] == 1 and hoje.ficha["aprovada_hoje"]
    assert "aprovada em 2 pesquisas seguidas" in ficha.descrever(hoje.ficha)


def test_pesquisa_refeita_no_mesmo_dia_nao_confirma(banco):
    """Duas pesquisas no mesmo dia olham quase os mesmos dados: a segunda não confirma a primeira."""
    gravar_pesquisa(banco, FIM - 3600, [hipotese()])
    hoje = hipotese()
    fichas, dias = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    ficha.aplicar([hoje], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert dias == [] and not hoje.aprovado and hoje.ficha["confirmando"]


def test_vale_a_ultima_pesquisa_de_cada_dia_e_falta_conta_como_nao_aprovada(banco):
    gravar_pesquisa(banco, FIM - 3 * DIA, [hipotese()])
    gravar_pesquisa(banco, FIM - 2 * DIA - 7200, [hipotese()])
    gravar_pesquisa(banco, FIM - 2 * DIA, [hipotese(aprovado=False)])  # a última daquele dia
    gravar_pesquisa(banco, FIM - DIA, [hipotese(ativo="OUTRO")])  # EURUSD não apareceu nesse dia
    fichas, dias = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    f = fichas[("EURUSD", "M1>5", "engolfo")]
    assert len(dias) == 3 and [d["aprovado"] for d in f.historico] == [True, False, False]
    assert f.aprovacoes_seguidas() == 0 and f.dias_acima() == 2


def test_sem_confirmacao_exigida_nada_muda(banco):
    hoje = hipotese()
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    ficha.aplicar([hoje], fichas, dias_confirmacao=1, z=Z, min_ops_abaixo=30)
    assert hoje.aprovado


# ------------------------------------------------------------------ placar ao vivo
def test_placar_ao_vivo_bem_pior_que_o_teste_tira_do_plano(banco):
    """Perto do caso da TAOUSD-OTC (57% no teste e 5 de 15 ao vivo, chance de 5,6% por azar):
    com mais uma perda, 5 de 16, a chance cai para 3,4% e ela sai do plano."""
    operacoes(banco, "TAOUSD-OTC", vitorias=5, derrotas=11)
    h = hipotese(ativo="TAOUSD-OTC", vitorias=185, derrotas=139, aprovado=False)  # 57,1% no teste
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    efeito = ficha.aplicar([h], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert efeito["fora"] == [h] and "5 de 16" in h.ficha["fora"]
    assert montar_plano([h], max_exploracao=6) == []


def test_azar_pequeno_ao_vivo_nao_tira_do_plano(banco):
    operacoes(banco, "SMH", vitorias=4, derrotas=5)  # 9 operações: pouco para julgar
    h = hipotese(ativo="SMH", aprovado=False)
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    efeito = ficha.aplicar([h], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert not efeito["fora"] and len(montar_plano([h], max_exploracao=6)) == 1


def test_operacoes_antigas_saem_da_conta(banco):
    operacoes(banco, "TAOUSD-OTC", vitorias=5, derrotas=10, quando=FIM - 90 * DIA)
    h = hipotese(ativo="TAOUSD-OTC", vitorias=185, derrotas=139, aprovado=False)
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    ficha.aplicar([h], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert not h.ficha["fora"] and h.ficha["vivo"][:2] == [0, 0]


def test_resultado_ao_vivo_entra_no_pior_caso_e_na_ordem(banco):
    operacoes(banco, "BOM", vitorias=14, derrotas=4)
    operacoes(banco, "RUIM", vitorias=7, derrotas=9)
    bom, ruim = hipotese(ativo="BOM", aprovado=False), hipotese(ativo="RUIM", vitorias=210, aprovado=False)
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    ficha.aplicar([bom, ruim], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    assert bom.ficha["pior_caso_combinado"] > bom.wilson_inf
    assert ruim.ficha["pior_caso_combinado"] < ruim.wilson_inf
    plano = montar_plano([ruim, bom], max_exploracao=6)
    assert [e["ativo"] for e in plano] == ["BOM", "RUIM"]  # sem o ao vivo, RUIM viria primeiro
    assert plano[0]["p_inf"] == pytest.approx(bom.ficha["pior_caso_combinado"])
    assert plano[0]["p_inf_teste"] == pytest.approx(bom.wilson_inf)


def test_confirmando_vem_primeiro_entre_as_candidatas(banco):
    melhor = hipotese(ativo="MELHOR", vitorias=230, aprovado=False)
    nova = hipotese(ativo="NOVA")
    fichas, _ = ficha.montar_fichas(banco, FIM, dias_vivo=60)
    ficha.aplicar([melhor, nova], fichas, dias_confirmacao=2, z=Z, min_ops_abaixo=30)
    plano = montar_plano([melhor, nova], max_exploracao=6)
    assert [(e["ativo"], e["tipo"]) for e in plano] == [("NOVA", "exploracao"), ("MELHOR", "exploracao")]


def test_operacoes_do_modelo_contam_para_o_grupo(banco):
    operacoes(banco, "EURUSD-OTC", 3, 1, estrategia="modelo_ml")
    operacoes(banco, "GBPUSD-OTC", 2, 2, estrategia="modelo_ml")
    placar = ficha.placar_ao_vivo(banco, 0)
    assert placar[("OTC", "M1>5", "modelo_ml")][:2] == [5, 3]


def test_julgar_ao_vivo():
    assert ficha.julgar_ao_vivo(5, 16, 0.571, 0.54, Z, 30) is not None
    assert ficha.julgar_ao_vivo(5, 15, 0.571, 0.54, Z, 30) is None  # 5,6% de chance por azar: ainda fica
    assert ficha.julgar_ao_vivo(8, 15, 0.571, 0.54, Z, 30) is None
    assert ficha.julgar_ao_vivo(0, 0, 0.6, 0.54, Z, 30) is None
    # sem taxa de teste, vale só a regra antiga: muitas operações e melhor caso abaixo do necessário
    assert ficha.julgar_ao_vivo(12, 40, None, 0.54, Z, 30) is not None
    assert ficha.julgar_ao_vivo(12, 29, None, 0.54, Z, 30) is None


# ------------------------------------------------------------------ registro e executor
def test_usar_fichas_explica_no_registro(banco):
    cfg = Config()
    cfg.pesquisa.dias_confirmacao = 2
    gravar_pesquisa(banco, FIM - DIA, [hipotese(aprovado=False)])
    operacoes(banco, "TAOUSD-OTC", vitorias=5, derrotas=11)
    hoje, ruim = hipotese(), hipotese(ativo="TAOUSD-OTC", vitorias=185, derrotas=139, aprovado=False)
    mensagens = []
    memoria = usar_fichas(banco, cfg, [hoje, ruim], FIM, mensagens.append)
    texto = "\n".join(mensagens)
    assert memoria["dias"] == 1 and "Memória: comparei com as pesquisas de 1 dia(s)" in texto
    assert "Engolfo em EURUSD M1>5: passou no teste hoje" in texto or "EURUSD M1>5: passou no teste hoje" in texto
    assert "TAOUSD-OTC M1>5: fora do plano porque ao vivo acertou 5 de 16" in texto


def test_executor_usa_a_mesma_regra_e_mostra_a_ficha(cfg, banco):
    from iq_agent.simulado import ClienteSimulado
    from iq_agent.sintetico import gerar_velas

    operacoes(banco, "EURUSD", vitorias=5, derrotas=11, estrategia="sequencia", modo="M1>1")
    base = {"tipo": "exploracao", "ativo": "EURUSD", "modo": "M1>1", "tf": 60, "exp_min": 1,
            "estrategia": "sequencia", "params": {"n": 3}, "sessao": "todas", "vol": "qualquer",
            "payout": PAYOUT, "breakeven": breakeven(PAYOUT), "p_inf": 0.5, "taxa": 0.57,
            "ficha": {"vivo": [5, 10, 0, -6.5], "dias_vistos": 3, "dias_acima": 2}}
    outra = base | {"ativo": "GBPUSD", "ficha": {"vivo": [3, 1, 0, 1.55], "dias_vistos": 2, "dias_acima": 2}}
    cliente = ClienteSimulado({"EURUSD": gerar_velas(1000, semente=1), "GBPUSD": gerar_velas(1000, semente=2)})
    mensagens = []
    ex = Executor(cfg, banco, cliente, {"entradas": [base, outra]}, log=mensagens.append, exploracao=True)
    entradas = ex.entradas(FIM)
    assert [e.ativo for e in entradas] == ["GBPUSD"]
    assert any("Suspensa por desempenho ao vivo: EURUSD" in m and "5 de 16" in m for m in mensagens)
    ex._listar(entradas)
    assert any("ao vivo 3 de 4, +1.55" in m and "acima do necessário em 2 de 2" in m for m in mensagens)
    assert Entrada.de_dict(outra).taxa == 0.57
