import time

import pytest

from conftest import FIM, velas_de_cores
from iq_agent import app
from iq_agent.cenario import ler
from iq_agent.coleta import coletar, escolher_ativos
from iq_agent.config import Coleta
from iq_agent.pesquisa import pesquisar
from iq_agent.prontidao import avaliar
from iq_agent.relatorio import gerar, montar_html
from iq_agent.simulado import ClienteSimulado
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import Ativo, ErroIQ

DIA = 86400


# ------------------------------------------------------------------ coleta
def test_escolher_ativos():
    ativos = [Ativo(1, "EURUSD", "turbo", 0.85, True), Ativo(2, "GBPUSD", "turbo", 0.60, True),
              Ativo(3, "EURUSD-OTC", "turbo", 0.90, False), Ativo(4, "USDJPY", "turbo", 0.80, False),
              Ativo(5, "EURUSD", "binary", 0.95, True)]
    assert [a.nome for a in escolher_ativos(ativos, Coleta())] == ["EURUSD"]  # só turbo, aberto, payout ok
    nomes = [a.nome for a in escolher_ativos(ativos, Coleta(), {"EURUSD-OTC", "USDJPY"})]
    assert nomes == ["EURUSD", "EURUSD-OTC", "USDJPY"]  # fechados só entram se já tinham histórico
    assert [a.nome for a in escolher_ativos(ativos, Coleta(incluir_otc=False), {"EURUSD-OTC"})] == ["EURUSD"]
    assert [a.nome for a in escolher_ativos(ativos, Coleta(ativos=["gbpusd"]))] == ["GBPUSD"]
    assert len(escolher_ativos(ativos, Coleta(max_ativos=1, payout_minimo=0.0), {"EURUSD-OTC"})) == 1


def test_coleta_grava_so_velas_fechadas_e_depois_so_as_novas(cfg, banco):
    serie = gerar_velas(6 * 1440, semente=4, fim_ts=FIM)
    cfg.coleta.dias_historico = 3
    cliente = ClienteSimulado({"EURUSD": serie}, inicio_ts=FIM - DIA + 30)  # 30 s dentro de uma vela
    r = coletar(banco, cfg, cliente, log=lambda _: None)
    primeiro, ultimo, n = banco.intervalo_velas("EURUSD")
    assert r["ativos"] == 1 and r["falhas"] == 0 and n == r["velas_novas"]
    assert ultimo == FIM - DIA - 60  # a vela em formação ficou de fora
    assert primeiro >= FIM - DIA + 30 - 3 * DIA and n == pytest.approx(3 * 1440, abs=2)
    assert banco.ativos("turbo")["EURUSD"].payout == 0.85

    cliente.dormir_ate(FIM - DIA + 3600 + 5)
    r2 = coletar(banco, cfg, cliente, log=lambda _: None)
    assert 55 <= r2["velas_novas"] <= 61  # só a hora que passou
    v = banco.carregar_velas("EURUSD")
    assert (v.ts[1:] - v.ts[:-1] == 60).all()  # sem buracos nem repetidas


def test_coleta_interrompida_completa_o_comeco_depois(cfg, banco):
    serie = gerar_velas(6 * 1440, semente=4, fim_ts=FIM)
    cfg.coleta.dias_historico = 3
    agora = FIM - DIA
    banco.inserir_serie("EURUSD", serie.fatia(-1440 - 500, -1440))  # só as últimas 500 velas chegaram
    coletar(banco, cfg, ClienteSimulado({"EURUSD": serie}, inicio_ts=agora), log=lambda _: None)
    primeiro, _, n = banco.intervalo_velas("EURUSD")
    assert primeiro <= agora - 3 * DIA + 60 and n == pytest.approx(3 * 1440, abs=2)


def test_coleta_sem_ativos_avisa():
    class SemAtivos(ClienteSimulado):
        def ativos(self):
            return []

    from iq_agent.banco import Banco
    from iq_agent.config import Config

    with pytest.raises(ErroIQ, match="nenhum ativo"):
        coletar(Banco(":memory:"), Config(), SemAtivos({"X": gerar_velas(10, 1, fim_ts=FIM)}), log=lambda _: None)


# ------------------------------------------------------------------ prontidão para a conta real
def registrar(banco, quando, status, tipo="aprovado", conta="PRACTICE", lucro=None):
    banco.registrar_operacao(ts_sinal=int(quando), conta=conta, tipo=tipo, ativo="EURUSD", modo="M1>1",
                             estrategia="sequencia", config={}, direcao="call", valor=10.0, payout=0.85,
                             status=status, lucro=lucro if lucro is not None else (8.5 if status == "win" else -10.0))


def plano_aprovado(banco, agora):
    banco.salvar_pesquisa("r1", agora, {}, [], {"entradas": [{"tipo": "aprovado"}]})


def test_conta_real_bloqueada_por_padrao(cfg, banco):
    pode, explicacao, itens = app.conferir_real(cfg, banco, time.time())
    assert not pode and "desligada" in explicacao and itens == []


def test_prontidao_exige_historico_suficiente_e_lucrativo(cfg, banco):
    agora = float(FIM)
    cfg.real.permitido = True
    cfg.real.min_operacoes_demo, cfg.real.min_dias_demo = 40, 3
    pronto, itens = avaliar(banco, cfg, agora)
    assert not pronto and not any(i.ok for i in itens)

    for dia in range(4):  # 60 operações em 4 dias, 70% de acerto
        for i in range(15):
            registrar(banco, agora - dia * DIA - i * 120, "win" if i % 10 < 7 else "loss")
    plano_aprovado(banco, agora - 3600)
    pronto, itens = avaliar(banco, cfg, agora)
    assert pronto and all(i.ok for i in itens)
    pode, explicacao, _ = app.conferir_real(cfg, banco, agora)
    assert pode and "cumpridos" in explicacao


def test_prontidao_ignora_exploracao_e_reprova_taxa_baixa(cfg, banco):
    agora = float(FIM)
    cfg.real.permitido = True
    cfg.real.min_operacoes_demo, cfg.real.min_dias_demo = 40, 3
    for dia in range(4):
        for i in range(15):
            registrar(banco, agora - dia * DIA - i * 120, "win", tipo="exploracao")  # não conta
            registrar(banco, agora - dia * DIA - i * 120, "win" if i % 2 else "loss")  # 47% de acerto
    plano_aprovado(banco, agora - 3600)
    pronto, itens = avaliar(banco, cfg, agora)
    por_nome = {i.nome: i for i in itens}
    assert not pronto
    assert por_nome["Operações na demo com estratégias aprovadas"].detalhe == "60 de 40"
    assert not por_nome["Taxa de acerto na demo acima do breakeven, com confiança"].ok
    assert not por_nome["Resultado positivo na demo"].ok

    pode, _, _ = app.conferir_real(cfg, banco, agora)
    assert not pode
    cfg.real.exigir_prontidao = False  # decisão explícita do usuário no config.toml
    pode, explicacao, _ = app.conferir_real(cfg, banco, agora)
    assert pode and "NÃO foram cumpridos" in explicacao


def test_plano_velho_nao_vale_para_a_prontidao_nem_para_a_real(cfg, banco):
    agora = float(FIM)
    plano_aprovado(banco, agora - 3 * DIA)
    assert not {i.nome: i for i in avaliar(banco, cfg, agora)[1]}["Plano atual com estratégia aprovada"].ok
    with pytest.raises(ErroIQ, match="atualize o mercado"):
        app.plano_para_operar(banco, agora, "REAL", lambda _: None)
    avisos = []
    assert app.plano_para_operar(banco, agora, "PRACTICE", avisos.append) == {"entradas": [{"tipo": "aprovado"}]}
    assert avisos  # na demo só avisa


# ------------------------------------------------------------------ cenário e relatório
def test_cenario_le_tendencia_e_sequencia():
    subida = gerar_velas(900, semente=8, fim_ts=FIM, vol=0.00002)
    subida.close += 0.00004 * (subida.ts - subida.ts[0]) / 60  # tendência forte de alta
    subida.open[1:] = subida.close[:-1]
    subida.high[:] = subida.close + 0.00001
    subida.low[:] = subida.open - 0.00001
    c = ler("X", subida)
    assert c.tendencia == "alta" and c.rsi > 70 and "tendência de alta" in c.leitura
    assert "sobrecomprado" in c.leitura
    alternado = ler("X", velas_de_cores("AB" * 200 + "BBBBB", inicio=FIM))
    assert alternado.sequencia == -6 and "6 velas seguidas de baixa" in alternado.leitura
    assert ler("X", gerar_velas(100, semente=8, fim_ts=FIM)) is None  # poucas velas


def test_relatorio_sem_pesquisa(cfg):
    pagina = montar_html(None, [], [], cfg, float(FIM))
    assert "Ainda sem pesquisa" in pagina and "<svg" not in pagina
    assert pagina.startswith("<!doctype html>")


def test_relatorio_completo(cfg, banco, tmp_path):
    corte = FIM - 2 * DIA
    series = {"EURUSD": gerar_velas(22 * 1440, semente=1, fim_ts=FIM),
              "EURUSD-OTC": gerar_velas(22 * 1440, semente=2, fim_ts=FIM, reversao_apos=3, forca=0.2)}
    for nome, v in series.items():
        banco.inserir_serie(nome, v.fechadas_ate(corte))
    banco.salvar_ativos([Ativo(1, "EURUSD", "turbo", 0.85, True), Ativo(2, "EURUSD-OTC", "turbo", 0.85, True)], corte)
    pesquisar(banco, cfg, log=lambda _: None, agora=corte)
    for i in range(30):
        registrar(banco, corte + i * 300, "win" if i % 3 else "loss")
    registrar(banco, corte + 9999, "win", conta="REAL")
    caminho = gerar(banco, cfg, agora=corte + DIA, caminho=tmp_path / "r.html")
    pagina = caminho.read_text(encoding="utf-8")
    for trecho in ("Plano do dia", "Hipóteses mais fortes", "Cenário atual dos ativos", "Conta demo", "Conta real",
                   "Todas as hipóteses", "EURUSD-OTC", "aprovada", 'id="pontos-demo"', 'id="pontos-real"'):
        assert trecho in pagina, trecho
    assert pagina.count("<svg") >= 3
    assert pagina.count('id="demo"') == 1 and pagina.count('id="pontos-demo"') == 1  # ids não se repetem


def test_relatorio_escapa_textos_vindos_de_fora(cfg):
    perigoso = "<img src=x onerror=alert(1)>"
    op = {"conta": "PRACTICE", "status": "win", "lucro": 1.0, "ts_sinal": FIM, "ativo": perigoso, "modo": "M1>1",
          "estrategia": perigoso, "direcao": "call", "tipo": "aprovado", "valor": 1.0, "payout": 0.85}
    pagina = montar_html(None, [op], [], cfg, float(FIM))
    assert perigoso not in pagina and "&lt;img src=x" in pagina
