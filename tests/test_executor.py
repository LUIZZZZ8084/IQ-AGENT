"""Executor com corretora simulada: diário, limites de risco e escolha de conta."""
import threading

import pytest

from conftest import FIM
from iq_agent.config import Config
from iq_agent.executor import Entrada, Executor, conciliar, sinal_atual
from iq_agent.simulado import ClienteSimulado
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import FRASE_CONTA_REAL, ContaRealBloqueada, ErroIQ, Resultado

INICIO = FIM - 1440 * 60  # o "ao vivo" começa um dia antes do fim da série


def entrada(tipo="aprovado", estrategia="sequencia", params=None, **extra) -> dict:
    base = {"tipo": tipo, "ativo": "PLANT", "modo": "M1>1", "tf": 60, "exp_min": 1, "estrategia": estrategia,
            "params": params or {"n": 3, "modo": "reversao"}, "sessao": "todas", "vol": "qualquer",
            "payout": 0.85, "breakeven": 0.5405, "p_inf": 0.58, "taxa": 0.6, "n_teste": 1000}
    return base | extra


def plano(*entradas) -> dict:
    return {"run_id": "teste", "entradas": list(entradas)}


@pytest.fixture
def mercado():
    return {"PLANT": gerar_velas(3 * 1440, semente=21, fim_ts=FIM, reversao_apos=3, forca=0.2)}


@pytest.fixture
def cliente(mercado):
    c = ClienteSimulado(mercado, inicio_ts=INICIO)
    c.conectar()
    return c


def rodar(cfg, banco, cliente, plano_, minutos=240, **kw):
    mensagens = []
    ex = Executor(cfg, banco, cliente, plano_, log=mensagens.append, **kw)
    return ex.rodar(minutos=minutos), mensagens


def test_conciliar_mensagem_e_saldo():
    ganho = Resultado("win", 8.5, "socket-option-closed")
    assert conciliar(ganho, 10, 0.85, 1000, 1008.5) == ("win", 8.5, "socket-option-closed")
    assert conciliar(ganho, 10, 0.85, 1000, 990.0)[2].endswith("saldo divergente")
    sem = Resultado("desconhecido", 0.0, "sem-mensagem")
    assert conciliar(sem, 10, 0.85, 1000, 1008.5) == ("win", 8.5, "saldo")
    assert conciliar(sem, 10, 0.85, 1000, 990.0) == ("loss", -10.0, "saldo")
    assert conciliar(sem, 10, 0.85, 1000, 1000.0) == ("equal", 0.0, "saldo")
    assert conciliar(sem, 10, 0.85, 1000, 1500.0)[0] == "desconhecido"  # saldo mexido por fora
    assert conciliar(Resultado("win", float("nan"), "x"), 10, 0.85, 1000, 1008.5)[1] == 8.5


def test_sinal_atual_usa_a_mesma_funcao_do_backtest(mercado):
    from iq_agent.estrategias import sequencia

    v = mercado["PLANT"].fatia(0, 1500)
    e = Entrada.de_dict(entrada())
    assert sinal_atual(e, v) == int(sequencia(v, n=3, modo="reversao")[-1])


def test_opera_na_demo_e_registra_no_diario(cfg, banco, cliente):
    saldo_inicial = cliente.saldo_demo().valor
    resumo, _ = rodar(cfg, banco, cliente, plano(entrada()))
    ops = banco.operacoes()
    assert resumo["operacoes"] == len(ops) > 0
    assert {o["conta"] for o in ops} == {"PRACTICE"}
    assert all(o["status"] in ("win", "loss", "equal") and o["opcao_id"] for o in ops)
    assert all(0 < o["valor"] <= cfg.risco.risco_por_operacao * saldo_inicial for o in ops)
    assert all(o["atraso_s"] <= cfg.operacao.atraso_maximo_s for o in ops)
    assert all(o["expiracao"] == o["ts_sinal"] + 60 for o in ops)  # expira no fim da vela em que entrou
    assert sum(o["lucro"] for o in ops) == pytest.approx(resumo["lucro"], abs=0.01)
    assert cliente.saldo_demo().valor == pytest.approx(saldo_inicial + resumo["lucro"], abs=0.01)
    assert cliente._saldos[1] == 1000.0  # a conta real simulada ficou intocada


def test_respeita_o_limite_de_operacoes_do_dia(cfg, banco, cliente):
    cfg.risco.max_operacoes_dia = 4
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=600)
    assert resumo["operacoes"] == 4
    assert any("limite de 4 operações" in m for m in mensagens)


def test_estado_do_dia_sobrevive_a_reinicio(cfg, banco, cliente):
    cfg.risco.max_operacoes_dia = 3
    rodar(cfg, banco, cliente, plano(entrada()), minutos=300)
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=120)  # "reiniciou" no mesmo dia
    assert resumo["operacoes"] == 0 and len(banco.operacoes()) == 3
    assert any("limite de 3 operações" in m for m in mensagens)


def test_exploracao_so_entra_quando_pedida(cfg, banco, cliente):
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada(tipo="exploracao")))
    assert resumo["operacoes"] == 0 and any("Nada a operar" in m for m in mensagens)
    resumo, _ = rodar(cfg, banco, cliente, plano(entrada(tipo="exploracao")), exploracao=True)
    ops = banco.operacoes()
    assert resumo["operacoes"] > 0 and {o["tipo"] for o in ops} == {"exploracao"}
    assert {o["valor"] for o in ops} == {cfg.operacao.valor_exploracao}  # valor fixo mínimo


def test_simular_nao_envia_ordem(cfg, banco, cliente):
    saldo = cliente.saldo_demo().valor
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), simular=True)
    assert resumo["operacoes"] == 0 and banco.operacoes() == []
    assert cliente.saldo_demo().valor == saldo and cliente._seq == 0
    assert any("[simulação]" in m for m in mensagens)


def test_resultado_e_obtido_pelo_saldo_quando_a_mensagem_nao_chega(cfg, banco, mercado):
    c = ClienteSimulado(mercado, inicio_ts=INICIO, enviar_resultado=False)
    cfg.risco.max_operacoes_dia = 6
    resumo, _ = rodar(cfg, banco, c, plano(entrada()))
    ops = banco.operacoes()
    assert resumo["operacoes"] == 6 and {o["fonte"] for o in ops} == {"saldo"}
    assert {o["status"] for o in ops} <= {"win", "loss", "equal"}
    assert c.saldo_demo().valor == pytest.approx(10_000 + resumo["lucro"], abs=0.01)


def test_conta_real_sem_liberacao_e_recusada(cfg, banco, cliente):
    with pytest.raises(ErroIQ, match="conexão está na conta PRACTICE"):
        Executor(cfg, banco, cliente, plano(entrada()), conta="REAL", log=lambda _: None).rodar(minutos=30)
    with pytest.raises(ContaRealBloqueada):
        cliente.usar_conta_real("quero operar")
    assert banco.operacoes() == [] and cliente._saldos[1] == 1000.0


def test_conta_real_liberada_opera_so_aprovadas_com_teto_proprio(cfg, banco, cliente):
    cfg.real.valor_maximo = 2.0
    cliente.usar_conta_real(FRASE_CONTA_REAL)
    saldo_demo = cliente.saldo_demo().valor
    explor = entrada(tipo="exploracao", estrategia="padrao_cores", params={"padrao": "AAB", "direcao": "call"})
    resumo, _ = rodar(cfg, banco, cliente, plano(explor, entrada()), conta="REAL", exploracao=True)
    ops = banco.operacoes()
    assert resumo["operacoes"] > 0
    assert {o["conta"] for o in ops} == {"REAL"}
    assert {o["tipo"] for o in ops} == {"aprovado"}  # exploração nunca vai para a conta real
    assert max(o["valor"] for o in ops) <= 2.0  # teto da seção [real]
    assert cliente.saldo_demo().valor == saldo_demo  # a demo não foi tocada
    assert cliente._saldos[1] == pytest.approx(1000.0 + resumo["lucro"], abs=0.01)


def test_diarios_das_duas_contas_nao_se_misturam(cfg, banco, cliente):
    cfg.risco.max_operacoes_dia = 2
    rodar(cfg, banco, cliente, plano(entrada()))
    cliente.usar_conta_real(FRASE_CONTA_REAL)
    resumo, _ = rodar(cfg, banco, cliente, plano(entrada()), conta="REAL")
    assert resumo["operacoes"] == 2  # o limite diário é contado por conta
    assert len(banco.operacoes(conta="PRACTICE")) == 2 and len(banco.operacoes(conta="REAL")) == 2


def test_parar_interrompe_o_laco(cfg, banco, cliente):
    parar = threading.Event()
    parar.set()
    mensagens = []
    resumo = Executor(cfg, banco, cliente, plano(entrada()), log=mensagens.append, parar=parar).rodar(minutos=600)
    assert resumo["operacoes"] == 0 and any("Parado a pedido" in m for m in mensagens)


def test_estrategia_falhando_ao_vivo_e_suspensa(cfg, banco, cliente):
    cfg.operacao.min_ops_para_suspender = 30
    for i in range(40):  # 10 acertos em 40: muito abaixo do breakeven
        banco.registrar_operacao(ts_sinal=INICIO - 3600 + i, conta="PRACTICE", tipo="aprovado", ativo="PLANT",
                                 modo="M1>1", estrategia="sequencia", config={}, direcao="call", valor=1.0,
                                 payout=0.85, status="win" if i < 10 else "loss", lucro=0.0)
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()))
    assert resumo["operacoes"] == 0
    assert any("Suspensa por desempenho" in m for m in mensagens)


def test_payout_baixo_ou_ativo_fechado_nao_opera(cfg, banco, mercado):
    c = ClienteSimulado(mercado, payouts={"PLANT": 0.60}, inicio_ts=INICIO)
    resumo, mensagens = rodar(cfg, banco, c, plano(entrada()))
    assert resumo["operacoes"] == 0 and any("payout" in m for m in mensagens)


def test_entrada_de_ativo_desconhecido_e_ignorada(cfg, banco, cliente):
    resumo, _ = rodar(cfg, banco, cliente, plano(entrada(ativo="NAOEXISTE")), minutos=60)
    assert resumo["operacoes"] == 0


def test_modo_continuo_renova_o_plano_a_cada_dia(cfg, banco):
    from datetime import date

    serie = {"PLANT": gerar_velas(5 * 1440, semente=21, fim_ts=FIM, reversao_apos=3, forca=0.2)}
    c = ClienteSimulado(serie, inicio_ts=FIM - 4 * 1440 * 60)
    cfg.risco.max_operacoes_dia = 2
    chamadas, mensagens = [], []

    def renovar():
        chamadas.append(c.agora())
        return plano(entrada())

    ex = Executor(cfg, banco, c, plano(entrada()), log=mensagens.append, renovar_plano=renovar)
    ex.rodar(minutos=3 * 1440)
    por_dia: dict[date, int] = {}
    for o in banco.operacoes():
        dia = date.fromtimestamp(o["ts_sinal"])
        por_dia[dia] = por_dia.get(dia, 0) + 1
    assert len(chamadas) == 3  # três viradas de dia em três dias de execução
    assert len(por_dia) >= 3 and max(por_dia.values()) == 2  # o limite diário vale em cada dia
    assert any("Volto a operar amanhã" in m for m in mensagens)
    assert any("Plano do dia atualizado" in m for m in mensagens)


def test_modo_continuo_espera_o_dia_seguinte_quando_nao_ha_plano(cfg, banco, cliente):
    planos = iter([plano(entrada())])
    cfg.risco.max_operacoes_dia = 1
    mensagens = []
    ex = Executor(cfg, banco, cliente, plano(), log=mensagens.append,
                  renovar_plano=lambda: next(planos, None))
    resumo = ex.rodar(minutos=1400)
    assert any("aguardando a atualização de amanhã" in m for m in mensagens)
    assert resumo["operacoes"] == 1  # só operou depois da virada, com o plano novo


def test_falha_na_renovacao_mantem_o_plano_anterior(cfg, banco, cliente):
    def renovar():
        raise ErroIQ("corretora fora do ar")

    cfg.risco.max_operacoes_dia = 1
    mensagens = []
    Executor(cfg, banco, cliente, plano(entrada()), log=mensagens.append, renovar_plano=renovar).rodar(minutos=1400)
    assert any("sigo com o plano anterior" in m for m in mensagens)
    assert len(banco.operacoes()) == 2  # um por dia, antes e depois da virada


def test_conta_real_para_a_meia_noite_mesmo_no_modo_continuo(cfg, banco, cliente):
    cliente.usar_conta_real(FRASE_CONTA_REAL)
    cfg.risco.max_operacoes_dia = 1
    chamadas, mensagens = [], []
    ex = Executor(cfg, banco, cliente, plano(entrada()), conta="REAL", log=mensagens.append,
                  renovar_plano=lambda: chamadas.append(1) or plano(entrada()))
    ex.rodar(minutos=2000)
    assert chamadas == [] and len(banco.operacoes(conta="REAL")) == 1
    assert any("limite de 1 operações" in m for m in mensagens) or any("meia-noite" in m for m in mensagens)


def test_operar_atualiza_o_mercado_quando_nao_ha_plano(cfg, banco, cliente):
    from iq_agent import app

    mensagens = []
    app.operar(cfg, banco, cliente, mensagens.append, minutos=30)
    assert any("atualizando o mercado antes de operar" in m for m in mensagens)
    assert banco.ultima_pesquisa() is not None and banco.intervalo_velas("PLANT")[2] > 1000
    assert (cfg.dir_relatorios / "relatorio.html").exists()


def test_ordem_sem_confirmacao_e_conciliada_pelo_saldo(cfg, banco, cliente):
    """A ordem entra, mas a resposta da corretora se perde: o bot descobre pelo saldo e segue."""
    cliente.defeito_ordem = "sem_confirmacao"
    cfg.risco.max_operacoes_dia = 3
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=600)
    ops = banco.operacoes()
    assert resumo["operacoes"] == 3 and len(ops) == 3
    assert all(o["status"] in ("win", "loss", "equal") and o["opcao_id"] is None for o in ops)
    assert all("ordem sem confirmação" in o["fonte"] for o in ops)
    assert any("a ordem entrou" in m for m in mensagens)
    assert cliente.saldo_demo().valor == pytest.approx(10_000 + resumo["lucro"], abs=0.01)


def test_ordem_que_nao_entra_descansa_o_ativo_e_depois_para_o_bot(cfg, banco, cliente):
    cliente.defeito_ordem = "nao_entra"
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=900)
    ops = banco.operacoes()
    assert resumo["operacoes"] == 0 and cliente.saldo_demo().valor == 10_000
    assert [o["status"] for o in ops] == ["falhou"] * 5  # cinco tentativas e para
    intervalos = [b["ts_sinal"] - a["ts_sinal"] for a, b in zip(ops, ops[1:])]
    assert min(intervalos) >= 1800  # o ativo descansa 30 minutos entre as tentativas
    assert any("O saldo não mudou" in m for m in mensagens)
    assert any("ordens seguidas não foram aceitas" in m for m in mensagens)


def test_ordem_recusada_registra_o_motivo_e_nao_insiste(cfg, banco, cliente):
    cliente.defeito_ordem = "recusa"
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=45)
    ops = banco.operacoes()
    assert resumo["operacoes"] == 0 and 1 <= len(ops) <= 2  # no máximo uma nova tentativa em 45 minutos
    assert {o["status"] for o in ops} == {"recusada"} and "indisponível" in ops[0]["erro"]
    assert any("ordem recusada pela corretora (ativo indisponível" in m for m in mensagens)
    cliente.defeito_ordem = None  # o problema passou: uma nova execução opera normalmente
    resumo, _ = rodar(cfg, banco, cliente, plano(entrada()), minutos=120)
    assert resumo["operacoes"] > 0


def test_conta_invalida():
    with pytest.raises(ValueError):
        Executor(Config(), None, None, plano(), conta="TORNEIO")


def test_registro_nao_confunde_estrategias_vigiadas_com_operacoes(cfg, banco, cliente):
    """O usuário leu '6 entradas no plano' como 6 operações: o texto precisa deixar a diferença clara."""
    cfg.risco.max_operacoes_dia = 2
    outra = entrada(tipo="exploracao", estrategia="padrao_cores", params={"padrao": "AAB", "direcao": "call"},
                    sessao="ny")
    _, mensagens = rodar(cfg, banco, cliente, plano(entrada(), outra), exploracao=True, minutos=600)
    texto = "\n".join(mensagens)
    assert "Vigiando 2 estratégia(s) do plano" in texto and "entrada(s)" not in texto
    assert "não operações" in texto and "uma operação por vez" in texto
    assert "- Sequência em PLANT: expira em 1 min, opera em qualquer horário (aprovada)" in texto
    assert "- Padrão de velas em PLANT: expira em 1 min, opera em" in texto and "(exploração)" in texto
    assert texto.count("Aberta:") == 2  # as operações de verdade aparecem como 'Aberta'


def test_bot_avisa_de_tempos_em_tempos_que_segue_rodando(cfg, banco, cliente):
    cfg.risco.max_operacoes_dia = 1
    _, mensagens = rodar(cfg, banco, cliente, plano(entrada(sessao="asia")), minutos=200,
                         renovar_plano=lambda: None)
    pulsos = [m for m in mensagens if m.startswith("Rodando, ")]
    assert len(pulsos) >= 4  # uma linha a cada meia hora, em 200 minutos
    assert "Hoje: " in pulsos[0] and "operações" in pulsos[0]


# ------------------------------------------------------------------ quedas de conexão
def test_queda_de_internet_longa_nao_para_o_bot(cfg, banco, cliente):
    """Foi o que aconteceu ao vivo: a internet caiu e o bot desistiu em 4 minutos. Agora ele insiste."""
    inicio = cliente.agora()
    cliente.quedas = [(inicio + 20 * 60, inicio + 20 * 60 + 3 * 3600)]  # 3 horas fora do ar
    cfg.risco.max_operacoes_dia = 30
    resumo, mensagens = rodar(cfg, banco, cliente, plano(entrada()), minutos=420)
    texto = "\n".join(mensagens)
    assert "A conexão com a corretora caiu" in texto and "Vou tentar reconectar até ela voltar" in texto
    assert "Ainda sem conexão há" in texto
    assert "Conexão restabelecida depois de 18" in texto  # ~3 horas = 180 minutos
    assert "Muitas falhas seguidas" not in texto
    depois = [o for o in banco.operacoes() if o["ts_sinal"] > inicio + 20 * 60 + 3 * 3600]
    assert depois, "o bot precisa voltar a operar quando a conexão volta"
    assert not [o for o in banco.operacoes() if inicio + 21 * 60 < o["ts_sinal"] < inicio + 20 * 60 + 3 * 3600]
    assert cliente.saldo_demo().valor == pytest.approx(10_000 + resumo["lucro"], abs=0.01)


def test_queda_com_operacao_aberta_e_resolvida_pelo_saldo(cfg, banco, mercado):
    class CaiDepoisDaOrdem(ClienteSimulado):
        def abrir_opcao(self, *args, **kwargs):
            ordem = super().abrir_opcao(*args, **kwargs)
            if not self.quedas:  # a internet cai 5 segundos depois da primeira ordem, por 30 minutos
                self.quedas = [(self._t + 5, self._t + 1800)]
            return ordem

    c = CaiDepoisDaOrdem(mercado, inicio_ts=INICIO)
    cfg.risco.max_operacoes_dia = 3
    resumo, mensagens = rodar(cfg, banco, c, plano(entrada()), minutos=300)
    ops = banco.operacoes()
    assert resumo["operacoes"] == 3 and len(ops) == 3
    primeira = ops[0]
    assert primeira["status"] in ("win", "loss", "equal")  # não ficou sem registro nem virou "empate" falso
    assert "conexão caiu com a operação aberta" in primeira["fonte"] and "saldo" in primeira["fonte"]
    assert primeira["lucro"] == pytest.approx(primeira["saldo_depois"] - primeira["saldo_antes"], abs=0.01)
    assert all("conexão caiu" not in o["fonte"] for o in ops[1:])
    assert any("Conexão restabelecida" in m for m in mensagens)
    assert c.saldo_demo().valor == pytest.approx(10_000 + resumo["lucro"], abs=0.01)


def test_login_recusado_ao_reconectar_para_o_bot(cfg, banco, cliente):
    """Insistir com senha recusada pode bloquear a conta: nesse caso o bot para na hora."""
    inicio = cliente.agora()
    cliente.quedas = [(inicio + 10 * 60, inicio + 10 * 60 + 600)]

    def log(m):
        mensagens.append(m)
        if "A conexão com a corretora caiu" in m:
            cliente.quedas, cliente.login_recusado = [], True  # a rede voltou, mas a senha não vale mais

    mensagens = []
    Executor(cfg, banco, cliente, plano(entrada()), log=log).rodar(minutos=300)
    texto = "\n".join(mensagens)
    assert "Parando: a corretora recusou o login ao reconectar" in texto
    assert texto.count("Ainda sem conexão") == 0 and "Conexão restabelecida" not in texto


def test_parar_durante_a_queda_encerra_sem_esperar(cfg, banco, cliente):
    inicio = cliente.agora()
    cliente.quedas = [(inicio + 5 * 60, inicio + 30 * 86400)]  # fora do ar por um mês
    parar = threading.Event()
    mensagens = []

    def log(m):
        mensagens.append(m)
        if "Ainda sem conexão" in m:
            parar.set()

    Executor(cfg, banco, cliente, plano(entrada()), log=log, parar=parar).rodar(minutos=None)
    assert any("Parado a pedido" in m for m in mensagens)
    assert cliente.agora() < inicio + 3600  # saiu logo, sem esperar a conexão voltar
