"""Painel de controle, com a janela escondida e a corretora simulada."""
import time

import pytest

from conftest import FIM
from iq_agent.simulado import ClienteSimulado
from iq_agent.sintetico import gerar_velas
from iq_agent.tipos import FRASE_CONTA_REAL

tk = pytest.importorskip("tkinter")
painel_mod = pytest.importorskip("iq_agent.painel")


@pytest.fixture(scope="module")
def raiz_tk():
    """Uma única raiz Tk para o módulo: criar várias seguidas no mesmo processo é instável."""
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sem ambiente gráfico")
    raiz.withdraw()
    yield raiz
    raiz.destroy()


@pytest.fixture
def painel(cfg, monkeypatch, raiz_tk):
    raiz = tk.Toplevel(raiz_tk)
    raiz.withdraw()
    avisos: list[tuple[str, str]] = []
    for nome in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(painel_mod.messagebox, nome, lambda titulo, texto, _n=nome, **kw: avisos.append((_n, texto)))
    monkeypatch.setattr(painel_mod.messagebox, "askyesno", lambda *a, **kw: True)
    p = painel_mod.Painel(raiz, cfg, reler_config=False)
    p.avisos = avisos
    p.credenciais = ("teste@exemplo.com", "senha-de-teste")
    p.cliente = ClienteSimulado({"PLANT": gerar_velas(2 * 1440, semente=21, fim_ts=FIM, reversao_apos=3, forca=0.2)},
                                inicio_ts=FIM - 1440 * 60)
    yield p
    p.encerrar_agendamento()
    p.cliente = None
    raiz.destroy()


def esperar(p, segundos=20.0):
    """Roda o laço da janela até a tarefa em segundo plano terminar."""
    limite = time.time() + segundos
    while time.time() < limite:
        p.raiz.update()
        p._drenar_uma_vez()
        if not p._trabalhando() and p.fila.empty():
            return
        time.sleep(0.01)
    raise AssertionError("a tarefa do painel não terminou a tempo")


@pytest.fixture(autouse=True)
def drenar_sem_reagendar(monkeypatch):
    """Nos testes o laço é manual: troca o agendamento periódico por uma drenagem única."""
    def uma_vez(self):
        import queue

        try:
            while True:
                tipo, valor = self.fila.get_nowait()
                if tipo == "log":
                    self._escrever(valor)
                elif tipo == "saldo":
                    self.rot_saldo.config(text=valor)
                elif tipo == "estado":
                    self.rot_estado.config(text=valor)
                elif tipo == "fim":
                    self._ocupar(False)
        except queue.Empty:
            pass

    monkeypatch.setattr(painel_mod.Painel, "_drenar_uma_vez", uma_vez, raising=False)


def texto_do_log(p) -> str:
    return p.texto.get("1.0", "end")


def test_abre_na_conta_demo(painel):
    assert painel.conta.get() == "PRACTICE"
    assert "DEMO" in painel.faixa.cget("text")
    assert painel.botoes["parar"].instate(["disabled"])
    assert painel.botoes["iniciar"].instate(["!disabled"])


def test_conectar_mostra_o_saldo_demo(painel):
    painel._conectar()
    esperar(painel)
    assert "Saldo demo: 10,000.00" in painel.rot_saldo.cget("text")
    assert painel.rot_estado.cget("text") == "Conectado"


def test_conta_real_fica_bloqueada_enquanto_o_config_nao_permitir(painel):
    painel.conta.set("REAL")
    painel._ao_trocar_conta()
    assert painel.conta.get() == "PRACTICE" and painel.frase_real is None
    assert painel.cliente.conta == "PRACTICE"
    assert any("desligada" in texto for _, texto in painel.avisos)


def test_conta_real_exige_a_frase_digitada(painel, monkeypatch):
    painel.cfg.real.permitido = True
    painel.cfg.real.exigir_prontidao = False
    monkeypatch.setattr(painel, "_dialogo_confirmar_real", lambda *a: "quero sim")
    painel.conta.set("REAL")
    painel._ao_trocar_conta()
    assert painel.conta.get() == "PRACTICE" and painel.cliente.conta == "PRACTICE"

    monkeypatch.setattr(painel, "_dialogo_confirmar_real", lambda *a: None)  # cancelou
    painel.conta.set("REAL")
    painel._ao_trocar_conta()
    assert painel.conta.get() == "PRACTICE"

    monkeypatch.setattr(painel, "_dialogo_confirmar_real", lambda *a: FRASE_CONTA_REAL)
    painel.exploracao.set(True)
    painel.conta.set("REAL")
    painel._ao_trocar_conta()
    assert painel.conta.get() == "REAL" and painel.cliente.conta == "REAL"
    assert "REAL" in painel.faixa.cget("text")
    assert painel.exploracao.get() is False  # exploração nunca na conta real

    painel.conta.set("PRACTICE")
    painel._ao_trocar_conta()
    assert painel.cliente.conta == "PRACTICE" and painel.frase_real is None


def test_iniciar_sem_plano_atualiza_o_mercado_sozinho(painel):
    painel.continuo.set(False)
    painel._iniciar_bot()
    esperar(painel, 60)
    registro = texto_do_log(painel)
    assert "atualizando o mercado antes de operar" in registro
    assert "Coleta concluída" in registro and "Pesquisa concluída" in registro
    assert "Nada a operar" in registro  # um dia de histórico não aprova nada
    assert painel.botoes["iniciar"].instate(["!disabled"])  # os botões voltaram


def test_bot_opera_na_demo_pelo_painel(painel):
    from iq_agent.banco import Banco

    painel.cfg.risco.max_operacoes_dia = 3
    painel.continuo.set(False)
    banco = Banco(painel.cfg.db)
    resumo = {"ativos": 1, "velas": 0, "periodo": [None, None], "hipoteses": 1, "testaveis": 1, "aprovadas": 1,
              "p_menor_5": 1, "esperado_acaso": 0.1, "taxa_media": None, "breakeven_medio": None}
    banco.salvar_pesquisa("r1", painel.cliente.agora(), resumo, [], {"run_id": "r1", "entradas": [{
        "tipo": "aprovado", "ativo": "PLANT", "modo": "M1>1", "tf": 60, "exp_min": 1, "estrategia": "sequencia",
        "params": {"n": 3, "modo": "reversao"}, "sessao": "todas", "vol": "qualquer", "payout": 0.85,
        "breakeven": 0.5405, "p_inf": 0.58, "taxa": 0.6, "n_teste": 1000}]})
    banco.fechar()
    painel._iniciar_bot()
    esperar(painel, 60)
    banco = Banco(painel.cfg.db)
    ops = banco.operacoes()
    banco.fechar()
    assert len(ops) == 3 and {o["conta"] for o in ops} == {"PRACTICE"}
    assert "Resumo: 3 operações" in texto_do_log(painel)
    assert "Não consegui atualizar o relatório" not in texto_do_log(painel)
    assert (painel.cfg.dir_relatorios / "relatorio.html").exists()


def test_simulacao_fica_evidente_na_faixa_e_no_botao(painel):
    assert painel.botoes["iniciar"].cget("text") == "2. Iniciar bot"
    painel.simular.set(True)
    painel._pintar_faixa()
    assert "SIMULAÇÃO: nenhuma ordem é enviada" in painel.faixa.cget("text")
    assert painel.botoes["iniciar"].cget("text") == "2. Iniciar simulação"
    painel.simular.set(False)
    painel._pintar_faixa()
    assert painel.faixa.cget("text") == "CONTA DEMO · treino"
    assert painel.botoes["iniciar"].cget("text") == "2. Iniciar bot"


def test_iniciar_em_simulacao_avisa_e_nao_abre_operacoes(painel):
    from iq_agent.banco import Banco

    painel.continuo.set(False)
    painel.simular.set(True)
    banco = Banco(painel.cfg.db)
    resumo = {"ativos": 1, "velas": 0, "periodo": [None, None], "hipoteses": 1, "testaveis": 1, "aprovadas": 1,
              "p_menor_5": 1, "esperado_acaso": 0.1, "taxa_media": None, "breakeven_medio": None}
    banco.salvar_pesquisa("r1", painel.cliente.agora(), resumo, [], {"run_id": "r1", "entradas": [{
        "tipo": "aprovado", "ativo": "PLANT", "modo": "M1>1", "tf": 60, "exp_min": 1, "estrategia": "sequencia",
        "params": {"n": 3, "modo": "reversao"}, "sessao": "todas", "vol": "qualquer", "payout": 0.85,
        "breakeven": 0.5405, "p_inf": 0.58, "taxa": 0.6, "n_teste": 1000}]})
    banco.fechar()
    painel._iniciar_bot()
    assert painel.marca_simular.instate(["disabled"])  # as opções ficam travadas enquanto o bot roda
    painel.raiz.after(1500, painel._parar_bot)
    esperar(painel, 60)
    registro = texto_do_log(painel)
    assert "ATENÇÃO: 'Só simular' está marcado" in registro and "[simulação] entraria em" in registro
    banco = Banco(painel.cfg.db)
    assert banco.operacoes() == []
    banco.fechar()
    assert painel.marca_simular.instate(["!disabled"])


def test_sem_exploracao_o_painel_avisa(painel):
    painel.continuo.set(False)
    painel.exploracao.set(False)
    painel._iniciar_bot()
    esperar(painel, 60)
    assert "'Exploração' está desmarcada" in texto_do_log(painel)
