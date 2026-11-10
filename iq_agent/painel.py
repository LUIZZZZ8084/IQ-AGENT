"""Painel de controle: uma janela para atualizar o mercado, ligar e desligar o bot e
escolher a conta (demo ou real).

A troca para a conta real é sempre manual: exige a seção [real] liberada no
config.toml, mostra os critérios de prontidão e pede a frase de confirmação digitada.
Toda vez que o painel é aberto, ele começa na conta demo.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from . import app
from .banco import Banco
from .config import Config, ErroConfig, carregar, credenciais
from .relatorio import gerar
from .tipos import FRASE_CONTA_REAL, ErroIQ

registro = logging.getLogger("iq_agent.painel")

COR_DEMO, COR_REAL = "#1c5cab", "#b3261e"


class Painel:
    def __init__(self, raiz: tk.Tk, cfg: Config, reler_config: bool = True):
        self.raiz, self.cfg = raiz, cfg
        self.reler_config = reler_config  # os testes desligam para não ler o config.toml real
        self.fila: queue.Queue = queue.Queue()
        self.cliente = None
        self.trabalho: threading.Thread | None = None
        self.bot_rodando = False
        self.parar = threading.Event()
        self.credenciais: tuple[str, str] | None = None
        self.frase_real: str | None = None  # digitada pelo usuário nesta sessão; nunca é salva
        self.conta = tk.StringVar(value="PRACTICE")
        self.exploracao = tk.BooleanVar(value=cfg.operacao.modo_exploracao)
        self.simular = tk.BooleanVar(value=False)
        self.continuo = tk.BooleanVar(value=True)
        self.log = app.com_hora(lambda texto: self.fila.put(("log", texto)))
        self._montar()
        self._pintar_faixa()
        raiz.protocol("WM_DELETE_WINDOW", self._fechar)
        self._agendado = raiz.after(150, self._drenar)

    # ------------------------------------------------------------ interface
    def _montar(self) -> None:
        r = self.raiz
        self.faixa = tk.Label(r, font=("Segoe UI", 15, "bold"), fg="white", pady=10)
        self.faixa.pack(fill="x")

        linha = ttk.Frame(r, padding=(12, 8, 12, 0))
        linha.pack(fill="x")
        self.rot_estado = ttk.Label(linha, text="Desconectado")
        self.rot_estado.pack(side="left")
        self.rot_saldo = ttk.Label(linha, text="", font=("Segoe UI", 11, "bold"))
        self.rot_saldo.pack(side="right")

        conta = ttk.LabelFrame(r, text="Conta usada pelo bot", padding=10)
        conta.pack(fill="x", padx=12, pady=(8, 0))
        self.radio_demo = ttk.Radiobutton(conta, text="Demo (treino, dinheiro fictício)", variable=self.conta,
                                          value="PRACTICE", command=self._ao_trocar_conta)
        self.radio_demo.pack(side="left")
        self.radio_real = ttk.Radiobutton(conta, text="Real (dinheiro de verdade)", variable=self.conta,
                                          value="REAL", command=self._ao_trocar_conta)
        self.radio_real.pack(side="left", padx=(18, 0))
        ttk.Button(conta, text="Ver critérios para a conta real", command=self._mostrar_prontidao).pack(side="right")

        acoes = ttk.LabelFrame(r, text="Ações", padding=10)
        acoes.pack(fill="x", padx=12, pady=(8, 0))
        self.botoes: dict[str, ttk.Button] = {}
        for chave, texto, comando in (
            ("conectar", "Conectar", self._conectar),
            ("atualizar", "1. Atualizar mercado", self._atualizar_mercado),
            ("iniciar", "2. Iniciar bot", self._iniciar_bot),
            ("parar", "Parar bot", self._parar_bot),
            ("relatorio", "Abrir relatório", self._abrir_relatorio),
            ("diagnostico", "Diagnóstico", self._diagnostico),
        ):
            botao = ttk.Button(acoes, text=texto, command=comando)
            botao.pack(side="left", padx=(0, 8))
            self.botoes[chave] = botao
        self.botoes["parar"].state(["disabled"])

        opcoes = ttk.Frame(r, padding=(12, 8, 12, 0))
        opcoes.pack(fill="x")
        self.marca_exploracao = ttk.Checkbutton(
            opcoes, variable=self.exploracao,
            text="Exploração: operar também candidatos sem vantagem comprovada, com valor mínimo (só demo)")
        self.marca_exploracao.pack(anchor="w")
        ttk.Checkbutton(opcoes, variable=self.simular,
                        text="Simular: mostrar o que o bot faria, sem enviar nenhuma ordem").pack(anchor="w")
        ttk.Checkbutton(opcoes, variable=self.continuo,
                        text="Contínuo: deixar rodando e atualizar o mercado sozinho a cada novo dia (só demo)"
                        ).pack(anchor="w")

        self.texto = ScrolledText(r, height=18, state="disabled", font=("Consolas", 10), wrap="word")
        self.texto.pack(fill="both", expand=True, padx=12, pady=12)
        self._escrever("Painel aberto. Passo 1: Atualizar mercado (baixa os dados reais e refaz a pesquisa). "
                       "Passo 2: Iniciar bot.")

    def _pintar_faixa(self) -> None:
        if self.conta.get() == "REAL":
            self.faixa.config(text="CONTA REAL · dinheiro de verdade", bg=COR_REAL)
        else:
            self.faixa.config(text="CONTA DEMO · treino", bg=COR_DEMO)

    def _escrever(self, texto: str) -> None:
        self.texto.config(state="normal")
        self.texto.insert("end", texto + "\n")
        self.texto.see("end")
        self.texto.config(state="disabled")

    def _ocupar(self, ocupado: bool, bot: bool = False) -> None:
        self.bot_rodando = ocupado and bot
        estado = ["disabled"] if ocupado else ["!disabled"]
        for chave, botao in self.botoes.items():
            if chave not in ("parar", "relatorio"):
                botao.state(estado)
        self.radio_demo.state(estado)
        self.radio_real.state(estado)
        self.botoes["parar"].state(["!disabled"] if self.bot_rodando else ["disabled"])

    def _drenar(self) -> None:
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
        self._agendado = self.raiz.after(150, self._drenar)

    def encerrar_agendamento(self) -> None:
        """Cancela a leitura periódica da fila (ao fechar a janela)."""
        try:
            self.raiz.after_cancel(self._agendado)
        except tk.TclError:
            pass

    # ------------------------------------------------------------ tarefas em segundo plano
    def _trabalhando(self) -> bool:
        return self.trabalho is not None and self.trabalho.is_alive()

    def _executar(self, nome: str, tarefa, bot: bool = False) -> bool:
        if self._trabalhando():
            messagebox.showinfo("IQ-AGENT", "Já existe uma tarefa em andamento. Espere terminar.")
            return False
        if not self._reler():
            return False
        self.parar.clear()
        self._ocupar(True, bot)

        def alvo() -> None:
            try:
                tarefa()
            except (ErroIQ, ErroConfig) as exc:
                self.log(f"Erro: {exc}")
            except Exception as exc:  # noqa: BLE001 - nada pode fechar a janela em silêncio
                registro.exception("Falha na tarefa %s", nome)
                self.log(f"Erro inesperado: {type(exc).__name__}: {exc}")
            finally:
                self.fila.put(("fim", nome))

        self.trabalho = threading.Thread(target=alvo, name=f"iq-{nome}", daemon=True)
        self.trabalho.start()
        return True

    def _reler(self) -> bool:
        """Relê o config.toml, para que mudanças valham sem fechar o painel."""
        if not self.reler_config:
            return True
        try:
            self.cfg = carregar()
            return True
        except ErroConfig as exc:
            messagebox.showerror("Erro no config.toml", str(exc))
            return False

    def _obter_credenciais(self) -> bool:
        """Usa o .env se existir; senão pergunta e guarda só na memória, até fechar o painel."""
        if self.credenciais:
            return True
        try:
            self.credenciais = credenciais()
            return True
        except ErroConfig:
            pass
        dados = self._dialogo_login()
        if dados:
            self.credenciais = dados
        return bool(dados)

    def _garantir_conexao(self) -> None:
        """Chamado dentro das tarefas: conecta (ou reconecta) e atualiza o saldo na tela."""
        if self.cliente is None:
            self.cliente = app.novo_cliente(*self.credenciais)
        if not self.cliente.conectado:
            self.log("Conectando à IQ Option...")
            try:
                self.cliente.conectar()
            except ErroIQ:
                self.credenciais = None  # permite digitar de novo se a senha estava errada
                self.cliente = None
                raise
        self._mostrar_saldo()
        self.fila.put(("estado", "Conectado"))

    def _mostrar_saldo(self) -> None:
        saldo = self.cliente.saldo_ativo(atualizar=True)
        nome = "real" if saldo.nome_conta == "REAL" else "demo"
        self.fila.put(("saldo", f"Saldo {nome}: {saldo.valor:,.2f} {saldo.moeda}"))

    # ------------------------------------------------------------ ações
    def _conectar(self) -> None:
        if self._obter_credenciais():
            self._executar("conectar", self._garantir_conexao)

    def _atualizar_mercado(self) -> None:
        if not self._obter_credenciais():
            return

        def tarefa() -> None:
            self._garantir_conexao()
            banco = Banco(self.cfg.db)
            try:
                app.atualizar_mercado(self.cfg, banco, self.cliente, self.log)
            finally:
                banco.fechar()
            self.log("Pronto. Clique em 'Abrir relatório' para ver o plano do dia.")

        self._executar("atualizar", tarefa)

    def _iniciar_bot(self) -> None:
        conta, exploracao, simular = self.conta.get(), self.exploracao.get(), self.simular.get()
        continuo = self.continuo.get()
        if conta == "REAL":
            if not self.frase_real:
                self.conta.set("PRACTICE")
                self._pintar_faixa()
                return
            if not simular and not messagebox.askyesno(
                    "Conta real", "O bot vai operar com DINHEIRO REAL, sozinho, até você clicar em Parar "
                                  "ou até bater um limite de risco.\n\nIniciar agora?", icon="warning"):
                return
        if not self._obter_credenciais():
            return
        frase = self.frase_real

        def tarefa() -> None:
            self._garantir_conexao()
            if conta == "REAL":
                self.cliente.usar_conta_real(frase)
            else:
                self.cliente.usar_conta_demo()
            self._mostrar_saldo()
            banco = Banco(self.cfg.db)
            try:
                app.operar(self.cfg, banco, self.cliente, self.log, conta=conta, exploracao=exploracao,
                           simular=simular, parar=self.parar, continuo=continuo)
            finally:
                banco.fechar()
            if self.cliente is not None and self.cliente.conectado:
                self._mostrar_saldo()

        self._executar("bot", tarefa, bot=True)

    def _parar_bot(self) -> None:
        if self.bot_rodando:
            self.parar.set()
            self._escrever("Parando... Se houver uma operação aberta, o bot espera o resultado antes de sair.")

    def _diagnostico(self) -> None:
        if not self._obter_credenciais():
            return

        def tarefa() -> None:
            self._garantir_conexao()
            self.cliente.capturar = True
            app.diagnosticar(self.cfg, self.cliente, self.log)

        self._executar("diagnostico", tarefa)

    def _abrir_relatorio(self) -> None:
        if self._trabalhando():
            caminho = self.cfg.dir_relatorios / "relatorio.html"
            if not caminho.exists():
                messagebox.showinfo("IQ-AGENT", "O relatório ainda não foi gerado. Espere a tarefa terminar.")
                return
        else:
            banco = Banco(self.cfg.db)
            try:
                caminho = gerar(banco, self.cfg)
            finally:
                banco.fechar()
        webbrowser.open(caminho.as_uri())

    # ------------------------------------------------------------ conta demo / real
    def _prontidao(self) -> tuple[bool, str, str]:
        banco = Banco(self.cfg.db)
        try:
            pode, explicacao, itens = app.conferir_real(self.cfg, banco, time.time())
        finally:
            banco.fechar()
        lista = "\n".join(f"{'[OK]' if i.ok else '[  ]'} {i.nome}: {i.detalhe}" for i in itens)
        return pode, explicacao, lista

    def _mostrar_prontidao(self) -> None:
        if not self._reler():
            return
        _, explicacao, lista = self._prontidao()
        messagebox.showinfo("Critérios para a conta real", f"{explicacao}\n\n{lista}".strip())

    def _ao_trocar_conta(self) -> None:
        if self.conta.get() == "PRACTICE":
            self.frase_real = None
            if self.cliente is not None:
                self.cliente.usar_conta_demo()
            self._pintar_faixa()
            self.marca_exploracao.state(["!disabled"])
            self._escrever("Conta escolhida: DEMO.")
            self._atualizar_saldo_se_conectado()
            return
        if self._reler():
            pode, explicacao, lista = self._prontidao()
        else:
            pode, explicacao, lista = False, "", ""
        if not pode:
            self.conta.set("PRACTICE")
            if explicacao:
                messagebox.showwarning("Conta real bloqueada", f"{explicacao}\n\n{lista}".strip())
            return
        frase = self._dialogo_confirmar_real(explicacao, lista)
        if frase != FRASE_CONTA_REAL:
            self.conta.set("PRACTICE")
            if frase is not None:
                messagebox.showinfo("Conta real", "A frase não confere. Continuando na conta demo.")
            return
        self.frase_real = frase
        if self.cliente is not None:
            self.cliente.usar_conta_real(frase)
        self.exploracao.set(False)
        self.marca_exploracao.state(["disabled"])
        self._pintar_faixa()
        self._escrever("Conta escolhida: REAL (liberada por você nesta sessão). Exploração desligada.")
        self._atualizar_saldo_se_conectado()

    def _atualizar_saldo_se_conectado(self) -> None:
        if self.cliente is not None and self.cliente.conectado and not self._trabalhando():
            self._executar("saldo", self._mostrar_saldo)

    # ------------------------------------------------------------ diálogos
    def _janela_modal(self, titulo: str) -> tk.Toplevel:
        janela = tk.Toplevel(self.raiz)
        janela.title(titulo)
        janela.transient(self.raiz)
        janela.resizable(False, False)
        janela.grab_set()
        return janela

    def _dialogo_login(self) -> tuple[str, str] | None:
        """Pede e-mail e senha. Os dados ficam só na memória deste painel."""
        janela = self._janela_modal("Entrar na IQ Option")
        corpo = ttk.Frame(janela, padding=16)
        corpo.pack()
        ttk.Label(corpo, wraplength=340, justify="left",
                  text="Digite o login da sua conta IQ Option. Ele é usado só para conectar e não é "
                       "gravado em lugar nenhum.").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Label(corpo, text="E-mail").grid(row=1, column=0, sticky="w")
        email = ttk.Entry(corpo, width=36)
        email.grid(row=1, column=1, pady=3)
        ttk.Label(corpo, text="Senha").grid(row=2, column=0, sticky="w")
        senha = ttk.Entry(corpo, width=36, show="•")
        senha.grid(row=2, column=1, pady=3)
        resposta: list[tuple[str, str]] = []

        def confirmar(*_) -> None:
            if email.get().strip() and senha.get():
                resposta.append((email.get().strip(), senha.get()))
                janela.destroy()

        botoes = ttk.Frame(corpo)
        botoes.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(botoes, text="Cancelar", command=janela.destroy).pack(side="right")
        ttk.Button(botoes, text="Entrar", command=confirmar).pack(side="right", padx=(0, 8))
        janela.bind("<Return>", confirmar)
        email.focus_set()
        self.raiz.wait_window(janela)
        return resposta[0] if resposta else None

    def _dialogo_confirmar_real(self, explicacao: str, lista: str) -> str | None:
        janela = self._janela_modal("Liberar a conta real")
        corpo = ttk.Frame(janela, padding=16)
        corpo.pack()
        aviso = ("Você está prestes a liberar a CONTA REAL.\n\n"
                 "O bot vai abrir operações com dinheiro de verdade, sem pedir confirmação a cada uma. "
                 "Perdas são definitivas. Resultado na demo não garante resultado na real.\n\n"
                 f"Teto por operação na real: {self.cfg.real.valor_maximo:.2f}. Só entram estratégias aprovadas.")
        ttk.Label(corpo, text=aviso, wraplength=460, justify="left").pack(anchor="w")
        ttk.Label(corpo, text=f"\n{explicacao}\n{lista}".rstrip(), wraplength=460, justify="left",
                  font=("Consolas", 9)).pack(anchor="w")
        ttk.Label(corpo, text=f'\nPara confirmar, digite exatamente:  {FRASE_CONTA_REAL}').pack(anchor="w")
        campo = ttk.Entry(corpo, width=44)
        campo.pack(anchor="w", pady=(4, 0))
        resposta: list[str] = []

        def confirmar(*_) -> None:
            resposta.append(campo.get().strip())
            janela.destroy()

        botoes = ttk.Frame(corpo)
        botoes.pack(anchor="e", pady=(14, 0))
        ttk.Button(botoes, text="Cancelar (ficar na demo)", command=janela.destroy).pack(side="right")
        ttk.Button(botoes, text="Liberar conta real", command=confirmar).pack(side="right", padx=(0, 8))
        campo.focus_set()
        self.raiz.wait_window(janela)
        return resposta[0] if resposta else None

    def _fechar(self) -> None:
        if self.bot_rodando:
            messagebox.showwarning("IQ-AGENT", "O bot está rodando. Clique em 'Parar bot' e espere ele encerrar "
                                               "antes de fechar, para não perder o registro de uma operação aberta.")
            return
        if self.cliente is not None:
            try:
                self.cliente.fechar()
            except Exception:  # noqa: BLE001
                pass
        self.encerrar_agendamento()
        self.raiz.destroy()


def abrir(cfg: Config) -> None:
    app.preparar_logs(cfg)
    raiz = tk.Tk()
    raiz.title("IQ-AGENT")
    raiz.geometry("900x640")
    raiz.minsize(760, 520)
    Painel(raiz, cfg)
    raiz.mainloop()
