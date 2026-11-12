"""Executor: opera o plano do dia, uma operação por vez.

A cada fechamento de vela ele confere o gestor de risco, lê as velas dos ativos do
plano, calcula o sinal com a mesma função usada no backtest e, se houver sinal, abre
a operação, espera o resultado e anota tudo no diário.

Conta: demo por padrão. Na conta real só entram estratégias aprovadas (nunca
exploração) e o valor por operação respeita também o teto da seção [real].
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

from . import ml
from .banco import Banco
from .config import Config
from .estatistica import wilson, z_unilateral
from .estrategias import ESTRATEGIAS, NOMES_CURTOS, VELAS_AQUECIMENTO
from .filtros import descrever_sessao, mascara_vol, razao_vol, sessao_inclui
from .iqclient import expiracao_turbo
from .risco import GestorRisco
from .tipos import Ativo, ErroConexao, ErroIQ, ErroLogin, OrdemRecusada, Resultado
from .velas import Velas

ATUALIZAR_ATIVOS_S = 600
MAX_FALHAS_SEGUIDAS = 10
PULSO_S = 1800  # a cada meia hora o bot escreve uma linha dizendo que segue rodando
MAX_ORDENS_COM_PROBLEMA = 5  # ordens seguidas recusadas ou sem confirmação antes de parar o bot
CASTIGO_ATIVO_S = 1800  # depois de um problema, o ativo fica esse tempo sem novas tentativas
ROTULO_CONTA = {"PRACTICE": "DEMO", "REAL": "REAL"}


@dataclass(frozen=True)
class Entrada:
    tipo: str  # "aprovado" ou "exploracao"
    ativo: str
    modo: str
    tf: int
    exp_min: int
    estrategia: str
    params: dict
    sessao: str
    vol: str
    breakeven: float
    p_inf: float | None

    @property
    def chave(self) -> tuple[str, str, str]:
        return self.ativo, self.modo, self.estrategia

    @property
    def config(self) -> dict:
        return {"params": self.params, "sessao": self.sessao, "vol": self.vol}

    @classmethod
    def de_dict(cls, d: dict) -> "Entrada":
        return cls(tipo=d["tipo"], ativo=d["ativo"], modo=d["modo"], tf=int(d["tf"]), exp_min=int(d["exp_min"]),
                   estrategia=d["estrategia"], params=dict(d["params"]), sessao=d["sessao"], vol=d["vol"],
                   breakeven=float(d["breakeven"]), p_inf=d.get("p_inf"))


class _Encerrar(Exception):
    """Limite do dia atingido (perda máxima, meta ou número de operações)."""


class _Abortar(Exception):
    """Condição que exige parar o bot na hora."""


def duracao(segundos: float) -> str:
    """Tempo por extenso: '5 s', '12 min' ou '3 h 05 min'."""
    segundos = max(0, round(segundos))
    if segundos < 90:
        return f"{segundos} s"
    minutos = round(segundos / 60)
    if minutos < 120:
        return f"{minutos} min"
    return f"{minutos // 60} h {minutos % 60:02d} min"


def inicio_do_dia(ts: float) -> float:
    """Meia-noite (horário local) do dia que contém `ts`."""
    return datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def sinal_atual(entrada: Entrada, v: Velas) -> int:
    """Sinal na última vela fechada: +1 CALL, -1 PUT, 0 nada. Mesma lógica do backtest."""
    if entrada.estrategia == ml.NOME:
        return ml.sinal_atual(v, entrada.params)
    sinal = int(ESTRATEGIAS[entrada.estrategia].func(v, **entrada.params)[-1])
    if sinal and entrada.vol != "qualquer" and not bool(mascara_vol(razao_vol(v), entrada.vol)[-1]):
        return 0
    return sinal


def conciliar(resultado: Resultado, valor: float, payout: float, saldo_antes: float,
              saldo_depois: float) -> tuple[str, float, str]:
    """Combina a mensagem da corretora com a variação do saldo. Devolve (status, lucro, fonte)."""
    delta = saldo_depois - saldo_antes
    tolerancia = max(0.02, 0.02 * valor)
    if resultado.status in ("win", "loss", "equal"):
        lucro = resultado.lucro
        if math.isnan(lucro):
            lucro = delta if resultado.status == "win" and delta > 0 else (
                valor * payout if resultado.status == "win" else -valor)
        confere = abs(delta - lucro) <= tolerancia
        return resultado.status, round(lucro, 2), resultado.fonte if confere else f"{resultado.fonte}; saldo divergente"
    # Sem mensagem: decide pelo saldo (só vale porque há uma operação aberta por vez).
    if abs(delta + valor) <= tolerancia:
        return "loss", round(-valor, 2), "saldo"
    if abs(delta) <= tolerancia:
        return "equal", 0.0, "saldo"
    if delta > tolerancia and abs(delta - valor * payout) <= max(tolerancia, 0.1 * valor):
        return "win", round(delta, 2), "saldo"
    return "desconhecido", round(delta, 2), "saldo inconclusivo"


class Executor:
    def __init__(self, cfg: Config, banco: Banco, cliente, plano: dict, exploracao: bool = False,
                 simular: bool = False, log: Callable[[str], None] = print, conta: str = "PRACTICE",
                 parar: threading.Event | None = None,
                 renovar_plano: Callable[[], dict | None] | None = None):
        if conta not in ROTULO_CONTA:
            raise ValueError(f"Conta desconhecida: {conta}")
        self.cfg, self.banco, self.cliente, self.plano = cfg, banco, cliente, plano
        self.conta = conta
        # Exploração é coleta de dados com estratégias NÃO comprovadas: só existe na demo.
        self.exploracao = (exploracao or cfg.operacao.modo_exploracao) and conta == "PRACTICE"
        self.risco_cfg = cfg.risco if conta == "PRACTICE" else replace(
            cfg.risco, valor_maximo=min(cfg.risco.valor_maximo, cfg.real.valor_maximo))
        self.simular = simular
        self.log = log
        self.parar = parar or threading.Event()
        # Modo contínuo (só demo): a cada novo dia refaz a pesquisa e segue operando.
        # Na conta real o bot sempre para à meia-noite, para exigir uma nova partida manual.
        self.renovar_plano = renovar_plano if conta == "PRACTICE" else None
        self._vivos: dict[str, Ativo] = {}
        self._ativos_em = -math.inf
        self._saldo = 0.0
        self._ultimo_aviso = ""
        self._ordens_com_problema = 0
        self._castigo: dict[str, float] = {}  # ativo -> instante até o qual não tentar de novo
        self.resumo = {"operacoes": 0, "vitorias": 0, "derrotas": 0, "empates": 0, "lucro": 0.0}

    # ------------------------------------------------------------ preparação
    def entradas(self, agora: float) -> list[Entrada]:
        """Entradas do plano que serão operadas (aprovadas; exploração só na demo e se pedida)."""
        todas = [Entrada.de_dict(d) for d in self.plano.get("entradas", [])]
        escolhidas = [e for e in todas if e.tipo == "aprovado" or (self.exploracao and e.tipo == "exploracao")]
        suspensas = self._suspensas(escolhidas, agora)
        for e in escolhidas:
            if e.chave in suspensas:
                self.log(f"Suspensa por desempenho ao vivo: {e.ativo} {e.modo} {e.estrategia} ({suspensas[e.chave]})")
        conhecidas = [e for e in escolhidas if e.estrategia in ESTRATEGIAS or e.estrategia == ml.NOME]
        for e in conhecidas:
            if e.estrategia == ml.NOME and ml.carregar_modelo(e.params.get("arquivo")) is None:
                self.log(f"Modelo de {e.ativo} {e.modo} não encontrado em disco; atualize o mercado para treiná-lo.")
        return [e for e in conhecidas if e.chave not in suspensas
                and (e.estrategia != ml.NOME or ml.carregar_modelo(e.params.get("arquivo")) is not None)]

    def _suspensas(self, entradas: list[Entrada], agora: float) -> dict[tuple, str]:
        """Entradas cujo resultado ao vivo já mostra, com confiança, taxa abaixo do breakeven."""
        placar: dict[tuple, list[int]] = {}
        for o in self.banco.operacoes(desde=agora - 14 * 86400, conta=self.conta):
            if o["status"] in ("win", "loss"):
                p = placar.setdefault((o["ativo"], o["modo"], o["estrategia"]), [0, 0])
                p[0] += o["status"] == "win"
                p[1] += 1
        z = z_unilateral(self.cfg.pesquisa.confianca)
        saida = {}
        for e in entradas:
            vitorias, n = placar.get(e.chave, (0, 0))
            if n >= self.cfg.operacao.min_ops_para_suspender and wilson(vitorias, n, z)[1] < e.breakeven:
                saida[e.chave] = f"{vitorias}/{n} acertos, precisa de {e.breakeven:.1%}"
        return saida

    def _atualizar_ativos(self, forcar: bool = False) -> None:
        """Relê a lista de ativos e payouts a cada 10 minutos."""
        c = self.cliente
        agora = c.agora()
        if not (forcar or agora - self._ativos_em >= ATUALIZAR_ATIVOS_S):
            return
        try:
            self._vivos = {a.nome: a for a in c.ativos() if a.tipo == "turbo"}
            self._ativos_em = agora
        except ErroConexao:
            # A lista é grande e às vezes a corretora demora a mandar. Se a conexão continua de pé
            # e já existe uma lista, segue com ela e tenta de novo em 1 minuto, sem tratar como queda.
            if not self._vivos or not getattr(c, "conectado", False):
                raise
            self._ativos_em = agora - ATUALIZAR_ATIVOS_S + 60
            self._avisar("A corretora demorou a mandar a lista de ativos; sigo com a anterior e tento de novo "
                         "em 1 minuto.")

    def _avisar(self, texto: str) -> None:
        if texto != self._ultimo_aviso:
            self.log(texto)
            self._ultimo_aviso = texto

    def _dormir(self, instante: float) -> None:
        """Espera até `instante`, acordando a cada segundo para ver se pediram para parar."""
        c = self.cliente
        while not self.parar.is_set() and c.agora() < instante:
            c.dormir_ate(min(instante, c.agora() + 1.0))

    def _esperar_conexao(self, motivo: Exception) -> bool:
        """A conexão caiu: tenta reconectar até conseguir, com intervalos de até 2 minutos.

        Devolve False se pediram para parar no meio. Levanta _Abortar se o login for
        recusado, porque insistir com senha errada pode bloquear a conta.
        """
        c = self.cliente
        inicio, tentativa = c.agora(), 0
        self.log(f"A conexão com a corretora caiu ({motivo}). Vou tentar reconectar até ela voltar.")
        while not self.parar.is_set():
            tentativa += 1
            self._dormir(c.agora() + min(120, 5 * tentativa))
            if self.parar.is_set():
                break
            try:
                c.garantir()
            except ErroLogin as erro:
                raise _Abortar(f"a corretora recusou o login ao reconectar ({erro})") from None
            except ErroIQ as erro:
                if tentativa <= 2 or tentativa % 10 == 0:
                    self.log(f"Ainda sem conexão há {duracao(c.agora() - inicio)} ({erro}). Sigo tentando.")
                continue
            self.log(f"Conexão restabelecida depois de {duracao(c.agora() - inicio)}.")
            self._ativos_em = -math.inf  # a lista de ativos e payouts pode ter mudado
            return True
        return False

    def _conferir_conta(self) -> None:
        if self.cliente.conta != self.conta:
            raise _Abortar(f"a conexão está na conta {self.cliente.conta}, mas o executor foi iniciado "
                           f"para {self.conta}")

    def _renovar(self, entradas: list[Entrada], agora: float) -> list[Entrada]:
        """Novo dia no modo contínuo: atualiza o mercado e troca o plano."""
        if self.renovar_plano is None:
            return entradas
        try:
            novo = self.renovar_plano()
        except ErroIQ as exc:
            self.log(f"Não consegui atualizar o mercado ({exc}); sigo com o plano anterior.")
            return entradas
        if novo:
            self.plano = novo
            entradas = self.entradas(agora)
            self.log(f"Plano do dia atualizado: {len(entradas)} estratégia(s) em observação.")
            self._listar(entradas)
        return entradas

    def _listar(self, entradas: list[Entrada]) -> None:
        """Mostra o que o bot está vigiando. Cada linha é uma estratégia, não uma operação."""
        for e in entradas:
            nome = NOMES_CURTOS.get(e.estrategia, e.estrategia)
            tipo = "aprovada" if e.tipo == "aprovado" else "exploração"
            self.log(f"  - {nome} em {e.ativo}: expira em {e.exp_min} min, opera em {descrever_sessao(e.sessao)} "
                     f"({tipo})")

    # ------------------------------------------------------------ laço principal
    def rodar(self, minutos: float | None = None) -> dict:
        c = self.cliente
        rotulo = ROTULO_CONTA[self.conta]
        if c.conta != self.conta:
            raise ErroIQ(f"A conexão está na conta {c.conta}, mas o executor foi iniciado para {self.conta}.")
        saldo = c.saldo_ativo(atualizar=True)
        if saldo.nome_conta != self.conta:
            raise ErroIQ("O saldo devolvido pela corretora não é o da conta escolhida; abortando.")
        self._saldo = saldo.valor
        agora = c.agora()
        entradas = self.entradas(agora)
        continuo = self.renovar_plano is not None
        if not entradas:
            dica = ("" if self.exploracao or self.conta == "REAL"
                    else " Marque 'exploração' para operar candidatos na demo com valor mínimo.")
            self.log("Nada a operar: o plano não tem estratégias aprovadas." + dica)
            if not continuo:
                return self.resumo
            self.log("Modo contínuo: fico aguardando a atualização de amanhã.")
        dia = inicio_do_dia(agora)
        gestor = GestorRisco.do_historico(self.risco_cfg, self.cfg.operacao, self._saldo,
                                          self.banco.operacoes(desde=dia, conta=self.conta))
        self.log(f"Conta {rotulo}: {self._saldo:.2f} {saldo.moeda}. Vigiando {len(entradas)} estratégia(s) do plano"
                 + (" [SIMULAÇÃO: nenhuma ordem será enviada]" if self.simular else ""))
        self._listar(entradas)
        if entradas:
            self.log("Essas são as estratégias vigiadas, não operações. O bot abre uma operação por vez, "
                     "quando uma delas dá sinal; cada operação aparece abaixo como 'Aberta'.")
        fim = agora + minutos * 60 if minutos else None
        falhas = 0
        pausado = False  # True depois de bater um limite do dia, no modo contínuo
        pulso = agora  # de tempos em tempos o bot avisa que segue vivo, mesmo sem operar
        try:
            while not self.parar.is_set():
                agora = c.agora()
                if fim is not None and agora >= fim:
                    self.log("Tempo de execução encerrado.")
                    break
                if inicio_do_dia(agora) != dia:
                    if self.conta == "REAL":
                        self.log("Novo dia: na conta real o bot para à meia-noite. Atualize o mercado e "
                                 "inicie de novo.")
                        break
                    dia = inicio_do_dia(agora)
                    self.log("Novo dia: limites de risco reiniciados.")
                    entradas = self._renovar(entradas, agora)
                    try:
                        self._saldo = c.saldo_ativo(atualizar=True).valor
                    except ErroIQ:
                        pass  # sem conexão na virada do dia: segue com o último saldo conhecido
                    gestor = GestorRisco(self.risco_cfg, self.cfg.operacao, self._saldo)
                    pausado = False
                try:
                    ativo_hoje = bool(entradas) and not pausado
                    if ativo_hoje:
                        self._atualizar_ativos()
                    fronteira = (int(c.agora()) // 60 + 1) * 60
                    self._dormir(fronteira + self.cfg.operacao.espera_fechamento_s)
                    if self.parar.is_set():
                        break
                    if ativo_hoje:
                        self._ciclo(fronteira, entradas, gestor)
                    falhas = 0
                    if c.agora() - pulso >= PULSO_S:
                        pulso = c.agora()
                        estado = ("aguardando sinal" if ativo_hoje else
                                  "parado por hoje (limite do dia)" if pausado else "sem estratégias no plano")
                        self.log(f"Rodando, {estado}. Hoje: {gestor.operacoes} operações, resultado "
                                 f"{gestor.lucro:+.2f}.")
                except _Abortar as motivo:
                    self.log(f"Parando: {motivo}")
                    break
                except _Encerrar as fim_dia:
                    if not continuo:
                        self.log(f"Encerrando: {fim_dia}")
                        break
                    pausado = True
                    self.log(f"Por hoje chega: {fim_dia}. Volto a operar amanhã, depois de atualizar o mercado.")
                except ErroConexao as exc:
                    # Queda de internet ou da corretora: insiste até voltar. Só para se pedirem ou se o
                    # login for recusado.
                    try:
                        if not self._esperar_conexao(exc):
                            break
                        self._saldo = c.saldo_ativo(atualizar=True).valor  # o saldo pode ter mudado na queda
                        falhas = 0
                    except _Abortar as motivo:
                        self.log(f"Parando: {motivo}")
                        break
                    except ErroIQ:
                        pass  # caiu de novo logo depois de voltar: a próxima volta do laço trata
                except ErroIQ as exc:
                    # Erro que não é de conexão (resposta inesperada da corretora): tem limite de tentativas.
                    falhas += 1
                    self.log(f"Problema com a corretora ({exc}). Tentativa {falhas}/{MAX_FALHAS_SEGUIDAS}.")
                    if falhas >= MAX_FALHAS_SEGUIDAS:
                        self.log("Muitas falhas seguidas; parando.")
                        break
                    self._dormir(c.agora() + min(60, 5 * falhas))
            if self.parar.is_set():
                self.log("Parado a pedido.")
        except KeyboardInterrupt:
            self.log("Interrompido pelo usuário.")
        r = self.resumo
        self.log(f"Resumo: {r['operacoes']} operações, {r['vitorias']} ganhos, {r['derrotas']} perdas, "
                 f"{r['empates']} empates, resultado {r['lucro']:+.2f}. Saldo {rotulo}: {self._saldo:.2f}")
        return r

    def _ciclo(self, fronteira: int, entradas: list[Entrada], gestor: GestorRisco) -> None:
        c = self.cliente
        decisao = gestor.pode_operar(c.agora())
        if not decisao.pode:
            if decisao.encerrar_dia:
                raise _Encerrar(decisao.motivo)
            self._avisar(f"Sem operar: {decisao.motivo}")
            return
        elegiveis = [e for e in entradas
                     if fronteira % e.tf == 0 and sessao_inclui(e.sessao, fronteira) and e.ativo in self._vivos
                     and c.agora() >= self._castigo.get(e.ativo, 0.0)]
        if not elegiveis:
            return
        ids = sorted({self._vivos[e.ativo].id for e in elegiveis})
        series = c.velas_varias(ids, 60, 1000, ate=fronteira)
        for e in elegiveis:
            ativo = self._vivos[e.ativo]
            bruto = series.get(ativo.id)
            if bruto is None:
                continue
            v1 = bruto.fechadas_ate(fronteira)
            if len(v1) < VELAS_AQUECIMENTO or int(v1.ts[-1]) != fronteira - 60:
                continue  # dados atrasados ou insuficientes: melhor não operar
            v = v1 if e.tf == 60 else v1.reamostrar(e.tf)
            if len(v) < 100 or int(v.ts[-1]) != fronteira - e.tf:
                continue
            direcao = sinal_atual(e, v)
            if direcao and self._operar(e, ativo, direcao, fronteira, gestor):
                return  # uma operação por vez

    def _operar(self, e: Entrada, ativo: Ativo, direcao: int, fronteira: int, gestor: GestorRisco) -> bool:
        c = self.cliente
        lado = "call" if direcao > 0 else "put"
        rotulo = f"{e.ativo} {lado.upper()} {e.exp_min}min ({e.estrategia}, {e.tipo})"
        if not ativo.aberto:
            self._avisar(f"{rotulo}: ativo fechado agora, pulando")
            return False
        if ativo.payout < self.cfg.coleta.payout_minimo:
            self._avisar(f"{rotulo}: payout {ativo.payout:.0%} abaixo do mínimo, pulando")
            return False
        valor, motivo = gestor.valor(self._saldo, e.p_inf, ativo.payout, e.tipo == "exploracao", ativo.aposta_min)
        if valor <= 0:
            self._avisar(f"{rotulo}: {motivo}")
            return False
        atraso = c.agora() - fronteira
        if atraso > self.cfg.operacao.atraso_maximo_s:
            self._avisar(f"{rotulo}: sinal chegou {atraso:.1f}s atrasado, pulando")
            return False
        if self.simular:
            self.log(f"[simulação] entraria em {rotulo} com {valor:.2f} (payout {ativo.payout:.0%}; {motivo})")
            return True

        self._conferir_conta()
        comum = dict(ts_sinal=fronteira, conta=self.conta, tipo=e.tipo, ativo=e.ativo, modo=e.modo,
                     estrategia=e.estrategia, config=e.config, direcao=lado, valor=valor, payout=ativo.payout,
                     saldo_antes=self._saldo, atraso_s=round(atraso, 2), run_id=self.plano.get("run_id"))
        envio = c.agora()
        try:
            ordem = c.abrir_opcao(ativo, lado, valor, e.exp_min)
        except OrdemRecusada as exc:
            self.log(f"{rotulo}: ordem recusada pela corretora ({exc})")
            self.banco.registrar_operacao(**comum, status="recusada", lucro=0.0, erro=str(exc)[:300])
            self._problema_na_ordem(e.ativo)
            return False
        except ErroIQ as exc:
            return self._ordem_sem_confirmacao(e, ativo, valor, rotulo, comum, envio, exc, gestor)
        self._ordens_com_problema = 0
        self.log(f"[{ROTULO_CONTA[self.conta]}] Aberta: {rotulo} valor {valor:.2f}, payout {ativo.payout:.0%}, "
                 f"atraso {atraso:.1f}s")
        abortar = None
        try:
            resultado = c.aguardar_resultado(ordem)
            status, lucro, fonte, saldo_depois = self._fechar_contas(resultado, valor, ativo.payout)
        except ErroIQ as exc:
            # A conexão caiu com a operação aberta. O resultado sai na corretora de qualquer jeito:
            # espera a conexão voltar e descobre pelo saldo, para a operação não ficar sem registro.
            try:
                status, lucro, fonte, saldo_depois = self._fechar_contas_apos_queda(ordem.expiracao, valor,
                                                                                   ativo.payout, exc)
            except _Abortar as motivo:
                abortar = motivo
                status, lucro, fonte, saldo_depois = ("desconhecido", 0.0, "conexão caiu com a operação aberta",
                                                      self._saldo)
        self._registrar_resultado(comum, ordem.id, ordem.aberta_em, ordem.expiracao, status, lucro, fonte,
                                  saldo_depois, gestor)
        if abortar is not None:
            raise abortar
        return True

    def _fechar_contas_apos_queda(self, expiracao: int, valor: float, payout: float,
                                  motivo: Exception) -> tuple[str, float, str, float]:
        """Reconecta (insistindo) e descobre pelo saldo o resultado da operação que estava aberta."""
        c = self.cliente
        sem_mensagem = Resultado("desconhecido", 0.0, "sem-mensagem")
        while True:
            if not self._esperar_conexao(motivo):
                return "desconhecido", 0.0, "conexão caiu com a operação aberta", self._saldo
            try:
                c.dormir_ate(expiracao + 20)
                status, lucro, fonte, saldo_depois = self._fechar_contas(sem_mensagem, valor, payout)
                return status, lucro, f"{fonte}; conexão caiu com a operação aberta", saldo_depois
            except ErroIQ as outra:
                motivo = outra

    def _problema_na_ordem(self, ativo: str) -> None:
        """Conta ordens seguidas com problema. O ativo descansa; se o problema se repete, o bot para."""
        self._castigo[ativo] = self.cliente.agora() + CASTIGO_ATIVO_S
        self._ordens_com_problema += 1
        if self._ordens_com_problema >= MAX_ORDENS_COM_PROBLEMA:
            raise _Abortar(f"{MAX_ORDENS_COM_PROBLEMA} ordens seguidas não foram aceitas pela corretora. "
                           "Veja os motivos nas linhas acima.")
        self.log(f"  {ativo} fica {CASTIGO_ATIVO_S // 60} minutos sem novas tentativas.")

    def _ordem_sem_confirmacao(self, e: Entrada, ativo: Ativo, valor: float, rotulo: str, comum: dict,
                               envio: float, exc: Exception, gestor: GestorRisco) -> bool:
        """A corretora não confirmou a abertura. A ordem pode ter entrado ou não: o saldo decide."""
        c = self.cliente
        self.log(f"{rotulo}: a corretora não confirmou a ordem ({exc}). Conferindo pelo saldo se ela entrou...")
        try:
            c.garantir()
            saldo_agora = c.saldo_ativo(atualizar=True).valor
        except ErroLogin as erro:
            raise _Abortar(f"a corretora recusou o login ao reconectar ({erro})") from None
        except ErroIQ as queda:  # a conexão caiu junto com a ordem: espera voltar antes de conferir
            if not self._esperar_conexao(queda):
                self.banco.registrar_operacao(**comum, status="desconhecido", lucro=0.0,
                                              erro="conexão caiu ao enviar a ordem; bot parado antes de conferir")
                return False
            saldo_agora = c.saldo_ativo(atualizar=True).valor
        if abs((self._saldo - saldo_agora) - valor) > max(0.02, 0.02 * valor):
            self.log("  O saldo não mudou: a ordem não entrou.")
            self.banco.registrar_operacao(**comum, status="falhou", lucro=0.0, erro=str(exc)[:300])
            self._problema_na_ordem(e.ativo)
            return False
        expiracao = expiracao_turbo(envio, e.exp_min)
        self.log(f"  O saldo caiu {valor:.2f}: a ordem entrou. Aguardando o resultado...")
        c.dormir_ate(expiracao + 20)
        sem_mensagem = Resultado("desconhecido", 0.0, "sem-mensagem")
        status, lucro, fonte, saldo_depois = self._fechar_contas(sem_mensagem, valor, ativo.payout)
        self._ordens_com_problema = 0
        self._registrar_resultado(comum, None, envio, expiracao, status, lucro,
                                  f"{fonte}; ordem sem confirmação", saldo_depois, gestor)
        return True

    def _registrar_resultado(self, comum: dict, opcao_id: int | None, ts_ordem: float, expiracao: int, status: str,
                             lucro: float, fonte: str, saldo_depois: float, gestor: GestorRisco) -> None:
        """Anota a operação encerrada no diário e atualiza o gestor de risco e o resumo."""
        self.banco.registrar_operacao(**comum, ts_ordem=ts_ordem, expiracao=expiracao, opcao_id=opcao_id,
                                      status=status, lucro=lucro, saldo_depois=saldo_depois, fonte=fonte)
        self._saldo = saldo_depois
        if status in ("win", "loss", "equal"):
            gestor.registrar(status, lucro, self.cliente.agora())
            r = self.resumo
            r["operacoes"] += 1
            r["lucro"] = round(r["lucro"] + lucro, 2)
            r[{"win": "vitorias", "loss": "derrotas", "equal": "empates"}[status]] += 1
        nome = {"win": "GANHOU", "loss": "PERDEU", "equal": "EMPATE"}.get(status, "INDEFINIDO")
        self.log(f"  {nome} {lucro:+.2f} | saldo {saldo_depois:.2f} | dia {gestor.lucro:+.2f} "
                 f"em {gestor.operacoes} operações")

    def _fechar_contas(self, resultado: Resultado, valor: float, payout: float) -> tuple[str, float, str, float]:
        """Lê o saldo após a expiração (com algumas tentativas, pois a liquidação pode atrasar)."""
        c = self.cliente
        status, lucro, fonte, saldo_depois = "desconhecido", 0.0, "", self._saldo
        for tentativa in range(8):
            saldo_depois = c.saldo_ativo(atualizar=True).valor
            status, lucro, fonte = conciliar(resultado, valor, payout, self._saldo, saldo_depois)
            liquidado = "divergente" not in fonte and "inconclusivo" not in fonte
            # sem mensagem, "perdeu" pode ser só o saldo ainda não creditado: confirma mais vezes
            if liquidado and not (resultado.status == "desconhecido" and status == "loss" and tentativa < 4):
                break
            c.dormir_ate(c.agora() + 1.5)
        return status, lucro, fonte, saldo_depois
