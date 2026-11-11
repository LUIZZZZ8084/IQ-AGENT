"""Conexão com a IQ Option: login por HTTP e mensagens por websocket.

A IQ Option não publica uma API oficial. Este cliente fala o mesmo protocolo que o
aplicativo usa, então pode parar de funcionar se a corretora mudar alguma coisa.

Conta usada nas ordens: por padrão, o saldo de prática (demo). A conta real só é
usada depois de `usar_conta_real(frase)`, chamada com a frase de confirmação que o
usuário digita na hora. A liberação vale só enquanto o programa estiver aberto: toda
nova execução começa na demo.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import requests
import websocket

from .tipos import (FRASE_CONTA_REAL, NOMES_CONTA, PRATICA, REAL, Ativo, ContaRealBloqueada, ErroConexao,
                    ErroIQ, ErroLogin, Ordem, OrdemRecusada, Resultado, Saldo)
from .velas import Velas

log = logging.getLogger("iq_agent.cliente")

OPCAO_TURBO = 3
MAX_VELAS_POR_PEDIDO = 1000


# --------------------------------------------------------------------------- funções puras
def expiracao_turbo(agora: float, minutos: int) -> int:
    """Timestamp de expiração de uma opção turbo de `minutos` (1 a 5).

    A corretora expira em minutos cheios e exige mais de 30 s até a primeira expiração.
    """
    if not 1 <= minutos <= 5:
        raise ValueError("Opções turbo aceitam expiração de 1 a 5 minutos.")
    fronteira = (int(agora) // 60 + 1) * 60
    if fronteira - agora <= 30:
        fronteira += 60
    return fronteira + (minutos - 1) * 60


def _num(valor: Any) -> float | None:
    try:
        return None if valor is None else float(valor)
    except (TypeError, ValueError):
        return None


def interpretar_velas(msg: Any) -> list[tuple]:
    """Converte a resposta 'candles' em tuplas (ts, open, high, low, close, volume)."""
    saida = []
    for c in (msg or {}).get("candles") or []:
        try:
            saida.append((int(c["from"]), float(c["open"]), float(c["max"]), float(c["min"]),
                          float(c["close"]), float(c.get("volume") or 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
    return saida


def interpretar_ativos(msg: Any) -> list[Ativo]:
    """Converte a resposta 'initialization-data' em lista de ativos turbo e binary."""
    saida = []
    for tipo in ("turbo", "binary"):
        ativos = ((msg or {}).get(tipo) or {}).get("actives") or {}
        for chave, info in ativos.items():
            try:
                nome = str(info["name"])
                nome = nome.split(".", 1)[1] if "." in nome else nome
                comissao = float(info["option"]["profit"]["commission"])
                aberto = bool(info.get("enabled", False)) and not bool(info.get("is_suspended", False))
                saida.append(Ativo(
                    id=int(info.get("id", chave)), nome=nome, tipo=tipo,
                    payout=round((100.0 - comissao) / 100.0, 4), aberto=aberto,
                    aposta_min=_num(info.get("minimal_bet")), aposta_max=_num(info.get("maximal_bet")),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return saida


def interpretar_saldos(lista: Any) -> list[Saldo]:
    saida = []
    for s in lista or []:
        try:
            saida.append(Saldo(id=int(s["id"]), tipo=int(s["type"]), valor=float(s["amount"]),
                               moeda=str(s.get("currency", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return saida


_STATUS = {"win": "win", "loose": "loss", "lose": "loss", "loss": "loss", "equal": "equal"}


def interpretar_fechamento(nome: str, corpo: Any) -> tuple[int, Resultado] | None:
    """Lê as mensagens de encerramento de uma opção. Devolve (id da opção, resultado)."""
    if not isinstance(corpo, dict):
        return None
    try:
        if nome == "socket-option-closed":
            opcao, bruto = corpo["id"], corpo.get("win")
            investido, total = _num(corpo.get("sum")), _num(corpo.get("win_amount"))
        elif nome == "option-closed":
            opcao = corpo.get("option_id", corpo.get("id"))
            bruto = corpo.get("result", corpo.get("win"))
            investido = _num(corpo.get("amount", corpo.get("sum")))
            total = _num(corpo.get("profit_amount", corpo.get("win_amount")))
        elif nome == "position-changed":
            if corpo.get("status") != "closed" or corpo.get("external_id") is None:
                return None
            opcao = corpo["external_id"]
            investido, total = _num(corpo.get("invest")), _num(corpo.get("close_profit"))
            if investido is None or total is None:
                return None
            bruto = "win" if total > investido else "equal" if total == investido else "loose"
        else:
            return None
        status = _STATUS.get(str(bruto).lower())
        if status is None or opcao is None:
            return None
        if status == "equal":
            lucro = 0.0
        elif status == "loss":
            lucro = -investido if investido is not None else float("nan")
        else:
            lucro = total - investido if None not in (total, investido) else float("nan")
        return int(opcao), Resultado(status=status, lucro=lucro, fonte=nome)
    except (KeyError, TypeError, ValueError):
        return None


def formato(obj: Any, profundidade: int = 3) -> Any:
    """Estrutura de uma mensagem sem os valores (para diagnóstico sem expor dados pessoais)."""
    if isinstance(obj, dict):
        if profundidade <= 0:
            return f"dict({len(obj)})"
        itens = list(obj.items())
        if len(itens) > 12 and all(str(k).isdigit() for k, _ in itens[:12]):
            return {"<id>": formato(itens[0][1], profundidade - 1), "...": f"{len(itens)} itens"}
        return {str(k): formato(v, profundidade - 1) for k, v in itens[:40]}
    if isinstance(obj, list):
        return [formato(obj[0], profundidade - 1), f"... {len(obj)} itens"] if obj else []
    return type(obj).__name__


# --------------------------------------------------------------------------- cliente
class _Pendente:
    __slots__ = ("nomes", "evento", "resposta", "erro")

    def __init__(self, nomes: tuple[str, ...]):
        self.nomes = nomes
        self.evento = threading.Event()
        self.resposta: dict | None = None
        self.erro: str | None = None


class IQClient:
    def __init__(self, email: str, senha: str, host: str = "iqoption.com", timeout: float = 20.0,
                 capturar: bool = False):
        self._email, self._senha = email, senha
        self._host, self._timeout = host, timeout
        self._ws: websocket.WebSocket | None = None
        self._thread: threading.Thread | None = None
        self._parar = threading.Event()
        self._vivo = False
        self._trava = threading.Lock()
        self._pend: dict[str, _Pendente] = {}
        self._seq = 0
        self._srv_ms: float | None = None
        self._srv_mono = 0.0
        self._saldos: list[Saldo] = []
        self._tipo_conta = PRATICA
        self._fechamentos: dict[int, Resultado] = {}
        self.capturar = capturar
        self.amostras: dict[str, Any] = {}

    # ------------------------------------------------------------ relógio
    def agora(self) -> float:
        """Horário do servidor, em segundos (cai para o relógio local antes da 1ª sincronização)."""
        if self._srv_ms is None:
            return time.time()
        return self._srv_ms / 1000.0 + (time.monotonic() - self._srv_mono)

    def dormir_ate(self, instante: float) -> None:
        while True:
            falta = instante - self.agora()
            if falta <= 0:
                return
            time.sleep(min(falta, 0.5))

    # ------------------------------------------------------------ conexão
    @property
    def conectado(self) -> bool:
        return self._vivo

    def conectar(self) -> None:
        self.fechar()
        ssid = self._login()
        self._abrir_ws()
        self._autenticar(ssid)
        self._enviar({"name": "setOptions", "request_id": self._rid(), "msg": {"sendResults": True}})
        self._saldos = self._buscar_saldos()
        demo = self.saldo_demo()
        for saldo in self._saldos:
            if saldo.tipo not in (PRATICA, REAL):
                continue
            for instrumento in ("turbo-option", "binary-option"):
                self._enviar({
                    "name": "subscribeMessage", "request_id": self._rid(),
                    "msg": {"name": "portfolio.position-changed", "version": "2.0",
                            "params": {"routingFilters": {"instrument_type": instrumento,
                                                          "user_balance_id": saldo.id}}},
                })
        log.info("Conectado. Conta demo: %.2f %s", demo.valor, demo.moeda)

    def garantir(self) -> None:
        """Reconecta se a conexão caiu."""
        if not self._vivo:
            log.warning("Conexão caiu; reconectando...")
            self.conectar()

    # ------------------------------------------------------------ escolha da conta
    @property
    def conta(self) -> str:
        """'PRACTICE' (demo) ou 'REAL'."""
        return NOMES_CONTA[self._tipo_conta]

    def usar_conta_demo(self) -> None:
        self._tipo_conta = PRATICA

    def usar_conta_real(self, confirmacao: str) -> None:
        """Libera a conta real para esta conexão. Exige a frase digitada pelo usuário."""
        if confirmacao != FRASE_CONTA_REAL:
            raise ContaRealBloqueada("Conta real não liberada: a frase de confirmação não confere.")
        self._tipo_conta = REAL
        log.warning("CONTA REAL liberada pelo usuário para esta conexão.")

    def fechar(self) -> None:
        self._parar.set()
        self._vivo = False
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - fechar nunca deve derrubar o programa
                pass
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
        self._falhar_pendentes("conexão encerrada")

    def _login(self) -> str:
        url = f"https://auth.{self._host}/api/v2/login"
        try:
            resp = requests.post(url, json={"identifier": self._email, "password": self._senha},
                                 timeout=self._timeout)
        except requests.RequestException as exc:
            raise ErroConexao(f"Não consegui falar com o servidor de login: {type(exc).__name__}") from None
        try:
            dados = resp.json()
        except ValueError:
            dados = None
        if not isinstance(dados, dict):
            raise ErroLogin(f"Resposta inesperada do login (HTTP {resp.status_code}).")
        codigo = dados.get("code")
        if codigo == "success" and dados.get("ssid"):
            return str(dados["ssid"])
        if codigo == "verify":
            raise ErroLogin(
                "A conta pede verificação em duas etapas (código por SMS/e-mail). Esta versão ainda não "
                "trata esse passo: desative a verificação em duas etapas na IQ Option ou me avise para "
                "eu adicionar o suporte."
            )
        if codigo == "invalid_credentials":
            raise ErroLogin("E-mail ou senha incorretos. Tente de novo; se você usa o arquivo .env, confira "
                            "o que está nele.")
        raise ErroLogin(f"Login recusado ({codigo}): {dados.get('message', 'sem detalhe')}")

    def _abrir_ws(self) -> None:
        try:
            self._ws = websocket.create_connection(f"wss://{self._host}/echo/websocket", timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001
            raise ErroConexao(f"Não consegui abrir o websocket: {type(exc).__name__}: {exc}") from None
        self._ws.settimeout(30)  # o servidor manda timeSync a cada segundo; 30 s de silêncio = queda
        self._parar.clear()
        self._vivo = True
        self._thread = threading.Thread(target=self._leitor, name="iq-leitor", daemon=True)
        self._thread.start()

    def _autenticar(self, ssid: str) -> None:
        resp = self._requisitar({"name": "ssid", "msg": ssid}, nomes=("profile",))
        perfil = resp.get("msg")
        if not perfil:
            raise ErroLogin("O servidor recusou a sessão (ssid inválido).")
        if isinstance(perfil, dict):
            self._saldos = interpretar_saldos(perfil.get("balances"))

    # ------------------------------------------------------------ transporte
    def _rid(self) -> str:
        with self._trava:
            self._seq += 1
            return f"iqa{self._seq}"

    def _enviar(self, obj: dict) -> None:
        ws = self._ws
        if ws is None or not self._vivo:
            raise ErroConexao("Sem conexão com a corretora.")
        try:
            ws.send(json.dumps(obj))
        except Exception as exc:  # noqa: BLE001
            self._vivo = False
            raise ErroConexao(f"Falha ao enviar: {type(exc).__name__}") from None

    def _despachar(self, obj: dict, nomes: tuple[str, ...]) -> _Pendente:
        rid = self._rid()
        pend = _Pendente(nomes)
        with self._trava:
            self._pend[rid] = pend
        try:
            self._enviar({**obj, "request_id": rid})
        except ErroConexao:
            with self._trava:
                self._pend.pop(rid, None)
            raise
        return pend

    def _aguardar(self, pend: _Pendente, timeout: float | None = None) -> dict:
        if not pend.evento.wait(timeout or self._timeout):
            with self._trava:
                for rid, p in list(self._pend.items()):
                    if p is pend:
                        del self._pend[rid]
            raise ErroConexao(f"O servidor não respondeu a tempo ({'/'.join(pend.nomes)}).")
        if pend.erro:
            raise ErroIQ(pend.erro)
        return pend.resposta or {}

    def _requisitar(self, obj: dict, nomes: tuple[str, ...], timeout: float | None = None) -> dict:
        return self._aguardar(self._despachar(obj, nomes), timeout)

    def _mensagem(self, nome: str, versao: str, corpo: dict | None = None) -> dict:
        msg: dict[str, Any] = {"name": nome, "version": versao}
        if corpo is not None:
            msg["body"] = corpo
        return {"name": "sendMessage", "msg": msg}

    def _falhar_pendentes(self, motivo: str) -> None:
        with self._trava:
            pendentes, self._pend = list(self._pend.values()), {}
        for p in pendentes:
            p.erro = motivo
            p.evento.set()

    def _leitor(self) -> None:
        ws = self._ws
        while not self._parar.is_set() and ws is not None:
            try:
                bruto = ws.recv()
            except Exception as exc:  # noqa: BLE001
                if not self._parar.is_set():
                    log.warning("Conexão interrompida: %s", type(exc).__name__)
                break
            if not bruto:
                if not self._parar.is_set():
                    log.warning("O servidor fechou a conexão.")
                break
            try:
                self._tratar(bruto)
            except Exception:  # noqa: BLE001 - uma mensagem estranha não pode matar o leitor
                log.exception("Erro ao tratar mensagem do servidor")
        self._vivo = False
        self._falhar_pendentes("conexão caiu")

    def _tratar(self, bruto: str | bytes) -> None:
        try:
            m = json.loads(bruto)
        except (TypeError, ValueError):
            return
        if not isinstance(m, dict):
            return
        nome, corpo = m.get("name"), m.get("msg")
        if nome == "timeSync":
            try:
                self._srv_ms, self._srv_mono = float(corpo), time.monotonic()
            except (TypeError, ValueError):
                pass
            return
        if nome == "heartbeat":
            try:
                self._enviar({"name": "heartbeat", "msg": {"heartbeatTime": corpo,
                                                           "userTime": int(self.agora() * 1000)}})
            except ErroConexao:
                pass
            return
        if self.capturar and nome and nome not in self.amostras:
            self.amostras[str(nome)] = corpo
        fechamento = interpretar_fechamento(str(nome), corpo)
        if fechamento:
            self._fechamentos.setdefault(fechamento[0], fechamento[1])
        self._resolver(str(nome), str(m.get("request_id") or ""), m)

    def _resolver(self, nome: str, rid: str, m: dict) -> None:
        with self._trava:
            pend = self._pend.get(rid)
            if pend is not None:
                if nome in pend.nomes:
                    del self._pend[rid]
                elif nome == "result" and isinstance(m.get("msg"), dict) and m["msg"].get("success") is False:
                    del self._pend[rid]
                    pend.erro = f"Pedido recusado pelo servidor: {json.dumps(m['msg'])[:300]}"
                else:
                    return  # confirmação intermediária; a resposta de verdade vem depois
            else:
                # resposta sem request_id: entrega ao pedido mais antigo que espera esse nome
                for chave, candidato in self._pend.items():
                    if nome in candidato.nomes:
                        pend = candidato
                        del self._pend[chave]
                        break
                if pend is None:
                    return
        pend.resposta = m
        pend.evento.set()

    # ------------------------------------------------------------ dados
    def _buscar_saldos(self) -> list[Saldo]:
        try:
            resp = self._requisitar(self._mensagem("get-balances", "1.0"), nomes=("balances",), timeout=10)
            saldos = interpretar_saldos(resp.get("msg"))
            if saldos:
                return saldos
        except ErroIQ as exc:
            log.debug("get-balances falhou: %s", exc)
        return self._saldos  # o que veio no perfil

    def saldo_demo(self, atualizar: bool = False) -> Saldo:
        if atualizar or not self._saldos:
            self._saldos = self._buscar_saldos()
        for s in self._saldos:
            if s.tipo == PRATICA:
                return s
        raise ErroIQ("Não encontrei o saldo de prática (demo) nesta conta.")

    def saldo_ativo(self, atualizar: bool = False) -> Saldo:
        """Saldo da conta em uso (demo, a menos que o usuário tenha liberado a real)."""
        if atualizar or not self._saldos:
            self._saldos = self._buscar_saldos()
        for s in self._saldos:
            if s.tipo == self._tipo_conta:
                return s
        raise ErroIQ(f"Não encontrei o saldo da conta {self.conta}.")

    def ativos(self) -> list[Ativo]:
        resp = self._requisitar(self._mensagem("get-initialization-data", "3.0", {}),
                                nomes=("initialization-data",), timeout=40)
        return interpretar_ativos(resp.get("msg"))

    def _pedido_velas(self, ativo_id: int, tf: int, quantidade: int, ate: float | None) -> dict:
        return self._mensagem("get-candles", "2.0", {
            "active_id": int(ativo_id), "split_normalization": True, "size": int(tf),
            "to": int(ate if ate is not None else self.agora()),
            "count": int(min(quantidade, MAX_VELAS_POR_PEDIDO)),
        })

    def velas(self, ativo_id: int, tf: int = 60, quantidade: int = 1000, ate: float | None = None) -> list[tuple]:
        """Velas até o instante `ate` (inclui a vela ainda em formação, se houver)."""
        resp = self._requisitar(self._pedido_velas(ativo_id, tf, quantidade, ate), nomes=("candles",))
        return interpretar_velas(resp.get("msg"))

    def velas_varias(self, ativos_ids: list[int], tf: int = 60, quantidade: int = 1000,
                     ate: float | None = None) -> dict[int, Velas]:
        """Pede velas de vários ativos de uma vez (os pedidos saem juntos para ganhar tempo)."""
        pendentes = {a: self._despachar(self._pedido_velas(a, tf, quantidade, ate), ("candles",))
                     for a in ativos_ids}
        saida: dict[int, Velas] = {}
        for ativo_id, pend in pendentes.items():
            try:
                resp = self._aguardar(pend, timeout=8)
            except ErroIQ as exc:
                log.warning("Sem velas do ativo %s: %s", ativo_id, exc)
                continue
            saida[ativo_id] = Velas.de_linhas(interpretar_velas(resp.get("msg")), tf)
        return saida

    # ------------------------------------------------------------ operações
    def abrir_opcao(self, ativo: Ativo, direcao: str, valor: float, minutos: int) -> Ordem:
        """Abre uma opção turbo na conta em uso (demo por padrão; real só se liberada)."""
        saldo = self.saldo_ativo()
        if saldo.tipo != self._tipo_conta or saldo.tipo not in (PRATICA, REAL):  # redundante de propósito
            raise ContaRealBloqueada("Saldo não corresponde à conta escolhida.")
        if direcao not in ("call", "put"):
            raise ValueError("direcao precisa ser 'call' ou 'put'.")
        if valor <= 0:
            raise ValueError("valor precisa ser positivo.")
        agora = self.agora()
        expiracao = expiracao_turbo(agora, minutos)
        corpo = {"price": float(valor), "active_id": int(ativo.id), "expired": int(expiracao),
                 "direction": direcao, "option_type_id": OPCAO_TURBO, "user_balance_id": int(saldo.id)}
        resp = self._requisitar(self._mensagem("binary-options.open-option", "1.0", corpo),
                                nomes=("option",), timeout=10)
        msg = resp.get("msg")
        if not isinstance(msg, dict) or "id" not in msg:
            detalhe = msg.get("message") if isinstance(msg, dict) else msg
            raise OrdemRecusada(str(detalhe or "a corretora não informou o motivo"))
        return Ordem(id=int(msg["id"]), ativo_id=ativo.id, ativo=ativo.nome, direcao=direcao,
                     valor=float(valor), expiracao=int(expiracao), aberta_em=agora)

    def aguardar_resultado(self, ordem: Ordem, folga: float = 25.0) -> Resultado:
        """Espera a mensagem de encerramento. Se ela não vier, devolve 'desconhecido'
        (o executor então confere pelo saldo)."""
        limite = ordem.expiracao + folga
        while self.agora() < limite:
            resultado = self._fechamentos.get(ordem.id)
            if resultado is not None:
                return resultado
            if not self._vivo:
                break
            time.sleep(0.25)
        return self._fechamentos.get(ordem.id) or Resultado("desconhecido", 0.0, "sem-mensagem")
