"""Testa o cliente da corretora com um websocket falso (nenhuma conexão real é feita)."""
import json

import pytest

from iq_agent.iqclient import (IQClient, expiracao_turbo, formato, interpretar_ativos, interpretar_fechamento,
                               interpretar_saldos, interpretar_velas)
from iq_agent.tipos import FRASE_CONTA_REAL, ContaRealBloqueada, ErroIQ, OrdemRecusada

SALDOS = [{"id": 111, "type": 1, "amount": 250.0, "currency": "BRL"},
          {"id": 444, "type": 4, "amount": 10000.0, "currency": "BRL"}]
INIT = {"turbo": {"actives": {
    "1": {"id": 1, "name": "front.EURUSD", "enabled": True, "is_suspended": False, "minimal_bet": 1,
          "maximal_bet": 20000, "option": {"profit": {"commission": 15}}},
    "76": {"id": 76, "name": "front.EURUSD-OTC", "enabled": True, "is_suspended": True,
           "option": {"profit": {"commission": 20}}},
    "9": {"id": 9, "name": "quebrado"},
}}, "binary": {"actives": {"1": {"id": 1, "name": "front.EURUSD", "enabled": False,
                                 "option": {"profit": {"commission": 30}}}}}}


class WsFalso:
    """Responde na hora a cada pedido, como o servidor faria."""

    def __init__(self, cliente: IQClient, resposta_ordem: dict | None = None):
        self.cliente = cliente
        self.enviados: list[dict] = []
        self.resposta_ordem = resposta_ordem or {"id": 987654}

    def send(self, texto: str) -> None:
        pedido = json.loads(texto)
        self.enviados.append(pedido)
        rid = pedido.get("request_id")
        msg = pedido.get("msg")
        nome = msg.get("name") if isinstance(msg, dict) else None
        if pedido["name"] != "sendMessage":
            return
        self._responder({"name": "result", "request_id": rid, "msg": {"success": True}})  # confirmação intermediária
        if nome == "get-balances":
            self._responder({"name": "balances", "request_id": rid, "msg": SALDOS})
        elif nome == "get-initialization-data":
            self._responder({"name": "initialization-data", "request_id": rid, "msg": INIT})
        elif nome == "get-candles":
            ate = msg["body"]["to"]
            velas = [{"id": i, "from": ate - ate % 60 - 60 * (2 - i), "to": 0, "open": 1.1, "close": 1.2,
                      "min": 1.0, "max": 1.3, "volume": 5} for i in range(3)]
            self._responder({"name": "candles", "request_id": rid, "msg": {"candles": velas}})
        elif nome == "binary-options.open-option":
            self._responder({"name": "option", "request_id": rid, "msg": self.resposta_ordem})

    def _responder(self, obj: dict) -> None:
        self.cliente._tratar(json.dumps(obj))

    def close(self) -> None:
        pass


@pytest.fixture
def cliente():
    c = IQClient("teste@exemplo.com", "senha-de-teste", timeout=2)
    c._ws = WsFalso(c)
    c._vivo = True
    c._tratar(json.dumps({"name": "timeSync", "msg": 1_794_000_005_000}))
    return c


def ultimo_pedido(cliente, nome):
    return next(p for p in reversed(cliente._ws.enviados)
                if isinstance(p.get("msg"), dict) and p["msg"].get("name") == nome)


def test_expiracao_turbo():
    base = 1_794_000_000  # minuto cheio
    assert expiracao_turbo(base + 1, 1) == base + 60
    assert expiracao_turbo(base + 29, 1) == base + 60
    assert expiracao_turbo(base + 30, 1) == base + 120  # faltam só 30 s: vai para a expiração seguinte
    assert expiracao_turbo(base + 1, 5) == base + 300
    with pytest.raises(ValueError):
        expiracao_turbo(base, 6)


def test_interpretar_velas_ignora_linhas_quebradas():
    msg = {"candles": [{"from": 60, "open": 1, "close": 2, "min": 0.5, "max": 2.5, "volume": 3},
                       {"from": 120, "open": "x"}, {"open": 1}]}
    assert interpretar_velas(msg) == [(60, 1.0, 2.5, 0.5, 2.0, 3.0)]
    assert interpretar_velas(None) == [] and interpretar_velas({}) == []


def test_interpretar_ativos():
    ativos = {(a.nome, a.tipo): a for a in interpretar_ativos(INIT)}
    eurusd = ativos[("EURUSD", "turbo")]
    assert eurusd.id == 1 and eurusd.payout == 0.85 and eurusd.aberto and eurusd.aposta_min == 1.0
    assert not ativos[("EURUSD-OTC", "turbo")].aberto and ativos[("EURUSD-OTC", "turbo")].eh_otc
    assert not ativos[("EURUSD", "binary")].aberto
    assert len(ativos) == 3  # o ativo sem dados de payout é ignorado
    assert interpretar_ativos(None) == []


def test_interpretar_saldos():
    saldos = interpretar_saldos(SALDOS + [{"id": "x"}])
    assert [(s.id, s.nome_conta, s.eh_demo) for s in saldos] == [(111, "REAL", False), (444, "PRACTICE", True)]


def test_interpretar_fechamento_nos_tres_formatos():
    ident, r = interpretar_fechamento("socket-option-closed", {"id": 5, "win": "win", "sum": 10, "win_amount": 18.5})
    assert (ident, r.status, r.lucro) == (5, "win", 8.5)
    ident, r = interpretar_fechamento("option-closed", {"option_id": "6", "result": "loose", "amount": 10,
                                                        "profit_amount": 0})
    assert (ident, r.status, r.lucro) == (6, "loss", -10)
    ident, r = interpretar_fechamento("position-changed", {"external_id": 7, "status": "closed", "invest": 10,
                                                           "close_profit": 10})
    assert (ident, r.status, r.lucro) == (7, "equal", 0.0)
    assert interpretar_fechamento("position-changed", {"external_id": 7, "status": "open"}) is None
    assert interpretar_fechamento("socket-option-closed", {"id": 5, "win": "???"}) is None
    assert interpretar_fechamento("outra-coisa", {"id": 1}) is None
    assert interpretar_fechamento("option-closed", "texto") is None


def test_formato_nao_guarda_valores():
    assert formato({"email": "a@b.c", "saldos": [{"amount": 10.5}]}) == {
        "email": "str", "saldos": [{"amount": "float"}, "... 1 itens"]}


def test_relogio_segue_o_servidor(cliente):
    assert abs(cliente.agora() - 1_794_000_005) < 1.0


def test_velas_ativos_e_saldos(cliente):
    velas = cliente.velas(1, 60, 3)
    assert len(velas) == 3 and velas[-1][0] == 1_794_000_000
    corpo = ultimo_pedido(cliente, "get-candles")["msg"]["body"]
    assert corpo["active_id"] == 1 and corpo["size"] == 60 and corpo["count"] == 3
    assert {a.nome for a in cliente.ativos()} == {"EURUSD", "EURUSD-OTC"}
    assert cliente.saldo_demo(atualizar=True).id == 444
    varias = cliente.velas_varias([1, 76], 60, 3)
    assert set(varias) == {1, 76} and len(varias[1]) == 3


def test_ordem_vai_para_a_conta_demo_por_padrao(cliente):
    eurusd = next(a for a in cliente.ativos() if a.nome == "EURUSD")
    ordem = cliente.abrir_opcao(eurusd, "call", 2.5, 1)
    corpo = ultimo_pedido(cliente, "binary-options.open-option")["msg"]["body"]
    assert cliente.conta == "PRACTICE"
    assert corpo["user_balance_id"] == 444  # id do saldo de prática, nunca o real
    assert corpo == {"price": 2.5, "active_id": 1, "expired": 1_794_000_060, "direction": "call",
                     "option_type_id": 3, "user_balance_id": 444}
    assert ordem.id == 987654 and ordem.expiracao == 1_794_000_060


def test_conta_real_exige_a_frase_exata(cliente):
    eurusd = next(a for a in cliente.ativos() if a.nome == "EURUSD")
    for errada in ("", "sim", FRASE_CONTA_REAL.lower(), FRASE_CONTA_REAL + " "):
        with pytest.raises(ContaRealBloqueada):
            cliente.usar_conta_real(errada)
        assert cliente.conta == "PRACTICE"
    cliente.usar_conta_real(FRASE_CONTA_REAL)
    assert cliente.conta == "REAL" and cliente.saldo_ativo().id == 111
    cliente.abrir_opcao(eurusd, "put", 1.0, 1)
    assert ultimo_pedido(cliente, "binary-options.open-option")["msg"]["body"]["user_balance_id"] == 111
    cliente.usar_conta_demo()
    cliente.abrir_opcao(eurusd, "put", 1.0, 1)
    assert ultimo_pedido(cliente, "binary-options.open-option")["msg"]["body"]["user_balance_id"] == 444


def test_ordem_recusada_e_parametros_invalidos(cliente):
    eurusd = next(a for a in cliente.ativos() if a.nome == "EURUSD")
    cliente._ws.resposta_ordem = {"message": "Asset is closed"}
    with pytest.raises(OrdemRecusada, match="Asset is closed"):
        cliente.abrir_opcao(eurusd, "call", 1.0, 1)
    with pytest.raises(ValueError):
        cliente.abrir_opcao(eurusd, "compra", 1.0, 1)
    with pytest.raises(ValueError):
        cliente.abrir_opcao(eurusd, "call", 0, 1)


def test_resultado_chega_por_mensagem(cliente):
    eurusd = next(a for a in cliente.ativos() if a.nome == "EURUSD")
    ordem = cliente.abrir_opcao(eurusd, "call", 10, 1)
    cliente._tratar(json.dumps({"name": "socket-option-closed",
                                "msg": {"id": ordem.id, "win": "win", "sum": 10, "win_amount": 18.5}}))
    resultado = cliente.aguardar_resultado(ordem, folga=0)
    assert (resultado.status, resultado.lucro) == ("win", 8.5)


def test_sem_resposta_do_servidor_vira_erro_claro(cliente):
    cliente._ws.send = lambda texto: None  # servidor mudo
    with pytest.raises(ErroIQ, match="não respondeu"):
        cliente.velas(1, 60, 3)
    assert cliente._pend == {}  # o pedido pendente foi limpo


def test_pedido_recusado_pelo_servidor(cliente):
    def recusar(texto):
        rid = json.loads(texto)["request_id"]
        cliente._tratar(json.dumps({"name": "result", "request_id": rid, "msg": {"success": False, "reason": "x"}}))

    cliente._ws.send = recusar
    with pytest.raises(ErroIQ, match="recusado"):
        cliente.velas(1, 60, 3)


def test_mensagens_estranhas_nao_derrubam_o_leitor(cliente):
    for lixo in ("", "não é json", "[1, 2]", '{"name": "timeSync", "msg": "abc"}', '{"sem": "nome"}'):
        cliente._tratar(lixo)
    assert cliente.conectado
