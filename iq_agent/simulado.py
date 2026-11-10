"""Corretora simulada: mesma interface do IQClient, mas sobre velas já conhecidas.

Usada nos testes automáticos e no autoteste. O relógio é virtual: `dormir_ate` avança
o tempo na hora, sem esperar de verdade. Nada aqui fala com a corretora de verdade.
"""
from __future__ import annotations

import numpy as np

from .iqclient import expiracao_turbo
from .tipos import (FRASE_CONTA_REAL, NOMES_CONTA, PRATICA, REAL, Ativo, ContaRealBloqueada, ErroIQ, Ordem,
                    OrdemRecusada, Resultado, Saldo)
from .velas import Velas


class ClienteSimulado:
    def __init__(self, series: dict[str, Velas], payouts: dict[str, float] | None = None,
                 inicio_ts: float | None = None, saldo: float = 10_000.0, saldo_real: float = 1_000.0,
                 enviar_resultado: bool = True):
        self._series = series
        self._ativos = [
            Ativo(id=i + 1, nome=nome, tipo="turbo", payout=(payouts or {}).get(nome, 0.85), aberto=True,
                  aposta_min=1.0, aposta_max=20_000.0)
            for i, nome in enumerate(sorted(series))
        ]
        self._por_id = {a.id: a for a in self._ativos}
        primeiro = min(int(v.ts[0]) for v in series.values())
        self._t = float(inicio_ts if inicio_ts is not None else primeiro)
        self._saldos = {PRATICA: float(saldo), REAL: float(saldo_real)}
        self._tipo_conta = PRATICA
        self._ordens: dict[int, tuple[Ordem, float, float, int]] = {}  # id -> (ordem, entrada, payout, conta)
        self._seq = 0
        self.enviar_resultado = enviar_resultado  # False simula a mensagem de resultado que não chega
        self.conectado = False

    # ------------------------------------------------------------ relógio e conexão
    def agora(self) -> float:
        return self._t

    def dormir_ate(self, instante: float) -> None:
        self._t = max(self._t, float(instante))

    def conectar(self) -> None:
        self.conectado = True

    def garantir(self) -> None:
        self.conectado = True

    def fechar(self) -> None:
        self.conectado = False

    # ------------------------------------------------------------ escolha da conta
    @property
    def conta(self) -> str:
        return NOMES_CONTA[self._tipo_conta]

    def usar_conta_demo(self) -> None:
        self._tipo_conta = PRATICA

    def usar_conta_real(self, confirmacao: str) -> None:
        if confirmacao != FRASE_CONTA_REAL:
            raise ContaRealBloqueada("Conta real não liberada: a frase de confirmação não confere.")
        self._tipo_conta = REAL

    # ------------------------------------------------------------ dados
    def _saldo(self, tipo: int) -> Saldo:
        self._liquidar()
        return Saldo(id=tipo, tipo=tipo, valor=round(self._saldos[tipo], 2), moeda="USD")

    def saldo_demo(self, atualizar: bool = False) -> Saldo:
        return self._saldo(PRATICA)

    def saldo_ativo(self, atualizar: bool = False) -> Saldo:
        return self._saldo(self._tipo_conta)

    def ativos(self) -> list[Ativo]:
        return list(self._ativos)

    def _serie(self, ativo_id: int) -> Velas:
        if ativo_id not in self._por_id:
            raise ErroIQ(f"ativo {ativo_id} desconhecido")
        return self._series[self._por_id[ativo_id].nome]

    def _preco(self, ativo_id: int, instante: float) -> float:
        """Preço no instante: fechamento da vela que termina exatamente ali, ou a abertura
        da vela em andamento."""
        v = self._serie(ativo_id)
        fins = v.ts + v.tf
        j = int(np.searchsorted(fins, instante, side="left"))
        if j >= len(v):
            return float(v.close[-1])
        if fins[j] == instante:
            return float(v.close[j])
        if v.ts[j] <= instante:
            return float(v.open[j])
        if j == 0:
            raise ErroIQ("sem preço antes do início da série")
        return float(v.close[j - 1])

    def velas(self, ativo_id: int, tf: int = 60, quantidade: int = 1000, ate: float | None = None) -> list[tuple]:
        v = self._serie(ativo_id)
        ate = self._t if ate is None else ate
        fim = int(np.searchsorted(v.ts, ate, side="right"))  # inclui a vela em formação
        ini = max(0, fim - quantidade)
        return list(zip(v.ts[ini:fim].tolist(), v.open[ini:fim].tolist(), v.high[ini:fim].tolist(),
                        v.low[ini:fim].tolist(), v.close[ini:fim].tolist(), v.volume[ini:fim].tolist()))

    def velas_varias(self, ativos_ids: list[int], tf: int = 60, quantidade: int = 1000,
                     ate: float | None = None) -> dict[int, Velas]:
        return {a: Velas.de_linhas(self.velas(a, tf, quantidade, ate), tf) for a in ativos_ids}

    # ------------------------------------------------------------ operações
    def abrir_opcao(self, ativo: Ativo, direcao: str, valor: float, minutos: int) -> Ordem:
        conta = self._tipo_conta
        if valor > self._saldos[conta]:
            raise OrdemRecusada("saldo insuficiente")
        self._seq += 1
        expiracao = expiracao_turbo(self._t, minutos)
        ordem = Ordem(id=self._seq, ativo_id=ativo.id, ativo=ativo.nome, direcao=direcao, valor=float(valor),
                      expiracao=expiracao, aberta_em=self._t)
        self._saldos[conta] -= valor
        self._ordens[ordem.id] = (ordem, self._preco(ativo.id, self._t), self._por_id[ativo.id].payout, conta)
        return ordem

    def _resultado(self, ordem: Ordem, entrada: float, payout: float) -> Resultado:
        saida = self._preco(ordem.ativo_id, ordem.expiracao)
        diferenca = (saida - entrada) if ordem.direcao == "call" else (entrada - saida)
        if diferenca > 0:
            return Resultado("win", round(ordem.valor * payout, 2), "simulado")
        if diferenca == 0:
            return Resultado("equal", 0.0, "simulado")
        return Resultado("loss", -ordem.valor, "simulado")

    def _liquidar(self) -> None:
        for chave in [k for k, (o, _, _, _) in self._ordens.items() if o.expiracao <= self._t]:
            ordem, entrada, payout, conta = self._ordens.pop(chave)
            self._saldos[conta] += ordem.valor + self._resultado(ordem, entrada, payout).lucro

    def aguardar_resultado(self, ordem: Ordem, folga: float = 25.0) -> Resultado:
        self.dormir_ate(ordem.expiracao + 1)
        if ordem.id not in self._ordens:
            return Resultado("desconhecido", 0.0, "sem-mensagem")
        _, entrada, payout, _ = self._ordens[ordem.id]
        resultado = self._resultado(ordem, entrada, payout)
        self._liquidar()
        if not self.enviar_resultado:
            return Resultado("desconhecido", 0.0, "sem-mensagem")
        return resultado
