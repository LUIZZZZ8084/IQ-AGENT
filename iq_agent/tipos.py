"""Tipos compartilhados entre a conexão, o executor e o banco."""
from __future__ import annotations

from dataclasses import dataclass

# Tipos de saldo usados pela corretora.
REAL = 1
TORNEIO = 2
PRATICA = 4

NOMES_CONTA = {REAL: "REAL", TORNEIO: "TORNEIO", PRATICA: "PRACTICE"}

# Frase que o usuário precisa digitar, a cada execução, para liberar a conta real.
FRASE_CONTA_REAL = "OPERAR NA CONTA REAL"


class ErroIQ(Exception):
    """Erro genérico ao falar com a corretora."""


class ErroLogin(ErroIQ):
    """Login recusado ou incompleto (senha errada, verificação em duas etapas...)."""


class ErroConexao(ErroIQ):
    """Conexão caiu ou o servidor não respondeu a tempo."""


class OrdemRecusada(ErroIQ):
    """A corretora recusou a abertura da operação."""


class ContaRealBloqueada(ErroIQ):
    """Tentativa de usar a conta real sem a liberação manual do usuário."""


@dataclass(frozen=True)
class Saldo:
    id: int
    tipo: int
    valor: float
    moeda: str

    @property
    def eh_demo(self) -> bool:
        return self.tipo == PRATICA

    @property
    def nome_conta(self) -> str:
        return NOMES_CONTA.get(self.tipo, f"TIPO{self.tipo}")


@dataclass(frozen=True)
class Ativo:
    id: int
    nome: str
    tipo: str  # "turbo" (1 a 5 min) ou "binary" (15 min ou mais)
    payout: float  # 0.85 = paga 85% sobre o valor investido
    aberto: bool
    aposta_min: float | None = None
    aposta_max: float | None = None

    @property
    def eh_otc(self) -> bool:
        return self.nome.upper().endswith("-OTC")


@dataclass(frozen=True)
class Ordem:
    id: int
    ativo_id: int
    ativo: str
    direcao: str  # "call" ou "put"
    valor: float
    expiracao: int  # timestamp (s) de expiração
    aberta_em: float  # horário do servidor (s) no envio


@dataclass(frozen=True)
class Resultado:
    status: str  # "win", "loss", "equal" ou "desconhecido"
    lucro: float  # líquido: positivo no ganho, negativo na perda
    fonte: str  # de onde veio a informação ("mensagem", "saldo", ...)
