"""Gestão de risco: quanto arriscar por operação e quando parar.

Não existe martingale aqui. O valor nunca aumenta depois de uma perda.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Operacao, Risco
from .estatistica import kelly


@dataclass
class Decisao:
    pode: bool
    motivo: str = ""
    encerrar_dia: bool = False


class GestorRisco:
    def __init__(self, risco: Risco, operacao: Operacao, saldo_inicial_dia: float):
        self.cfg = risco
        self.op = operacao
        self.saldo_inicial = float(saldo_inicial_dia)
        self.operacoes = 0
        self.lucro = 0.0
        self.perdas_seguidas = 0
        self.pausa_ate = 0.0

    @classmethod
    def do_historico(cls, risco: Risco, operacao: Operacao, saldo_atual: float,
                     operacoes_hoje: list[dict]) -> "GestorRisco":
        """Reconstrói o estado do dia a partir do diário (para o caso de o bot ser reiniciado)."""
        feitas = [o for o in operacoes_hoje if o.get("status") in ("win", "loss", "equal")]
        lucro = sum(float(o.get("lucro") or 0.0) for o in feitas)
        gestor = cls(risco, operacao, saldo_atual - lucro)
        gestor.operacoes = len(feitas)
        gestor.lucro = lucro
        for o in reversed(feitas):
            if o["status"] != "loss":
                break
            gestor.perdas_seguidas += 1
        return gestor

    def pode_operar(self, agora: float) -> Decisao:
        c = self.cfg
        if self.operacoes >= c.max_operacoes_dia:
            return Decisao(False, f"limite de {c.max_operacoes_dia} operações no dia atingido", True)
        if self.lucro <= -c.perda_maxima_diaria * self.saldo_inicial:
            return Decisao(False, f"perda máxima diária atingida ({self.lucro:+.2f})", True)
        if c.meta_diaria > 0 and self.lucro >= c.meta_diaria * self.saldo_inicial:
            return Decisao(False, f"meta diária atingida ({self.lucro:+.2f})", True)
        if agora < self.pausa_ate:
            return Decisao(False, f"em pausa após {c.max_perdas_seguidas} perdas seguidas "
                                  f"(faltam {math.ceil((self.pausa_ate - agora) / 60)} min)")
        return Decisao(True)

    def valor(self, saldo: float, p_inf: float | None, payout: float, exploracao: bool = False,
              aposta_min: float | None = None) -> tuple[float, str]:
        """Valor da operação. Devolve (0, motivo) quando não deve operar.

        Entradas aprovadas: Kelly fracionado calculado com a taxa de acerto PESSIMISTA
        (limite inferior do intervalo), limitado ao teto por operação.
        Exploração (só demo): valor fixo mínimo.
        """
        c = self.cfg
        teto = min(c.risco_por_operacao * saldo, c.valor_maximo)
        minimo = max(c.valor_minimo, aposta_min or 0.0)
        if teto < minimo:
            return 0.0, (f"saldo pequeno demais: o teto de risco ({teto:.2f}) é menor que a aposta "
                         f"mínima ({minimo:.2f})")
        if exploracao:
            valor = max(minimo, self.op.valor_exploracao)
            if valor > teto:
                return 0.0, f"valor de exploração ({valor:.2f}) acima do teto de risco ({teto:.2f})"
            return round(valor, 2), "exploração: valor fixo"
        fracao = kelly(p_inf or 0.0, payout) * c.kelly_fracao
        if fracao <= 0:
            return 0.0, "sem vantagem com o payout atual"
        valor = min(fracao * saldo, teto)
        if valor < minimo:
            return round(minimo, 2), "Kelly abaixo da aposta mínima: usando a mínima"
        return math.floor(valor * 100) / 100, "Kelly fracionado"

    def registrar(self, status: str, lucro: float, agora: float) -> None:
        self.operacoes += 1
        self.lucro += lucro
        if status == "loss":
            self.perdas_seguidas += 1
            if self.perdas_seguidas >= self.cfg.max_perdas_seguidas:
                self.pausa_ate = agora + self.cfg.pausa_apos_perdas_min * 60
                self.perdas_seguidas = 0
        elif status == "win":
            self.perdas_seguidas = 0
