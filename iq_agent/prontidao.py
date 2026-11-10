"""Prontidão para a conta real: o histórico na demo sustenta a mudança?

Os critérios vêm da seção [real] do config.toml. Só contam operações de estratégias
APROVADAS pela pesquisa (as de exploração não entram).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .banco import Banco
from .config import Config
from .estatistica import breakeven, wilson, z_unilateral

VALIDADE_PLANO_S = 36 * 3600


@dataclass(frozen=True)
class Item:
    nome: str
    ok: bool
    detalhe: str


def avaliar(banco: Banco, cfg: Config, agora: float) -> tuple[bool, list[Item]]:
    """Devolve (pronto, lista de critérios com o resultado de cada um)."""
    r = cfg.real
    ops = [o for o in banco.operacoes(desde=agora - r.janela_dias * 86400, conta="PRACTICE")
           if o["tipo"] == "aprovado" and o["status"] in ("win", "loss", "equal")]
    decididas = [o for o in ops if o["status"] in ("win", "loss")]
    n = len(decididas)
    vitorias = sum(o["status"] == "win" for o in decididas)
    dias = len({date.fromtimestamp(o["ts_sinal"]) for o in ops})
    lucro = sum(float(o["lucro"] or 0.0) for o in ops)

    if n:
        equilibrio = sum(breakeven(float(o["payout"])) for o in decididas) / n
        inferior = wilson(vitorias, n, z_unilateral(cfg.pesquisa.confianca))[0]
        taxa_ok = inferior > equilibrio
        taxa_txt = (f"{vitorias}/{n} = {vitorias / n:.1%}; no pior caso provável {inferior:.1%}; "
                    f"precisa passar de {equilibrio:.1%}")
    else:
        taxa_ok, taxa_txt = False, "ainda sem operações"

    pesquisa = banco.ultima_pesquisa()
    aprovadas = 0
    if pesquisa and agora - pesquisa["ts"] <= VALIDADE_PLANO_S:
        aprovadas = sum(e["tipo"] == "aprovado" for e in pesquisa["plano"].get("entradas", []))

    itens = [
        Item("Operações na demo com estratégias aprovadas", n >= r.min_operacoes_demo,
             f"{n} de {r.min_operacoes_demo}"),
        Item("Dias diferentes operando na demo", dias >= r.min_dias_demo, f"{dias} de {r.min_dias_demo}"),
        Item("Taxa de acerto na demo acima do breakeven, com confiança", taxa_ok, taxa_txt),
        Item("Resultado positivo na demo", lucro > 0, f"{lucro:+.2f}"),
        Item("Plano atual com estratégia aprovada", aprovadas > 0,
             f"{aprovadas} aprovada(s)" if aprovadas else "nenhuma (ou plano com mais de 36 h)"),
    ]
    return all(i.ok for i in itens), itens
