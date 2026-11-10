"""Coleta diária: lista de ativos, payouts e velas de 1 minuto."""
from __future__ import annotations

from typing import Callable

from .banco import Banco
from .config import Coleta, Config
from .tipos import Ativo, ErroIQ

TF = 60


def escolher_ativos(ativos: list[Ativo], cfg: Coleta, ja_acompanhados: set[str] = frozenset()) -> list[Ativo]:
    """Ativos turbo a acompanhar: os abertos agora (maior payout primeiro) e os que já têm histórico."""
    turbo = [a for a in ativos if a.tipo == "turbo"]
    if cfg.ativos:
        desejados = {n.upper() for n in cfg.ativos}
        return [a for a in turbo if a.nome.upper() in desejados]
    bons = [a for a in turbo if a.payout >= cfg.payout_minimo and (cfg.incluir_otc or not a.eh_otc)]
    candidatos = [a for a in bons if a.aberto or a.nome in ja_acompanhados]
    candidatos.sort(key=lambda a: (not a.aberto, -a.payout, a.nome))
    return candidatos[: cfg.max_ativos]


def _baixar_intervalo(banco: Banco, cliente, ativo: Ativo, alvo: int, ate: float, agora: float,
                      pausa: float) -> int:
    """Baixa velas fechadas do intervalo [alvo, ate], andando para trás em lotes."""
    gravadas, anterior = 0, None
    while ate > alvo:
        lote = cliente.velas(ativo.id, TF, 1000, ate)
        if not lote:
            break
        gravadas += banco.inserir_velas(ativo.nome, TF, [l for l in lote if alvo <= l[0] and l[0] + TF <= agora])
        mais_antigo = min(l[0] for l in lote)
        if mais_antigo <= alvo or (anterior is not None and mais_antigo >= anterior):
            break
        anterior, ate = mais_antigo, mais_antigo - 1
        cliente.dormir_ate(cliente.agora() + pausa)
    return gravadas


def baixar_ativo(banco: Banco, cliente, ativo: Ativo, inicio_janela: int, agora: float, pausa: float) -> int:
    """Completa o histórico do ativo: primeiro o trecho recente que falta, depois o antigo."""
    primeiro, ultimo, _ = banco.intervalo_velas(ativo.nome, TF)
    if ultimo is None:
        return _baixar_intervalo(banco, cliente, ativo, inicio_janela, agora, agora, pausa)
    gravadas = _baixar_intervalo(banco, cliente, ativo, max(inicio_janela, ultimo + TF), agora, agora, pausa)
    if primeiro - inicio_janela > 3600:  # coleta anterior foi interrompida: completa o começo
        gravadas += _baixar_intervalo(banco, cliente, ativo, inicio_janela, primeiro - 1, agora, pausa)
    return gravadas


def coletar(banco: Banco, cfg: Config, cliente, log: Callable[[str], None] = print) -> dict:
    agora = cliente.agora()
    todos = cliente.ativos()
    if not todos:
        raise ErroIQ("A corretora não devolveu nenhum ativo turbo/binary. Rode o diagnóstico.")
    banco.salvar_ativos(todos, agora)
    escolhidos = escolher_ativos(todos, cfg.coleta, set(banco.ativos_com_velas(TF)))
    abertos = sum(a.aberto for a in escolhidos)
    log(f"{len(todos)} ativos listados; acompanhando {len(escolhidos)} ({abertos} abertos agora).")
    inicio_janela = int(agora - cfg.coleta.dias_historico * 86400)
    total, falhas = 0, 0
    for i, ativo in enumerate(escolhidos, 1):
        try:
            n = baixar_ativo(banco, cliente, ativo, inicio_janela, agora, cfg.coleta.pausa_entre_pedidos)
        except ErroIQ as exc:
            falhas += 1
            log(f"  [{i}/{len(escolhidos)}] {ativo.nome}: falhou ({exc})")
            cliente.garantir()
            continue
        total += n
        log(f"  [{i}/{len(escolhidos)}] {ativo.nome}: +{n} velas (payout {ativo.payout:.0%}"
            f"{'' if ativo.aberto else ', fechado agora'})")
    banco.apagar_velas_antes(inicio_janela - 5 * 86400)
    return {"ativos": len(escolhidos), "abertos": abertos, "velas_novas": total, "falhas": falhas}
