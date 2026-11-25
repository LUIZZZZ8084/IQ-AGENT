"""Ficha de cada estratégia: a memória do bot de um dia para o outro.

A pesquisa diária olha só as velas da janela de histórico e não lembra do que concluiu
antes. A ficha guarda duas coisas que ficam de fora dela:

  1. O que as pesquisas dos dias anteriores disseram da estratégia. Uma aprovação só
     passa a valer quando se repete em pesquisas de dias diferentes: um dia bom isolado
     costuma ser sorte do recorte das janelas de teste. As janelas de dias seguidos se
     sobrepõem quase inteiras, então a repetição não é prova independente; ela confirma
     que o resultado não depende do dia exato em que o teste foi cortado.

  2. Como ela foi ao vivo. Operações ao vivo acontecem em mercado que a estratégia nunca
     viu, por isso são a prova mais honesta. Elas entram na conta de duas formas:
       - somadas ao placar do teste, mudam o "pior caso" usado no tamanho da operação e
         na ordem das candidatas;
       - se o placar ao vivo é bem pior do que o teste prometia, a estratégia sai do plano.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .estatistica import p_valor_binomial, valor_esperado, wilson

DIAS_FICHA = 10  # quantas pesquisas anteriores (a última de cada dia) a ficha lembra
MIN_OPS_COMPARAR = 10  # operações ao vivo antes de comparar com o que o teste prometia
P_PIOR_AO_VIVO = 0.05  # chance máxima de um placar ao vivo tão ruim acontecer por azar


@dataclass
class Ficha:
    ativo: str
    modo: str
    estrategia: str
    historico: list[dict] = field(default_factory=list)  # pesquisas anteriores, da mais antiga à mais recente
    vivo_vitorias: int = 0
    vivo_derrotas: int = 0
    vivo_empates: int = 0
    vivo_lucro: float = 0.0

    @property
    def vivo_n(self) -> int:
        return self.vivo_vitorias + self.vivo_derrotas

    def aprovacoes_seguidas(self) -> int:
        """Em quantas das últimas pesquisas (dias diferentes, contando da mais recente) ela foi aprovada."""
        seguidas = 0
        for dia in reversed(self.historico):
            if not dia["aprovado"]:
                break
            seguidas += 1
        return seguidas

    def dias_acima(self) -> int:
        """Em quantas pesquisas anteriores o pior caso do teste ficou acima do necessário."""
        return sum(1 for dia in self.historico if dia["acima"])


def chave(d) -> tuple[str, str, str]:
    """(ativo, modo, estratégia) de uma hipótese, entrada ou operação (dict ou objeto)."""
    if isinstance(d, dict):
        return d["ativo"], d["modo"], d["estrategia"]
    return d.ativo, d.modo, d.estrategia


def _dia(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def historico_de_pesquisas(banco, agora: float, dias: int = DIAS_FICHA) -> tuple[dict, list[str]]:
    """O que as pesquisas de dias ANTERIORES ao de `agora` disseram de cada hipótese.

    Usa a última pesquisa de cada dia. Pesquisas refeitas no mesmo dia de hoje ficam de
    fora: elas olham quase os mesmos dados e não confirmam nada.
    Devolve ({chave: [resumo por dia, do mais antigo ao mais recente]}, dias usados).
    """
    hoje = _dia(agora)
    vistos: list[str] = []
    por_chave: dict[tuple, list[dict]] = {}
    for ts, carregar in banco.pesquisas_anteriores(agora):
        dia = _dia(ts)
        if dia >= hoje or dia in vistos:
            continue
        vistos.append(dia)
        for h in carregar():
            acima = bool(h.get("testavel") and h.get("wilson_inf") is not None
                         and h["wilson_inf"] > h["breakeven"])
            por_chave.setdefault(chave(h), []).append({
                "dia": dia, "aprovado": bool(h.get("aprovado")), "acima": acima,
                "taxa": h.get("taxa"), "n": int(h.get("n") or 0),
            })
        if len(vistos) >= dias:
            break
    for lista in por_chave.values():
        lista.reverse()  # as pesquisas vieram da mais recente para a mais antiga
    vistos.reverse()
    # Uma hipótese que não apareceu num dia (ativo sem dados, por exemplo) conta como não aprovada.
    for k, lista in por_chave.items():
        presentes = {d["dia"] for d in lista}
        for dia in vistos:
            if dia not in presentes:
                lista.append({"dia": dia, "aprovado": False, "acima": False, "taxa": None, "n": 0})
        lista.sort(key=lambda d: d["dia"])
    return por_chave, vistos


def placar_ao_vivo(banco, desde: float) -> dict[tuple, list]:
    """Placar ao vivo por (ativo, modo, estratégia), nas duas contas: o mercado é o mesmo.

    Operações do modelo de aprendizado de máquina contam para o grupo do ativo, que é
    como a pesquisa julga o modelo.
    """
    from .ml import NOME as MODELO, grupo_de

    placar: dict[tuple, list] = {}
    for o in banco.operacoes(desde=desde):
        if o["status"] not in ("win", "loss", "equal"):
            continue
        ativo = grupo_de(o["ativo"]) if o["estrategia"] == MODELO else o["ativo"]
        p = placar.setdefault((ativo, o["modo"], o["estrategia"]), [0, 0, 0, 0.0])
        p[{"win": 0, "loss": 1, "equal": 2}[o["status"]]] += 1
        p[3] += float(o["lucro"] or 0.0)
    return placar


def montar_fichas(banco, agora: float, dias_vivo: float, dias: int = DIAS_FICHA) -> tuple[dict, list[str]]:
    """Fichas de todas as hipóteses com histórico ou operações ao vivo."""
    historico, vistos = historico_de_pesquisas(banco, agora, dias)
    vivo = placar_ao_vivo(banco, agora - dias_vivo * 86400)
    fichas: dict[tuple, Ficha] = {}
    for k in set(historico) | set(vivo):
        v, d, e, lucro = vivo.get(k, (0, 0, 0, 0.0))
        fichas[k] = Ficha(*k, historico=historico.get(k, []), vivo_vitorias=v, vivo_derrotas=d,
                          vivo_empates=e, vivo_lucro=lucro)
    return fichas, vistos


def julgar_ao_vivo(vitorias: int, n: int, taxa_teste: float | None, breakeven: float, z: float,
                   min_ops_abaixo: int) -> str | None:
    """Motivo para tirar a estratégia do plano pelo placar ao vivo, ou None se ela segue.

    Sai quando (a) acertar tão pouco ao vivo seria muito improvável se a taxa real fosse a
    que o teste mostrou, ou (b) com operações suficientes, até o melhor caso ao vivo fica
    abaixo do necessário para não perder dinheiro.
    """
    if n <= 0:
        return None
    if n >= MIN_OPS_COMPARAR and taxa_teste:
        chance = 1.0 - p_valor_binomial(vitorias + 1, n, taxa_teste)  # P(acertar `vitorias` ou menos)
        if chance < P_PIOR_AO_VIVO:
            return (f"ao vivo acertou {vitorias} de {n} ({vitorias / n:.0%}), bem abaixo dos "
                    f"{taxa_teste:.0%} do teste")
    if n >= min_ops_abaixo and wilson(vitorias, n, z)[1] < breakeven:
        return f"ao vivo acertou {vitorias} de {n}, abaixo do necessário ({breakeven:.1%}) com confiança"
    return None


def combinar(vitorias_teste: int, n_teste: int, vitorias_vivo: int, n_vivo: int,
             z: float) -> tuple[float | None, float | None]:
    """Taxa e pior caso juntando o placar do teste e o ao vivo."""
    v, n = vitorias_teste + vitorias_vivo, n_teste + n_vivo
    if n <= 0:
        return None, None
    return v / n, wilson(v, n, z)[0]


def aplicar(hipoteses: list, fichas: dict[tuple, Ficha], dias_confirmacao: int, z: float,
            min_ops_abaixo: int) -> dict[str, list]:
    """Usa a ficha para decidir o plano. Muda as hipóteses no lugar.

    - aprovada hoje, mas sem repetir a aprovação nos dias anteriores: fica "confirmando"
      (entra no plano só como exploração, na demo, até a aprovação se repetir);
    - placar ao vivo bem pior que o teste: sai do plano;
    - as demais recebem o placar combinado (teste + ao vivo).
    Devolve as listas de hipóteses que ficaram confirmando e as que saíram, para o registro.
    """
    confirmando, fora = [], []
    for h in hipoteses:
        f = fichas.get(chave(h))
        seguidas = f.aprovacoes_seguidas() if f else 0
        resumo = {
            "aprovacoes_seguidas": seguidas,
            "dias_acima": f.dias_acima() if f else 0,
            "dias_vistos": len(f.historico) if f else 0,
            "vivo": [f.vivo_vitorias, f.vivo_derrotas, f.vivo_empates, round(f.vivo_lucro, 2)] if f else [0, 0, 0, 0.0],
            "confirmando": False,
            "fora": None,
        }
        h.ficha = resumo
        if f is None:
            if h.aprovado and dias_confirmacao > 1:
                h.aprovado = False
                resumo["confirmando"] = True
                h.situacao = _texto_confirmando(dias_confirmacao - 1)
                confirmando.append(h)
            continue
        if f.vivo_n:
            resumo["taxa_combinada"], resumo["pior_caso_combinado"] = combinar(
                h.vitorias, h.n, f.vivo_vitorias, f.vivo_n, z)
        motivo = julgar_ao_vivo(f.vivo_vitorias, f.vivo_n, h.taxa, h.breakeven, z, min_ops_abaixo)
        if motivo:
            resumo["fora"] = motivo
            h.aprovado = False
            h.situacao = f"fora do plano: {motivo}"
            fora.append(h)
            continue
        if h.aprovado and seguidas + 1 < dias_confirmacao:
            h.aprovado = False
            resumo["confirmando"] = True
            h.situacao = _texto_confirmando(dias_confirmacao - 1 - seguidas)
            confirmando.append(h)
    for h in hipoteses:
        h.ficha["aprovada_hoje"] = bool(h.aprovado)
    return {"confirmando": confirmando, "fora": fora}


def _texto_confirmando(faltam: int) -> str:
    return (f"aprovada hoje; precisa se repetir em mais {faltam} pesquisa(s) de outro dia "
            f"(enquanto isso, só exploração na demo)")


def nota_de_ordem(h) -> float:
    """Valor esperado usado para ordenar candidatas: com o placar ao vivo quando existe."""
    f = getattr(h, "ficha", None) or {}
    taxa = f.get("taxa_combinada", h.taxa)
    return valor_esperado(taxa, h.payout) if taxa is not None else float("-inf")


def descrever(ficha: dict | None) -> str:
    """Resumo curto da ficha para o registro e o relatório."""
    if not ficha:
        return ""
    partes = []
    seguidas = ficha.get("aprovacoes_seguidas") or 0
    if ficha.get("fora"):
        partes.append(f"fora do plano: {ficha['fora']}")
    elif ficha.get("confirmando"):
        partes.append("aprovada hoje, aguardando confirmação")
    elif ficha.get("aprovada_hoje") and seguidas:
        partes.append(f"aprovada em {seguidas + 1} pesquisas seguidas")
    elif seguidas:
        partes.append(f"aprovada nas {seguidas} pesquisa(s) anteriores, mas não hoje")
    if ficha.get("dias_vistos"):
        partes.append(f"acima do necessário em {ficha['dias_acima']} de {ficha['dias_vistos']} pesquisas anteriores")
    v, d, e, lucro = ficha.get("vivo") or (0, 0, 0, 0.0)
    if v + d + e:
        partes.append(f"ao vivo {v} de {v + d}" + (f" e {e} empate(s)" if e else "") + f", {lucro:+.2f}")
    return "; ".join(partes)
