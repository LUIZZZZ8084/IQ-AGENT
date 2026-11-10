"""Relatório diário em HTML (um arquivo só, abre em qualquer navegador)."""
from __future__ import annotations

import html
import json
import math
import time
from datetime import datetime
from pathlib import Path

from .banco import Banco
from .cenario import Cenario, ler
from .config import Config
from .estatistica import breakeven, wilson, z_unilateral
from .estrategias import NOMES_CURTOS
from .filtros import SESSOES

MAX_GRAFICO = 12
LARGURA = 760


# --------------------------------------------------------------------------- formatação
def _e(texto) -> str:
    return html.escape(str(texto), quote=True)


def _num(valor: float, casas: int = 2) -> str:
    return f"{valor:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _pct(valor: float | None, casas: int = 1) -> str:
    return "–" if valor is None else _num(valor * 100, casas) + "%"


def _com_sinal(valor: float, casas: int = 2) -> str:
    return ("+" if valor > 0 else "−" if valor < 0 else "") + _num(abs(valor), casas)


def _pp(valor: float) -> str:
    return _com_sinal(valor, 1) + " p.p."


def _quando(ts: float | None, formato: str = "%d/%m %H:%M") -> str:
    return "–" if not ts else datetime.fromtimestamp(ts).strftime(formato)


def _sessao(nome: str) -> str:
    """Nome da sessão com as horas convertidas para o horário do computador."""
    if nome == "todas":
        return "qualquer horário"
    ini, fim = SESSOES[nome]
    desloc = round(datetime.now().astimezone().utcoffset().total_seconds() / 3600)
    return f"{(ini + desloc) % 24:02d}h–{(fim + desloc) % 24:02d}h"


def _config(cfg: dict | None) -> str:
    if not cfg:
        return "–"
    partes = [f"{k}={v}" for k, v in cfg.get("params", {}).items()]
    if cfg.get("vol") and cfg["vol"] != "qualquer":
        partes.append(f"volatilidade {cfg['vol']}")
    return ", ".join(partes) or "padrão"


def _estrategia(nome: str) -> str:
    return NOMES_CURTOS.get(nome, nome)


def _rotulo(h: dict) -> str:
    return f"{h['ativo']} · {h['modo']} · {_estrategia(h['estrategia'])}"


def _passos(maximo: float, alvo: int = 4) -> float:
    """Passo 'redondo' para os eixos."""
    if maximo <= 0:
        return 1.0
    bruto = maximo / alvo
    base = 10 ** math.floor(math.log10(bruto))
    for m in (1, 2, 2.5, 5, 10):
        if bruto <= m * base:
            return m * base
    return 10 * base


def _tabela(cabecalho: list[str], linhas: list[list[str]], numericas: set[int] = frozenset()) -> str:
    """Tabela HTML. As células já chegam escapadas."""
    th = "".join(f'<th class="{"n" if i in numericas else ""}">{_e(c)}</th>' for i, c in enumerate(cabecalho))
    corpo = "".join(
        "<tr>" + "".join(f'<td class="{"n" if i in numericas else ""}">{c}</td>' for i, c in enumerate(l)) + "</tr>"
        for l in linhas)
    return f'<div class="rolagem"><table><thead><tr>{th}</tr></thead><tbody>{corpo}</tbody></table></div>'


def _dados(titulo: str, tabela: str) -> str:
    return f"<details><summary>{_e(titulo)}</summary>{tabela}</details>"


def _vazio(texto: str) -> str:
    return f'<p class="vazio">{_e(texto)}</p>'


def _selo(tipo: str, texto: str) -> str:
    icone = {"bom": "✓", "alerta": "!", "critico": "×"}.get(tipo, "")
    return f'<span class="selo {tipo}"><span class="ico" aria-hidden="true">{icone}</span>{_e(texto)}</span>'


# --------------------------------------------------------------------------- gráficos (SVG)
def _barra_h(x0: float, x1: float, y: float, h: float, classe: str) -> str:
    """Barra horizontal: reta na linha de base (x0) e arredondada na ponta do dado (x1)."""
    largura = abs(x1 - x0)
    if largura < 0.5:
        return ""
    r = min(4.0, largura, h / 2)
    s = 1 if x1 > x0 else -1
    d = (f"M{x0:.1f},{y:.1f} H{x1 - s * r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} "
         f"Q{x1:.1f},{y + h:.1f} {x1 - s * r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return f'<path d="{d}" class="marca {classe}"/>'


def _coluna(x: float, y_topo: float, y_base: float, w: float, classe: str) -> str:
    """Coluna: reta na base e arredondada no topo."""
    altura = y_base - y_topo
    if altura < 0.5:
        return ""
    r = min(4.0, altura, w / 2)
    d = (f"M{x:.1f},{y_base:.1f} V{y_topo + r:.1f} Q{x:.1f},{y_topo:.1f} {x + r:.1f},{y_topo:.1f} "
         f"H{x + w - r:.1f} Q{x + w:.1f},{y_topo:.1f} {x + w:.1f},{y_topo + r:.1f} V{y_base:.1f} Z")
    return f'<path d="{d}" class="marca {classe}"/>'


def grafico_margens(hipoteses: list[dict]) -> str:
    """Margem da taxa de acerto sobre o breakeven, com intervalo de confiança, por hipótese."""
    esq, dire, topo, alt, base = 236, 96, 6, 30, 44
    n = len(hipoteses)
    altura = topo + alt * n + base
    plot = LARGURA - esq - dire
    linhas = [{"m": (h["taxa"] - h["breakeven"]) * 100, "lo": (h["wilson_inf"] - h["breakeven"]) * 100,
               "hi": (h["wilson_sup"] - h["breakeven"]) * 100, "h": h} for h in hipoteses]
    extremo = max(max(abs(l["lo"]), abs(l["hi"])) for l in linhas)
    lim = max(5.0, math.ceil(extremo / 5.0) * 5.0)

    def x(v: float) -> float:
        return esq + plot * (max(-lim, min(lim, v)) + lim) / (2 * lim)

    p = [f'<svg viewBox="0 0 {LARGURA} {altura}" class="graf" role="img" '
         f'aria-label="Margem sobre o breakeven das {n} hipóteses mais fortes">']
    fundo = topo + alt * n
    for t in (-lim, -lim / 2, 0.0, lim / 2, lim):
        p.append(f'<line x1="{x(t):.1f}" y1="{topo}" x2="{x(t):.1f}" y2="{fundo}" class="{"zero" if t == 0 else "grade"}"/>')
        p.append(f'<text x="{x(t):.1f}" y="{fundo + 16}" text-anchor="middle" class="t-eixo">{_com_sinal(t, 0) if t else "0"}</text>')
    p.append(f'<text x="{esq + plot / 2:.1f}" y="{fundo + 36}" text-anchor="middle" class="t-eixo">'
             "pontos percentuais acima ou abaixo do breakeven</text>")
    for i, l in enumerate(linhas):
        h, y0 = l["h"], topo + alt * i
        meio = y0 + alt / 2
        valor = f"{_pp(l['m'])} ({_pct(h['taxa'])} em {_num(h['n'], 0)} operações)"
        detalhe = f"{_rotulo(h)} · intervalo {_pp(l['lo'])} a {_pp(l['hi'])} · {h['situacao']}"
        p.append(f'<g class="alvo" tabindex="0" data-v="{_e(valor)}" data-l="{_e(detalhe)}">')
        p.append(f'<rect x="0" y="{y0}" width="{LARGURA}" height="{alt}" class="toque"/>')
        p.append(f'<text x="{esq - 12}" y="{meio + 4:.1f}" text-anchor="end" class="t-rot'
                 f'{" forte" if h["aprovado"] else ""}">{_e(_rotulo(h))}</text>')
        p.append(_barra_h(x(0), x(l["m"]), meio - 6, 12, "pos" if l["m"] >= 0 else "neg"))
        p.append(f'<path d="M{x(l["lo"]):.1f},{meio - 4:.1f} v8 M{x(l["lo"]):.1f},{meio:.1f} H{x(l["hi"]):.1f} '
                 f'M{x(l["hi"]):.1f},{meio - 4:.1f} v8" class="bigode"/>')
        p.append(f'<text x="{LARGURA - 4}" y="{meio + 4:.1f}" text-anchor="end" class="t-val">{_pp(l["m"])}</text>')
        p.append("</g>")
    p.append("</svg>")
    return "".join(p)


def grafico_distribuicao(margens: list[float]) -> str:
    """Histograma: quantas hipóteses ficaram em cada faixa de margem sobre o breakeven."""
    lim, passo = 14, 2
    faixas = 2 * lim // passo
    contagem = [0] * faixas
    for m in margens:
        contagem[int((min(max(m, -lim), lim - 1e-9) + lim) // passo)] += 1
    esq, dire, topo, base, altura = 44, 12, 24, 44, 240
    plot_w, plot_h = LARGURA - esq - dire, altura - topo - base
    passo_y = _passos(max(contagem))
    y_max = max(passo_y, math.ceil(max(contagem) / passo_y) * passo_y)
    banda = plot_w / faixas
    largura = min(24.0, banda - 2)
    fundo = topo + plot_h

    def y(v: float) -> float:
        return topo + plot_h * (1 - v / y_max)

    p = [f'<svg viewBox="0 0 {LARGURA} {altura}" class="graf" role="img" '
         'aria-label="Distribuição das hipóteses pela margem sobre o breakeven">']
    marca = 0.0
    while marca <= y_max + 1e-9:
        p.append(f'<line x1="{esq}" y1="{y(marca):.1f}" x2="{LARGURA - dire}" y2="{y(marca):.1f}" '
                 f'class="{"base" if marca == 0 else "grade"}"/>')
        p.append(f'<text x="{esq - 8}" y="{y(marca) + 4:.1f}" text-anchor="end" class="t-eixo">{marca:g}</text>')
        marca += passo_y
    for v in range(-lim + 2, lim, 4):
        p.append(f'<text x="{esq + plot_w * (v + lim) / (2 * lim):.1f}" y="{fundo + 16}" text-anchor="middle" '
                 f'class="t-eixo">{_com_sinal(v, 0) if v else "0"}</text>')
    centro = esq + plot_w / 2
    p.append(f'<line x1="{centro:.1f}" y1="{topo - 6}" x2="{centro:.1f}" y2="{fundo}" class="zero"/>')
    p.append(f'<text x="{centro + 6:.1f}" y="{topo - 10}" class="t-eixo">breakeven</text>')
    p.append(f'<text x="{esq + plot_w / 2:.1f}" y="{fundo + 36}" text-anchor="middle" class="t-eixo">'
             "taxa de acerto fora da amostra menos o breakeven (pontos percentuais)</text>")
    for i, c in enumerate(contagem):
        a, b = -lim + i * passo, -lim + (i + 1) * passo
        faixa = (f"{_com_sinal(a, 0)} ou menos" if i == 0 else f"{_com_sinal(a, 0)} ou mais" if i == faixas - 1
                 else f"de {_com_sinal(a, 0) if a else '0'} a {_com_sinal(b, 0) if b else '0'}") + " p.p."
        x0 = esq + banda * i
        p.append(f'<g class="alvo" tabindex="0" data-v="{c} hipótese{"s" if c != 1 else ""}" data-l="{_e(faixa)}">')
        p.append(f'<rect x="{x0:.1f}" y="{topo}" width="{banda:.1f}" height="{plot_h}" class="toque"/>')
        p.append(_coluna(x0 + (banda - largura) / 2, y(c), fundo, largura, "pos"))
        p.append("</g>")
    p.append("</svg>")
    return "".join(p)


def grafico_curva(operacoes: list[dict], ident: str) -> str:
    """Resultado acumulado, operação a operação."""
    acumulado, pontos, total = [0.0], [], 0.0
    for i, o in enumerate(operacoes, 1):
        total += float(o["lucro"] or 0.0)
        acumulado.append(total)
        nome = {"win": "ganhou", "loss": "perdeu", "equal": "empate"}.get(o["status"], o["status"])
        pontos.append({"v": f"Acumulado {_com_sinal(total)}",
                       "l": f"nº {i} · {_quando(o['ts_sinal'])} · {o['ativo']} {o['direcao'].upper()} · "
                            f"{nome} {_com_sinal(float(o['lucro'] or 0.0))}"})
    n = len(operacoes)
    esq, dire, topo, base, altura = 60, 84, 14, 36, 250
    plot_w, plot_h = LARGURA - esq - dire, altura - topo - base
    minimo, maximo = min(acumulado), max(acumulado)
    passo = _passos(max(maximo - minimo, 1.0))
    y_min, y_max = math.floor(minimo / passo) * passo, math.ceil(maximo / passo) * passo
    if y_min == y_max:
        y_max = y_min + passo

    def x(i: int) -> float:
        return esq + plot_w * i / max(n, 1)

    def y(v: float) -> float:
        return topo + plot_h * (1 - (v - y_min) / (y_max - y_min))

    for i, ponto in enumerate(pontos, 1):
        ponto["x"], ponto["y"] = round(x(i), 1), round(y(acumulado[i]), 1)
    p = [f'<svg viewBox="0 0 {LARGURA} {altura}" class="graf curva" role="img" data-pontos="{ident}" '
         f'aria-label="Resultado acumulado em {n} operações: {_com_sinal(total)}">']
    marca = y_min
    while marca <= y_max + 1e-9:
        p.append(f'<line x1="{esq}" y1="{y(marca):.1f}" x2="{LARGURA - dire}" y2="{y(marca):.1f}" '
                 f'class="{"zero" if abs(marca) < 1e-9 else "grade"}"/>')
        p.append(f'<text x="{esq - 8}" y="{y(marca) + 4:.1f}" text-anchor="end" class="t-eixo">{_com_sinal(marca, 0) if marca else "0"}</text>')
        marca += passo
    for i in sorted({0, n // 2, n}):
        p.append(f'<text x="{x(i):.1f}" y="{topo + plot_h + 16}" text-anchor="middle" class="t-eixo">{i}</text>')
    p.append(f'<text x="{esq + plot_w / 2:.1f}" y="{topo + plot_h + 32}" text-anchor="middle" class="t-eixo">'
             "número de operações</text>")
    traco = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(acumulado))
    p.append(f'<path d="{traco} L{x(n):.1f},{y(0):.1f} L{x(0):.1f},{y(0):.1f} Z" class="lavado"/>')
    p.append(f'<path d="{traco}" class="linha"/>')
    p.append(f'<circle cx="{x(n):.1f}" cy="{y(total):.1f}" r="4.5" class="ponto"/>')
    p.append(f'<text x="{x(n) + 10:.1f}" y="{y(total) + 4:.1f}" class="t-val">{_com_sinal(total)}</text>')
    p.append(f'<line x1="0" y1="{topo}" x2="0" y2="{topo + plot_h}" class="mira" visibility="hidden"/>')
    p.append('<circle r="4.5" class="ponto foco" visibility="hidden"/>')
    p.append(f'<rect x="{esq}" y="{topo}" width="{plot_w}" height="{plot_h}" class="toque"/>')
    p.append("</svg>")
    # dentro de <script>, nenhum texto vindo de fora pode aparecer com <, > ou & crus
    dados = json.dumps(pontos, ensure_ascii=False)
    for caractere in "<>&":
        dados = dados.replace(caractere, f"\\u{ord(caractere):04x}")
    return "".join(p) + f'<script type="application/json" id="{ident}">{dados}</script>'


# --------------------------------------------------------------------------- blocos
def _placar(operacoes: list[dict]) -> dict:
    validas = [o for o in operacoes if o["status"] in ("win", "loss", "equal")]
    decididas = [o for o in validas if o["status"] != "equal"]
    vitorias = sum(o["status"] == "win" for o in decididas)
    return {"ops": validas, "n": len(decididas), "vitorias": vitorias,
            "taxa": vitorias / len(decididas) if decididas else None,
            "lucro": sum(float(o["lucro"] or 0.0) for o in validas)}


def _tile(rotulo: str, valor: str, nota: str = "", tom: str = "") -> str:
    return (f'<div class="tile"><div class="rot">{_e(rotulo)}</div><div class="val {tom}">{valor}</div>'
            f'<div class="nota">{_e(nota)}</div></div>')


def _bloco_veredito(pesquisa: dict | None) -> str:
    if pesquisa is None:
        return ('<section class="cartao veredito"><div class="heroi">–</div><div><h2>Ainda sem pesquisa</h2>'
                "<p>Abra o painel e clique em “Atualizar mercado”, ou rode o comando <code>diario</code>. "
                "O bot baixa o histórico real da corretora e testa as estratégias.</p></div></section>")
    r = pesquisa["resumo"]
    aprovadas = r["aprovadas"]
    if aprovadas:
        titulo = f"{aprovadas} estratégia{'s' if aprovadas > 1 else ''} com vantagem estatística hoje"
        texto = ("Elas passaram no teste fora da amostra e na correção para múltiplos testes. Veja o plano "
                 "abaixo. Vantagem no histórico não garante resultado futuro.")
    else:
        titulo = "Nenhuma estratégia com vantagem estatística hoje"
        texto = ("Nenhuma combinação de ativo e estratégia superou o breakeven com confiança nos dados reais "
                 "recentes. O plano do dia é não operar. Na demo, o modo exploração segue coletando dados.")
    return (f'<section class="cartao veredito"><div class="heroi">{aprovadas}</div>'
            f"<div><h2>{_e(titulo)}</h2><p>{_e(texto)}</p></div></section>")


def _bloco_kpis(pesquisa: dict | None, demo: dict, real: dict) -> str:
    tiles = []
    if pesquisa:
        r = pesquisa["resumo"]
        tiles.append(_tile("Hipóteses testadas", _num(r["hipoteses"], 0), f"{r['ativos']} ativos reais"))
        tiles.append(_tile("Com dados suficientes", _num(r["testaveis"], 0), "chegaram ao teste fora da amostra"))
        tiles.append(_tile("Com p < 5%", _num(r["p_menor_5"], 0), f"só por acaso, esperado ≈ {_num(r['esperado_acaso'], 1)}"))
        if r.get("taxa_media") is not None:
            tiles.append(_tile("Taxa média fora da amostra", _pct(r["taxa_media"]),
                               f"breakeven médio {_pct(r['breakeven_medio'])}"))
    if demo["ops"]:
        tom = "sobe" if demo["lucro"] > 0 else "desce" if demo["lucro"] < 0 else ""
        seta = "▲ " if demo["lucro"] > 0 else "▼ " if demo["lucro"] < 0 else ""
        tiles.append(_tile("Demo: taxa de acerto", _pct(demo["taxa"]), f"{demo['vitorias']} de {demo['n']} operações"))
        tiles.append(_tile("Demo: resultado", seta + _com_sinal(demo["lucro"]), "soma de todas as operações", tom))
    if real["ops"]:
        tom = "sobe" if real["lucro"] > 0 else "desce" if real["lucro"] < 0 else ""
        seta = "▲ " if real["lucro"] > 0 else "▼ " if real["lucro"] < 0 else ""
        tiles.append(_tile("Conta real: resultado", seta + _com_sinal(real["lucro"]),
                           f"{real['vitorias']} acertos em {real['n']} operações", tom))
    return f'<section class="kpis">{"".join(tiles)}</section>' if tiles else ""


def _bloco_plano(pesquisa: dict, cfg: Config) -> str:
    entradas = pesquisa["plano"].get("entradas", [])
    if not entradas:
        return ('<section class="cartao" id="plano"><h2>Plano do dia</h2>'
                + _vazio("Nenhuma entrada: nem aprovadas, nem candidatas de exploração.") + "</section>")
    linhas = []
    for e in entradas:
        aprovada = e["tipo"] == "aprovado"
        como = (f"Kelly fracionado, até {_pct(cfg.risco.risco_por_operacao)} do saldo" if aprovada
                else f"valor fixo {_num(cfg.operacao.valor_exploracao)}, só na demo")
        linhas.append([
            _selo("bom", "aprovada") if aprovada else _selo("alerta", "exploração"),
            f"<strong>{_e(e['ativo'])}</strong>", _e(e["modo"]), _e(_estrategia(e["estrategia"])),
            _e(_config(e)), _e(_sessao(e["sessao"])), _pct(e["taxa"]), _pct(e["p_inf"]), _pct(e["breakeven"]),
            _pct(e["payout"], 0), _num(e["n_teste"], 0), _e(como),
        ])
    tabela = _tabela(["Situação", "Ativo", "Modo", "Estratégia", "Configuração", "Horário", "Acerto",
                      "Pior caso", "Breakeven", "Payout", "Operações", "Quanto investir"], linhas, {6, 7, 8, 9, 10})
    nota = ("“Acerto” é a taxa nas janelas de teste (dados que a estratégia não viu ao ser escolhida). "
            "“Pior caso” é o limite inferior do intervalo de confiança: o tamanho da operação é calculado com ele. "
            "Entradas de exploração NÃO têm vantagem comprovada e nunca são operadas na conta real.")
    return f'<section class="cartao" id="plano"><h2>Plano do dia</h2>{tabela}<p class="nota">{_e(nota)}</p></section>'


def _bloco_margens(pesquisa: dict) -> str:
    testaveis = sorted((h for h in pesquisa["hipoteses"] if h["testavel"]), key=lambda h: h["p_valor"])
    if not testaveis:
        return ('<section class="cartao" id="hipoteses"><h2>Hipóteses mais fortes</h2>'
                + _vazio("Nenhuma hipótese teve operações suficientes fora da amostra.") + "</section>")
    topo = testaveis[:MAX_GRAFICO]
    legenda = ('<div class="legenda"><span><i class="q pos"></i>acima do breakeven</span>'
               '<span><i class="q neg"></i>abaixo do breakeven</span>'
               '<span><i class="tr"></i>intervalo de confiança</span></div>')
    linhas = [[_e(_rotulo(h)), _num(h["n"], 0), _pct(h["taxa"]), _pct(h["wilson_inf"]), _pct(h["wilson_sup"]),
               _pct(h["breakeven"]), _e(f"{h['p_valor']:.3g}"), _e(h["situacao"])] for h in topo]
    tabela = _tabela(["Hipótese", "Operações", "Acerto", "Mínimo", "Máximo", "Breakeven", "p-valor", "Situação"],
                     linhas, {1, 2, 3, 4, 5, 6})
    nota = ("Uma hipótese só é aprovada quando o intervalo inteiro fica à direita do zero e ela ainda "
            "sobrevive à correção para o número de testes feitos.")
    return (f'<section class="cartao" id="hipoteses"><h2>Hipóteses mais fortes</h2><p class="sub">Taxa de acerto fora da amostra '
            f"em relação ao breakeven de cada ativo</p>{legenda}{grafico_margens(topo)}"
            f'<p class="nota">{_e(nota)}</p>{_dados("Ver estes dados em tabela", tabela)}</section>')


def _bloco_distribuicao(pesquisa: dict) -> str:
    testaveis = [h for h in pesquisa["hipoteses"] if h["testavel"]]
    if len(testaveis) < 5:
        return ""
    margens = [(h["taxa"] - h["breakeven"]) * 100 for h in testaveis]
    acima = sum(m > 0 for m in margens)
    nota = (f"{acima} de {len(margens)} hipóteses ficaram acima do breakeven nas janelas de teste. Sem vantagem "
            "real, o esperado é a maioria ficar abaixo, porque o payout menor que 100% joga contra.")
    return ('<section class="cartao" id="panorama"><h2>Panorama de todas as hipóteses</h2><p class="sub">Quantas hipóteses '
            f"caíram em cada faixa de margem</p>{grafico_distribuicao(margens)}"
            f'<p class="nota">{_e(nota)}</p></section>')


def _bloco_cenarios(cenarios: list[Cenario], pesquisa: dict | None) -> str:
    if not cenarios:
        return ""
    no_plano = {e["ativo"] for e in (pesquisa or {}).get("plano", {}).get("entradas", [])}
    linhas = []
    for c in sorted(cenarios, key=lambda c: (c.ativo not in no_plano, c.ativo)):
        linhas.append([
            f"<strong>{_e(c.ativo)}</strong>" + (" " + _selo("", "no plano") if c.ativo in no_plano else ""),
            _e({"alta": "↗ alta", "baixa": "↘ baixa", "lateral": "→ lateral"}[c.tendencia]),
            _e(f"{c.volatilidade} ({_num(c.razao_vol, 2)}×)"), _num(c.rsi, 0),
            _com_sinal(c.variacao_1h * 100, 2) + "%",
            _e(c.leitura), _e(_quando(c.ts)),
        ])
    tabela = _tabela(["Ativo", "Tendência (5 min)", "Volatilidade", "RSI", "Última hora", "Leitura",
                      "Dados até"], linhas, {3, 4})
    return ('<section class="cartao" id="cenario"><h2>Cenário atual dos ativos</h2><p class="sub">Leitura das velas reais mais '
            f"recentes de cada ativo acompanhado</p>{tabela}</section>")


def _bloco_conta(titulo: str, placar: dict, pesquisa: dict | None, cfg: Config, ident: str) -> str:
    ops = placar["ops"]
    ancora = ident.removeprefix("pontos-")  # id da seção; `ident` é o id dos dados do gráfico
    if not ops:
        return (f'<section class="cartao" id="{ancora}"><h2>{_e(titulo)}</h2>'
                + _vazio("Ainda não há operações registradas nesta conta.") + "</section>")
    esperado = {(h["ativo"], h["modo"], h["estrategia"]): h for h in (pesquisa or {}).get("hipoteses", [])}
    z = z_unilateral(cfg.pesquisa.confianca)
    grupos: dict[tuple, list[dict]] = {}
    for o in ops:
        grupos.setdefault((o["ativo"], o["modo"], o["estrategia"]), []).append(o)
    comparacao = []
    for chave, lista in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        decididas = [o for o in lista if o["status"] != "equal"]
        n, vit = len(decididas), sum(o["status"] == "win" for o in decididas)
        h = esperado.get(chave)
        alvo = h["taxa"] if h and h.get("taxa") is not None else None
        equilibrio = sum(breakeven(float(o["payout"])) for o in decididas) / n if n else None
        if n < 20:
            leitura = "poucas operações para concluir"
        elif equilibrio is not None and wilson(vit, n, z)[1] < equilibrio:
            leitura = "abaixo do breakeven com confiança"
        elif alvo is not None and wilson(vit, n, z)[1] < alvo:
            leitura = "abaixo do que o backtest indicava"
        else:
            leitura = "compatível com o backtest"
        comparacao.append([
            _e(f"{chave[0]} · {chave[1]} · {_estrategia(chave[2])}"), _num(n, 0), _pct(vit / n if n else None),
            _pct(alvo), _pct(equilibrio), _com_sinal(sum(float(o["lucro"] or 0.0) for o in lista)), _e(leitura),
        ])
    ultimas = [[
        _e(_quando(o["ts_sinal"])), f"<strong>{_e(o['ativo'])}</strong>", _e(o["direcao"].upper()),
        _e(_estrategia(o["estrategia"])), _e("aprovada" if o["tipo"] == "aprovado" else "exploração"),
        _num(float(o["valor"])), _e({"win": "ganhou", "loss": "perdeu", "equal": "empate"}.get(o["status"], o["status"])),
        _com_sinal(float(o["lucro"] or 0.0)),
    ] for o in reversed(ops[-25:])]
    curva = grafico_curva(ops, ident)
    dados_curva = _tabela(["Nº", "Quando", "Ativo", "Resultado", "Acumulado"], [
        [str(i), _e(_quando(o["ts_sinal"])), _e(o["ativo"]), _com_sinal(float(o["lucro"] or 0.0)),
         _com_sinal(sum(float(x["lucro"] or 0.0) for x in ops[:i]))] for i, o in enumerate(ops, 1)][-200:], {0, 3, 4})
    return (
        f'<section class="cartao" id="{ancora}"><h2>{_e(titulo)}</h2><p class="sub">Resultado acumulado, '
        "operação a operação</p>"
        f'{curva}{_dados("Ver estes dados em tabela", dados_curva)}'
        "<h3>Ao vivo comparado ao backtest</h3>"
        + _tabela(["Estratégia", "Operações", "Acerto ao vivo", "Acerto no backtest", "Breakeven", "Resultado",
                   "Leitura"], comparacao, {1, 2, 3, 4, 5})
        + "<h3>Últimas operações</h3>"
        + _tabela(["Quando", "Ativo", "Direção", "Estratégia", "Tipo", "Valor", "Situação", "Resultado"],
                  ultimas, {5, 7})
        + "</section>")


def _bloco_todas(pesquisa: dict) -> str:
    ordenadas = sorted(pesquisa["hipoteses"], key=lambda h: (not h["testavel"], h["p_valor"]))[:80]
    linhas = [[_e(_rotulo(h)), _num(h["n"], 0), _pct(h["taxa"]), _pct(h["breakeven"]),
               _e(f"{h['p_valor']:.3g}") if h["testavel"] else "–",
               _e(f"{h['q_valor']:.3g}") if h.get("q_valor") is not None else "–",
               _e(_config(h.get("config"))), _e(h["situacao"])] for h in ordenadas]
    tabela = _tabela(["Hipótese", "Operações", "Acerto", "Breakeven", "p-valor", "q-valor", "Configuração de hoje",
                      "Situação"], linhas, {1, 2, 3, 4, 5})
    return ('<section class="cartao" id="todas"><h2>Todas as hipóteses</h2>'
            + _dados(f"Mostrar as {len(linhas)} primeiras, da mais forte para a mais fraca", tabela) + "</section>")


# --------------------------------------------------------------------------- página
_CSS = """
:root{color-scheme:light;--plano:#f9f9f7;--sup:#fcfcfb;--tinta:#0b0b0b;--tinta2:#52514e;--mudo:#898781;
--grade:#e1e0d9;--eixo:#c3c2b7;--borda:rgba(11,11,11,.10);--pos:#2a78d6;--neg:#e34948;--bom:#0ca30c;
--alerta:#fab219;--critico:#d03b3b;--sobe:#006300;--desce:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--plano:#0d0d0d;
--sup:#1a1a19;--tinta:#f0efec;--tinta2:#c3c2b7;--mudo:#898781;--grade:#2c2c2a;--eixo:#383835;
--borda:rgba(255,255,255,.10);--pos:#3987e5;--neg:#e66767;--sobe:#0ca30c;--desce:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--plano:#0d0d0d;--sup:#1a1a19;--tinta:#f0efec;--tinta2:#c3c2b7;
--mudo:#898781;--grade:#2c2c2a;--eixo:#383835;--borda:rgba(255,255,255,.10);--pos:#3987e5;--neg:#e66767;
--sobe:#0ca30c;--desce:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--plano);color:var(--tinta);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0 0 4px}h3{font-size:14px;margin:26px 0 8px}
.sub,.nota{color:var(--tinta2);font-size:13px;margin:2px 0 12px}.nota{margin:12px 0 0}
.cartao{background:var(--sup);border:1px solid var(--borda);border-radius:10px;padding:20px;margin-top:16px}
.veredito{display:flex;gap:22px;align-items:center}.veredito p{margin:4px 0 0;color:var(--tinta2);max-width:68ch}
.heroi{font-size:60px;font-weight:650;line-height:1;min-width:72px;text-align:center}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--sup);border:1px solid var(--borda);border-radius:10px;padding:14px 16px}
.tile .rot{font-size:12.5px;color:var(--tinta2)}.tile .val{font-size:25px;font-weight:600;margin-top:2px}
.tile .nota{font-size:12px;margin:2px 0 0}.val.sobe{color:var(--sobe)}.val.desce{color:var(--desce)}
.rolagem{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;color:var(--tinta2);border-bottom:1px solid var(--eixo);padding:6px 10px 6px 0;
white-space:nowrap}td{border-bottom:1px solid var(--grade);padding:7px 10px 7px 0;vertical-align:top}
th.n,td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.selo{display:inline-flex;align-items:center;gap:5px;font-size:12px;border:1px solid var(--borda);border-radius:99px;
padding:1px 8px;white-space:nowrap;color:var(--tinta)}
.selo .ico{display:inline-grid;place-items:center;width:14px;height:14px;border-radius:50%;font-size:10px;
font-weight:700;color:#fff}.selo.bom .ico{background:var(--bom)}.selo.alerta .ico{background:var(--alerta);color:#0b0b0b}
.selo.critico .ico{background:var(--critico)}.selo:not(.bom):not(.alerta):not(.critico) .ico{display:none}
.legenda{display:flex;flex-wrap:wrap;gap:16px;font-size:12.5px;color:var(--tinta2);margin:0 0 8px}
.legenda span{display:inline-flex;align-items:center;gap:6px}.legenda .q{width:12px;height:12px;border-radius:3px}
.legenda .q.pos{background:var(--pos)}.legenda .q.neg{background:var(--neg)}
.legenda .tr{width:18px;height:0;border-top:1.5px solid var(--tinta2)}
.graf{display:block;width:100%;height:auto;max-width:760px;font-family:inherit}
.graf .grade{stroke:var(--grade);stroke-width:1}.graf .base,.graf .zero{stroke:var(--eixo);stroke-width:1}
.graf .t-eixo{font-size:11px;fill:var(--mudo);font-variant-numeric:tabular-nums}
.graf .t-rot{font-size:12px;fill:var(--tinta2)}.graf .t-rot.forte{fill:var(--tinta);font-weight:600}
.graf .t-val{font-size:12px;fill:var(--tinta);font-variant-numeric:tabular-nums}
.graf .marca.pos{fill:var(--pos)}.graf .marca.neg{fill:var(--neg)}
.graf .bigode{stroke:var(--tinta2);stroke-width:1.5;fill:none}
.graf .toque{fill:transparent}.graf .alvo{outline:none}
.graf .alvo:hover .marca,.graf .alvo:focus .marca{filter:brightness(1.12)}
.graf .alvo:hover .toque,.graf .alvo:focus .toque{fill:var(--borda)}
.graf .linha{fill:none;stroke:var(--pos);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.graf .lavado{fill:var(--pos);opacity:.10}.graf .ponto{fill:var(--pos);stroke:var(--sup);stroke-width:2}
.graf .mira{stroke:var(--eixo);stroke-width:1}
details{margin-top:12px}summary{cursor:pointer;color:var(--tinta2);font-size:13px}details .rolagem{margin-top:8px}
.vazio{color:var(--tinta2);margin:8px 0 0}
code{background:var(--grade);border-radius:4px;padding:0 4px;font-size:13px}
#dica{position:fixed;z-index:9;pointer-events:none;background:var(--tinta);color:var(--sup);border-radius:6px;
padding:6px 9px;font-size:12.5px;max-width:320px;box-shadow:0 2px 10px rgba(0,0,0,.25)}
#dica strong{display:block;font-size:13.5px}#dica span{opacity:.82}
footer{color:var(--tinta2);font-size:12.5px;margin-top:22px}
@media (max-width:640px){.veredito{flex-direction:column;align-items:flex-start}}
"""

_JS = """
(function(){
  var dica=document.getElementById('dica'),forte=dica.querySelector('strong'),leve=dica.querySelector('span');
  function mostrar(v,l,x,y){forte.textContent=v;leve.textContent=l;dica.hidden=false;
    var w=dica.offsetWidth,h=dica.offsetHeight;
    dica.style.left=Math.max(8,Math.min(x+14,window.innerWidth-w-8))+'px';
    dica.style.top=Math.max(8,y-h-12<8?y+18:y-h-12)+'px';}
  function esconder(){dica.hidden=true;}
  document.querySelectorAll('.alvo').forEach(function(g){
    g.addEventListener('pointermove',function(e){mostrar(g.dataset.v,g.dataset.l,e.clientX,e.clientY);});
    g.addEventListener('pointerleave',esconder);
    g.addEventListener('focus',function(){var r=g.getBoundingClientRect();mostrar(g.dataset.v,g.dataset.l,r.left+r.width/2,r.top);});
    g.addEventListener('blur',esconder);
  });
  document.querySelectorAll('svg.curva').forEach(function(svg){
    var pts=JSON.parse(document.getElementById(svg.dataset.pontos).textContent);
    if(!pts.length)return;
    var mira=svg.querySelector('.mira'),foco=svg.querySelector('.foco'),area=svg.querySelector('.toque');
    area.addEventListener('pointermove',function(e){
      var r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*760/r.width,melhor=pts[0];
      for(var i=1;i<pts.length;i++){if(Math.abs(pts[i].x-vx)<Math.abs(melhor.x-vx))melhor=pts[i];}
      mira.setAttribute('x1',melhor.x);mira.setAttribute('x2',melhor.x);mira.setAttribute('visibility','visible');
      foco.setAttribute('cx',melhor.x);foco.setAttribute('cy',melhor.y);foco.setAttribute('visibility','visible');
      mostrar(melhor.v,melhor.l,e.clientX,e.clientY);});
    area.addEventListener('pointerleave',function(){mira.setAttribute('visibility','hidden');
      foco.setAttribute('visibility','hidden');esconder();});
  });
})();
"""


def montar_html(pesquisa: dict | None, operacoes: list[dict], cenarios: list[Cenario], cfg: Config,
                agora: float) -> str:
    demo = _placar([o for o in operacoes if o["conta"] == "PRACTICE"])
    real = _placar([o for o in operacoes if o["conta"] == "REAL"])
    blocos = [_bloco_veredito(pesquisa), _bloco_kpis(pesquisa, demo, real)]
    if pesquisa:
        blocos += [_bloco_plano(pesquisa, cfg), _bloco_margens(pesquisa)]
    blocos.append(_bloco_cenarios(cenarios, pesquisa))
    if pesquisa:
        blocos.append(_bloco_distribuicao(pesquisa))
    blocos.append(_bloco_conta("Conta demo", demo, pesquisa, cfg, "pontos-demo"))
    if real["ops"]:
        blocos.append(_bloco_conta("Conta real", real, pesquisa, cfg, "pontos-real"))
    if pesquisa:
        blocos.append(_bloco_todas(pesquisa))
    if pesquisa:
        r = pesquisa["resumo"]
        origem = (f"Pesquisa de {_quando(pesquisa['ts'])} sobre {_num(r['velas'], 0)} velas reais de 1 minuto "
                  f"({_quando(r['periodo'][0], '%d/%m')} a {_quando(r['periodo'][1], '%d/%m')}), baixadas da corretora.")
    else:
        origem = "Nenhuma pesquisa feita ainda."
    rodape = ("Opções binárias têm expectativa negativa para quem não tem vantagem real: com payout de 85% é preciso "
              "acertar mais de 54% só para empatar. Resultado em conta demo não garante resultado em conta real. "
              "Este relatório é gerado por regras estatísticas e não é recomendação de investimento.")
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>IQ-AGENT · relatório</title>'
        f"<style>{_CSS}</style></head><body><main><header><h1>IQ-AGENT · relatório do dia</h1>"
        f'<p class="sub">Gerado em {_e(_quando(agora, "%d/%m/%Y %H:%M"))}. {_e(origem)}</p></header>'
        + "".join(blocos)
        + f"<footer>{_e(rodape)}</footer></main>"
        '<div id="dica" role="status" hidden><strong></strong><span></span></div>'
        f"<script>{_JS}</script></body></html>"
    )


def gerar(banco: Banco, cfg: Config, agora: float | None = None, caminho: Path | None = None) -> Path:
    """Monta o relatório com a última pesquisa, o diário e o cenário atual dos ativos."""
    agora = agora if agora is not None else time.time()
    pesquisa = banco.ultima_pesquisa()
    cenarios = []
    for ativo in banco.ativos_com_velas(60):
        _, ultimo, _ = banco.intervalo_velas(ativo, 60)
        if ultimo is None:
            continue
        cenario = ler(ativo, banco.carregar_velas(ativo, 60, ultimo - 12 * 3600))
        if cenario is not None:
            cenarios.append(cenario)
    pagina = montar_html(pesquisa, banco.operacoes(), cenarios, cfg, agora)
    caminho = caminho or cfg.dir_relatorios / "relatorio.html"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(pagina, encoding="utf-8")
    return caminho
