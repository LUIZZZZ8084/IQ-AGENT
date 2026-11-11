"""Pesquisa diária: testa estratégias em cada ativo e monta o plano do dia.

Método (walk-forward):
  1. O histórico é dividido em um trecho inicial de treino e N janelas de teste.
  2. Antes de cada janela, escolhe-se a melhor configuração de cada estratégia usando
     SÓ os dados anteriores a ela. Se nenhuma mostra vantagem no treino, a estratégia
     fica de fora daquela janela.
  3. A configuração escolhida é "operada" na janela de teste. Só esses resultados fora
     da amostra contam para julgar a hipótese (ativo, modo, estratégia).
  4. Como centenas de hipóteses são testadas, algumas pareceriam boas por sorte. O
     controle de falsas descobertas (Benjamini-Hochberg) corrige isso.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .backtest import aplicar, desfechos
from .banco import Banco
from .config import Config, Pesquisa, interpretar_modo
from .estatistica import (benjamini_hochberg, breakeven, p_valor_binomial, valor_esperado, wilson,
                          wilson_inferior_vetor, z_unilateral)
from .estrategias import ESTRATEGIAS
from .filtros import SESSOES, VOLS, mascara_sessao, mascara_vol, razao_vol
from .velas import Velas

MIN_VELAS = 500


@dataclass
class Hipotese:
    ativo: str
    modo: str
    estrategia: str
    payout: float
    breakeven: float
    folds: int = 0
    folds_ativos: int = 0
    vitorias: int = 0
    derrotas: int = 0
    empates: int = 0
    taxa: float | None = None
    wilson_inf: float | None = None
    wilson_sup: float | None = None
    p_valor: float = 1.0
    q_valor: float | None = None
    ev: float | None = None
    ev_inf: float | None = None
    config: dict | None = None
    treino_n: int = 0
    treino_taxa: float | None = None
    testavel: bool = False
    aprovado: bool = False
    situacao: str = ""
    detalhe: dict | None = None  # dados extras (ex.: ativos de um grupo do modelo de aprendizado de máquina)

    @property
    def n(self) -> int:
        return self.vitorias + self.derrotas


def preencher_estatisticas(h: Hipotese, z: float, pq: Pesquisa) -> None:
    """Calcula taxa, intervalo, p-valor e valor esperado a partir do placar fora da amostra."""
    if h.n > 0:
        h.taxa = h.vitorias / h.n
        h.wilson_inf, h.wilson_sup = wilson(h.vitorias, h.n, z)
        h.p_valor = p_valor_binomial(h.vitorias, h.n, h.breakeven)
        h.ev = valor_esperado(h.taxa, h.payout)
        h.ev_inf = valor_esperado(h.wilson_inf, h.payout)
    h.testavel = h.n >= pq.min_trades_teste


def limites_folds(ts_ini: int, ts_fim: int, n_folds: int, dias_teste: float) -> np.ndarray | None:
    """Instantes que separam treino inicial e janelas de teste. None se o histórico é curto."""
    passo = int(dias_teste * 86400)
    n = n_folds
    while n >= 2 and (ts_fim - n * passo) - ts_ini < passo:
        n -= 1
    if n < 2:
        return None
    return ts_fim - passo * np.arange(n, -1, -1)


def avaliar_modo(ativo: str, v: Velas, tf_min: int, exp_min: int, payout: float, pq: Pesquisa,
                 estrategias: list[str]) -> list[Hipotese]:
    """Avalia todas as estratégias em um ativo, num modo (timeframe + expiração)."""
    modo = f"M{tf_min}>{exp_min}"
    k = exp_min // tf_min
    if len(v) < MIN_VELAS:
        return []
    limites = limites_folds(int(v.ts[0]), int(v.ts[-1]) + v.tf, pq.n_folds, pq.dias_teste)
    if limites is None:
        return []
    n_folds = len(limites) - 1
    p0 = breakeven(payout)
    z = z_unilateral(pq.confianca)

    res_call, res_put, valido = desfechos(v, k, pq.slippage_frac)
    # Cada operação pertence ao trecho em que ELA TERMINA: assim o treino nunca enxerga
    # um resultado que só seria conhecido dentro da janela de teste.
    trecho = np.searchsorted(limites, v.ts + (k + 1) * v.tf, side="left")
    razao = razao_vol(v)
    filtros = [(s, x, mascara_sessao(v, s) & mascara_vol(razao, x)) for s in SESSOES for x in VOLS]

    hipoteses = []
    for nome in estrategias:
        est = ESTRATEGIAS[nome]
        configs: list[dict] = []
        vit, der, emp = [], [], []
        for params in est.parametros():
            _, ganhou, perdeu, empatou = aplicar(est.func(v, **params), res_call, res_put, valido, k)
            for sessao, vol, masc in filtros:
                configs.append({"params": params, "sessao": sessao, "vol": vol})
                vit.append(np.bincount(trecho[ganhou & masc], minlength=n_folds + 2)[: n_folds + 1])
                der.append(np.bincount(trecho[perdeu & masc], minlength=n_folds + 2)[: n_folds + 1])
                emp.append(np.bincount(trecho[empatou & masc], minlength=n_folds + 2)[: n_folds + 1])
        hipoteses.append(_julgar(ativo, modo, nome, payout, p0, z, pq, configs,
                                 np.array(vit), np.array(der), np.array(emp), n_folds))
    return hipoteses


def _julgar(ativo: str, modo: str, estrategia: str, payout: float, p0: float, z: float, pq: Pesquisa,
            configs: list[dict], vit: np.ndarray, der: np.ndarray, emp: np.ndarray, n_folds: int) -> Hipotese:
    h = Hipotese(ativo=ativo, modo=modo, estrategia=estrategia, payout=payout, breakeven=p0, folds=n_folds)
    ac_vit, ac_der = np.cumsum(vit, axis=1), np.cumsum(der, axis=1)

    def melhor(ate: int) -> tuple[int, float]:
        """Melhor configuração usando os trechos 0..ate (inclusive) como treino.

        A nota é o lucro total pessimista do treino: nº de operações x valor esperado
        calculado com o limite inferior da taxa de acerto. Escolher só pela taxa
        favoreceria filtros estreitos que acertaram muito por sorte em poucas operações.
        """
        w = ac_vit[:, ate]
        n = w + ac_der[:, ate]
        ev_pessimista = wilson_inferior_vetor(w, n, z) * (1.0 + payout) - 1.0
        nota = np.where(n >= pq.min_trades_treino, n * ev_pessimista, -np.inf)
        c = int(np.argmax(nota))
        return c, float(nota[c])

    for fold in range(1, n_folds + 1):
        c, nota = melhor(fold - 1)
        if nota > 0:
            h.folds_ativos += 1
            h.vitorias += int(vit[c, fold])
            h.derrotas += int(der[c, fold])
            h.empates += int(emp[c, fold])

    c, nota = melhor(n_folds)
    if nota > 0:
        h.config = configs[c]
        h.treino_n = int(ac_vit[c, n_folds] + ac_der[c, n_folds])
        h.treino_taxa = float(ac_vit[c, n_folds] / h.treino_n)

    preencher_estatisticas(h, z, pq)
    return h


def finalizar(hipoteses: list[Hipotese], pq: Pesquisa) -> None:
    """Aplica a correção para múltiplos testes e decide o que está aprovado."""
    testaveis = [h for h in hipoteses if h.testavel]
    for h, q in zip(testaveis, benjamini_hochberg([h.p_valor for h in testaveis])):
        h.q_valor = float(q)
    for h in hipoteses:
        if not h.testavel:
            h.situacao = ("sem vantagem no treino" if h.folds_ativos == 0
                          else f"poucas operações fora da amostra ({h.n})")
            continue
        significativa = h.q_valor is not None and h.q_valor <= pq.fdr_q and (h.wilson_inf or 0.0) > h.breakeven
        if significativa and h.config is not None:
            h.aprovado, h.situacao = True, "aprovada"
        elif significativa:
            h.situacao = "aprovada no teste, mas sem configuração válida hoje"
        else:
            h.situacao = "sem vantagem comprovada"


def _entrada(h: Hipotese, tipo: str) -> dict:
    tf_min, exp_min = interpretar_modo(h.modo)
    return {
        "tipo": tipo, "ativo": h.ativo, "modo": h.modo, "tf": tf_min * 60, "exp_min": exp_min,
        "estrategia": h.estrategia, "params": h.config["params"], "sessao": h.config["sessao"],
        "vol": h.config["vol"], "payout": h.payout, "breakeven": h.breakeven, "n_teste": h.n,
        "taxa": h.taxa, "p_inf": h.wilson_inf, "ev": h.ev, "ev_inf": h.ev_inf, "q_valor": h.q_valor,
    }


def _entradas_do_modelo(h: Hipotese, tipo: str, info_ativos: dict | None, limite: int) -> list[dict]:
    """Uma hipótese do modelo vale para um grupo de ativos: vira uma entrada por ativo,
    começando pelos abertos de maior payout."""
    tf_min, exp_min = interpretar_modo(h.modo)
    info_ativos = info_ativos or {}
    nomes = (h.detalhe or {}).get("ativos", [])
    candidatos = sorted(nomes, key=lambda a: (not (info_ativos[a].aberto if a in info_ativos else False),
                                              -(info_ativos[a].payout if a in info_ativos else h.payout), a))
    entradas = []
    for ativo in candidatos[:limite]:
        payout = info_ativos[ativo].payout if ativo in info_ativos and info_ativos[ativo].payout > 0 else h.payout
        entradas.append({
            "tipo": tipo, "ativo": ativo, "modo": h.modo, "tf": tf_min * 60, "exp_min": exp_min,
            "estrategia": h.estrategia, "params": h.config["params"], "sessao": h.config["sessao"],
            "vol": h.config["vol"], "payout": payout, "breakeven": breakeven(payout), "n_teste": h.n,
            "taxa": h.taxa, "p_inf": h.wilson_inf, "ev": valor_esperado(h.taxa, payout),
            "ev_inf": valor_esperado(h.wilson_inf, payout), "q_valor": h.q_valor, "grupo": h.ativo,
        })
    return entradas


def montar_plano(hipoteses: list[Hipotese], max_exploracao: int, info_ativos: dict | None = None,
                 max_ativos_modelo: int = 10) -> list[dict]:
    """Entradas do plano: primeiro as aprovadas, depois as candidatas de exploração (só demo)."""
    from .ml import NOME as MODELO

    aprovadas = sorted((h for h in hipoteses if h.aprovado), key=lambda h: h.ev_inf, reverse=True)
    candidatas = sorted((h for h in hipoteses if h.testavel and not h.aprovado and h.config is not None),
                        key=lambda h: h.ev, reverse=True)[:max_exploracao]
    plano = []
    for tipo, lista, limite in (("aprovado", aprovadas, max_ativos_modelo), ("exploracao", candidatas, 2)):
        for h in lista:
            if h.estrategia == MODELO:
                plano += _entradas_do_modelo(h, tipo, info_ativos, limite)
            else:
                plano.append(_entrada(h, tipo))
    return plano


def pesquisar(banco: Banco, cfg: Config, log: Callable[[str], None] = print,
              agora: float | None = None, salvar: bool = True) -> dict:
    """Roda a pesquisa em todos os ativos com velas no banco e grava o resultado.

    Com `salvar=False` é um ensaio: nada é gravado no banco (o plano do bot não muda).
    """
    from . import ml

    pq = cfg.pesquisa
    agora = agora if agora is not None else time.time()
    estrategias = pq.estrategias or list(ESTRATEGIAS)
    desconhecidas = [e for e in estrategias if e not in ESTRATEGIAS]
    if desconhecidas:
        raise ValueError(f"Estratégias desconhecidas no config.toml: {', '.join(desconhecidas)}")
    modos = [interpretar_modo(m) for m in pq.modos]
    info_ativos = banco.ativos("turbo")
    desde = int(agora - cfg.coleta.dias_historico * 86400)

    hipoteses: list[Hipotese] = []
    total_velas, ts_min, ts_max, usados = 0, None, None, 0
    nomes = banco.ativos_com_velas(60)
    modelo = None
    if cfg.ml.ativo and ml.disponivel():
        modelo = ml.PesquisaML(pq, cfg.ml, nomes, cfg.dir_modelos, log)
    elif cfg.ml.ativo:
        log("Modelo de aprendizado de máquina desligado: falta a biblioteca scikit-learn (rode instalar.bat).")
    try:
        for i, ativo in enumerate(nomes, 1):
            v1 = banco.carregar_velas(ativo, 60, desde)
            if len(v1) < MIN_VELAS:
                log(f"  [{i}/{len(nomes)}] {ativo}: poucos dados ({len(v1)} velas), pulando")
                continue
            info = info_ativos.get(ativo)
            payout = info.payout if info and info.payout > 0 else pq.payout_padrao
            antes = len(hipoteses)
            for tf_min, exp_min in modos:
                v = v1 if tf_min == 1 else v1.reamostrar(tf_min * 60)
                hipoteses.extend(avaliar_modo(ativo, v, tf_min, exp_min, payout, pq, estrategias))
            if modelo is not None:
                modelo.adicionar(ativo, v1, payout)
            usados += 1
            total_velas += len(v1)
            ts_min = int(v1.ts[0]) if ts_min is None else min(ts_min, int(v1.ts[0]))
            ts_max = int(v1.ts[-1]) if ts_max is None else max(ts_max, int(v1.ts[-1]))
            log(f"  [{i}/{len(nomes)}] {ativo}: {len(v1)} velas, payout {payout:.0%}, "
                f"{len(hipoteses) - antes} hipóteses")
        if modelo is not None:
            log("Treinando o modelo de aprendizado de máquina nas velas reais (leva alguns minutos)...")
            do_modelo = modelo.avaliar()
            hipoteses.extend(do_modelo)
            for h in do_modelo:
                if h.n:
                    acerto = f"{h.taxa:.1%} em {h.n} operações fora da amostra"
                else:
                    amplo = ml.diagnostico_amplo(h.detalhe)
                    acerto = "ficou de fora em todas as janelas" + (
                        f"; se operasse sempre no limiar {amplo['limiar']:.0%}, teria acertado {amplo['taxa']:.1%} "
                        f"em {amplo['n']} operações fora da amostra" if amplo else "")
                log(f"  Modelo, {h.ativo} {h.modo}: {acerto} (precisa de {h.breakeven:.1%})")
    finally:
        if modelo is not None:
            modelo.fechar()

    finalizar(hipoteses, pq)
    testaveis = [h for h in hipoteses if h.testavel]
    resumo = {
        "ativos": usados, "velas": total_velas, "periodo": [ts_min, ts_max],
        "hipoteses": len(hipoteses), "testaveis": len(testaveis),
        "aprovadas": sum(h.aprovado for h in hipoteses),
        "p_menor_5": sum(h.p_valor < 0.05 for h in testaveis),
        "esperado_acaso": round(0.05 * len(testaveis), 1),
        "taxa_media": float(np.mean([h.taxa for h in testaveis])) if testaveis else None,
        "breakeven_medio": float(np.mean([h.breakeven for h in testaveis])) if testaveis else None,
        "fdr_q": pq.fdr_q, "confianca": pq.confianca,
    }
    run_id = datetime.fromtimestamp(agora, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plano = {"run_id": run_id, "gerado_em": int(agora),
             "entradas": montar_plano(hipoteses, cfg.operacao.max_candidatos_exploracao, info_ativos,
                                      cfg.ml.max_ativos_plano)}
    registro = [asdict(h) | {"n": h.n} for h in hipoteses]
    if salvar:
        banco.salvar_pesquisa(run_id, agora, resumo, registro, plano)
    return {"run_id": run_id, "ts": int(agora), "resumo": resumo, "hipoteses": registro, "plano": plano}
