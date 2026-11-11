"""Modelo de aprendizado de máquina local, treinado nas velas reais do banco.

É um gradient boosting pequeno (scikit-learn) que estima a chance de a próxima vela
(ou as próximas, conforme a expiração) fechar acima do preço de entrada. O bot só
entra quando essa chance passa de um limiar, para CALL, ou fica abaixo do espelho
dele, para PUT.

Um modelo por grupo de ativos e por modo. Os ativos OTC, cujo preço é gerado pela
corretora, formam um grupo; os de mercado aberto, outro. Juntar os ativos de um grupo
dá muito mais dados para achar um padrão fraco do que olhar um ativo por vez.

A validação segue o mesmo rigor das regras (walk-forward):
  1. Para cada janela de teste, o modelo é treinado só com dados anteriores a ela.
  2. O limiar é escolhido num trecho de validação que o modelo não viu no treino.
     Se nenhum limiar mostra lucro pessimista positivo, o modelo fica de fora da janela.
  3. Só o resultado nas janelas de teste conta para aprovar a hipótese do grupo.
  4. O modelo usado ao vivo passa pelo mesmo procedimento com os dados mais recentes.

Memória: os atributos de cada ativo vão para arquivos temporários em disco e só uma
amostra das linhas fica na memória para o treino.
"""
from __future__ import annotations

import math
import os
import pickle
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import indicadores as ind
from .backtest import desfechos, sem_sobreposicao
from .config import Ml, Pesquisa, interpretar_modo
from .estatistica import breakeven, wilson_inferior_vetor, z_unilateral
from .filtros import razao_vol
from .velas import Velas

NOME = "modelo_ml"
LIMIARES = (0.52, 0.54, 0.56, 0.58, 0.60, 0.65)
FRACAO_AJUSTE = 0.75  # parte inicial do treino usada para ajustar o modelo; o resto valida o limiar
SEMENTE = 7
AQUECIMENTO = 130  # velas iniciais descartadas (indicadores ainda instáveis)
MIN_VELAS_ML = 1500
MIN_LINHAS_AJUSTE = 3000
MIN_VALIDACAO = 300
VERSAO = 1  # muda quando a lista de atributos muda; modelos de outra versão são ignorados

NOMES_ATRIBUTOS = (
    "corpo_0", "corpo_1", "corpo_2", "corpo_3", "corpo_4",
    "retorno_3", "retorno_5", "retorno_10", "retorno_20", "retorno_60",
    "amplitude", "pavio_superior", "pavio_inferior",
    "rsi_7", "rsi_14", "bollinger_b", "estocastico_k", "estocastico_k_menos_d", "macd_hist",
    "dist_media_9", "dist_media_21", "media_9_menos_21", "volatilidade_relativa", "sequencia",
    "hora_seno", "hora_cosseno", "posicao_no_bloco",
)

GRUPOS = {"OTC": "Grupo OTC", "mercado": "Grupo mercado aberto"}


def grupo_de(ativo: str) -> str:
    """'OTC' para ativos com preço gerado pela corretora; 'mercado' para os demais."""
    return "OTC" if ativo.upper().endswith("-OTC") else "mercado"


def rotulo_grupo(grupo: str) -> str:
    return GRUPOS.get(grupo, grupo)


def disponivel() -> bool:
    """A biblioteca scikit-learn está instalada?"""
    try:
        import sklearn.ensemble  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- atributos
def atributos(v: Velas) -> np.ndarray:
    """Matriz (velas x atributos) com o que o modelo enxerga em cada vela.

    Tudo é causal (a linha t só usa dados até o fechamento da vela t) e sem unidade:
    preços entram divididos pela volatilidade recente, para valer em qualquer ativo.
    """
    n = len(v)
    if n < 2:
        return np.full((n, len(NOMES_ATRIBUTOS)), np.nan, dtype=np.float32)
    o, h, l, c = v.open, v.high, v.low, v.close
    with np.errstate(divide="ignore", invalid="ignore"):
        a = ind.atr(h, l, c, 14)
        a = np.where(a > 0, a, np.nan)
        corpo = (c - o) / a
        colunas = [corpo] + [ind.anterior(corpo, j) for j in range(1, 5)]
        colunas += [(c - ind.anterior(c, k)) / (a * math.sqrt(k)) for k in (3, 5, 10, 20, 60)]
        faixa = h - l
        faixa_ok = np.where(faixa > 0, faixa, 1.0)
        colunas.append(faixa / a)
        colunas.append(np.where(faixa > 0, (h - np.maximum(o, c)) / faixa_ok, 0.0))
        colunas.append(np.where(faixa > 0, (np.minimum(o, c) - l) / faixa_ok, 0.0))
        colunas.append(ind.rsi(c, 7) / 50.0 - 1.0)
        colunas.append(ind.rsi(c, 14) / 50.0 - 1.0)
        media, superior, _ = ind.bollinger(c, 20, 2.0)
        meia = superior - media
        colunas.append(np.where(meia > 0, (c - media) / np.where(meia > 0, meia, 1.0), np.where(np.isnan(meia), np.nan, 0.0)))
        k_, d_ = ind.estocastico(h, l, c, 14)
        colunas.append(k_ / 50.0 - 1.0)
        colunas.append((k_ - d_) / 50.0)
        colunas.append(ind.macd(c)[2] / a)
        m9, m21 = ind.ema(c, 9), ind.ema(c, 21)
        colunas += [(c - m9) / a, (c - m21) / a, (m9 - m21) / a]
        colunas.append(razao_vol(v) - 1.0)
    cor = np.sign(c - o)
    mudou = np.r_[True, cor[1:] != cor[:-1]]
    sequencia = (np.arange(n) - np.flatnonzero(mudou)[np.cumsum(mudou) - 1] + 1) * cor
    colunas.append(np.clip(sequencia, -8, 8))
    angulo = ((v.ts + v.tf) % 86400) / 86400.0 * 2.0 * math.pi  # hora da ENTRADA (abertura da vela seguinte)
    colunas += [np.sin(angulo), np.cos(angulo), ((v.ts // v.tf) % 5).astype(np.float64)]
    x = np.clip(np.column_stack(colunas), -20.0, 20.0).astype(np.float32)
    x[:AQUECIMENTO] = np.nan
    return x


# --------------------------------------------------------------------------- modelo
def treinar(x: np.ndarray, y: np.ndarray):
    """Gradient boosting pequeno e bem regularizado: o sinal, se existir, é fraco."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    modelo = HistGradientBoostingClassifier(
        max_iter=60, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=max(100, len(y) // 2000),
        l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=SEMENTE,
    )
    return modelo.fit(x, y)


def prever(modelo, x: np.ndarray) -> np.ndarray:
    """Chance estimada de alta (classe 1) para cada linha."""
    return modelo.predict_proba(x)[:, 1]


def sinais(p: np.ndarray, limiar: float) -> np.ndarray:
    """+1 (CALL) se a chance de alta >= limiar; -1 (PUT) se <= 1 - limiar; senão 0."""
    return np.where(p >= limiar, 1, np.where(p <= 1.0 - limiar, -1, 0)).astype(np.int8)


def salvar_modelo(caminho: Path, modelo, extra: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    provisorio = caminho.with_suffix(".tmp")
    with provisorio.open("wb") as arq:
        pickle.dump({"versao": VERSAO, "atributos": NOMES_ATRIBUTOS, "modelo": modelo, **extra}, arq)
    os.replace(provisorio, caminho)


_CACHE: dict[str, tuple[float, dict | None]] = {}


def carregar_modelo(caminho: str | Path | None) -> dict | None:
    """Lê um modelo salvo por `salvar_modelo`. Devolve None se faltar ou for de outra versão.

    O arquivo é um pickle: só deve ser aberto se foi gravado por este programa, na pasta
    `modelos` ao lado do banco de dados.
    """
    if not caminho:
        return None
    caminho = Path(caminho)
    if caminho.suffix != ".pkl" or caminho.parent.name != "modelos":
        return None
    try:
        mtime = caminho.stat().st_mtime
    except OSError:
        return None
    chave = str(caminho)
    if chave in _CACHE and _CACHE[chave][0] == mtime:
        return _CACHE[chave][1]
    pacote: dict | None
    try:
        with caminho.open("rb") as arq:
            pacote = pickle.load(arq)
        if not isinstance(pacote, dict) or pacote.get("versao") != VERSAO or "modelo" not in pacote:
            pacote = None
    except Exception:  # noqa: BLE001 - arquivo corrompido ou de outra versão das bibliotecas
        pacote = None
    _CACHE[chave] = (mtime, pacote)
    return pacote


def sinal_atual(v: Velas, params: dict) -> int:
    """Sinal do modelo para a última vela fechada: +1 CALL, -1 PUT, 0 nada."""
    pacote = carregar_modelo(params.get("arquivo"))
    if pacote is None or len(v) <= AQUECIMENTO:
        return 0
    x = atributos(v)[-1:]
    if not np.isfinite(x).all():
        return 0
    return int(sinais(prever(pacote["modelo"], x), float(params["limiar"]))[0])


# --------------------------------------------------------------------------- pesquisa
@dataclass
class _Serie:
    ativo: str
    grupo: str
    tf_min: int
    payout: float
    caminho: Path
    ts_ini: int
    ts_fim: int


class PesquisaML:
    """Recebe as velas de cada ativo, treina os modelos por grupo e devolve as hipóteses."""

    def __init__(self, pq: Pesquisa, ml: Ml, ativos: list[str], dir_modelos: Path,
                 log: Callable[[str], None] = print):
        self.pq, self.ml, self.dir_modelos, self.log = pq, ml, Path(dir_modelos), log
        self.modos = [interpretar_modo(m) for m in pq.modos]
        self.timeframes = sorted({tf for tf, _ in self.modos})
        contagem = Counter(grupo_de(a) for a in ativos)
        self.cota = {g: max(2000, ml.max_linhas_treino // max(1, q)) for g, q in contagem.items()}
        self.tmp = Path(tempfile.mkdtemp(prefix="iq_agent_ml_"))
        self.series: list[_Serie] = []
        self.amostra: dict[tuple[str, int], dict[str, list]] = {}
        self.rng = np.random.default_rng(SEMENTE)

    def fechar(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def adicionar(self, ativo: str, v1: Velas, payout: float) -> None:
        """Calcula atributos e desfechos de um ativo, guarda em disco e separa a amostra de treino."""
        grupo = grupo_de(ativo)
        for tf_min in self.timeframes:
            v = v1 if tf_min == 1 else v1.reamostrar(tf_min * 60)
            if len(v) < MIN_VELAS_ML // tf_min:
                continue
            x = atributos(v)
            ok = np.isfinite(x).all(axis=1)
            dados = {"x": x.astype(np.float16), "ts": v.ts, "ok": ok}
            rotulos = {}
            for tf2, exp_min in self.modos:
                if tf2 != tf_min:
                    continue
                chave = f"{tf_min}_{exp_min}"
                res_call, res_put, valido = desfechos(v, exp_min // tf_min, self.pq.slippage_frac)
                dados[f"rc_{chave}"], dados[f"rp_{chave}"], dados[f"va_{chave}"] = res_call, res_put, valido
                y = np.full(len(v), -1, dtype=np.int8)  # -1 = sem rótulo (inválida ou nem alta nem baixa)
                y[valido & (res_call > 0)] = 1
                y[valido & (res_put > 0)] = 0
                rotulos[chave] = y
            caminho = self.tmp / f"{len(self.series):04d}.npz"
            np.savez(caminho, **dados)
            self.series.append(_Serie(ativo, grupo, tf_min, payout, caminho, int(v.ts[0]), int(v.ts[-1]) + v.tf))
            candidatas = np.flatnonzero(ok)
            if candidatas.size == 0:
                continue
            escolhidas = np.sort(self.rng.choice(candidatas, size=min(self.cota.get(grupo, 2000), candidatas.size),
                                                 replace=False))
            lote = self.amostra.setdefault((grupo, tf_min), {"x": [], "ts": []})
            lote["x"].append(x[escolhidas])
            lote["ts"].append(v.ts[escolhidas])
            for chave, y in rotulos.items():
                lote.setdefault(f"y_{chave}", []).append(y[escolhidas])

    def avaliar(self) -> list:
        """Uma hipótese por (grupo, modo)."""
        saida = []
        for grupo in sorted({s.grupo for s in self.series}):
            for tf_min, exp_min in self.modos:
                h = self._avaliar_grupo(grupo, tf_min, exp_min)
                if h is not None:
                    saida.append(h)
        return saida

    def _avaliar_grupo(self, grupo: str, tf_min: int, exp_min: int):
        from .pesquisa import Hipotese, limites_folds, preencher_estatisticas

        pq = self.pq
        series = [s for s in self.series if s.grupo == grupo and s.tf_min == tf_min]
        lote = self.amostra.get((grupo, tf_min))
        chave, modo = f"{tf_min}_{exp_min}", f"M{tf_min}>{exp_min}"
        if not series or lote is None or f"y_{chave}" not in lote:
            return None
        k, tf = exp_min // tf_min, tf_min * 60
        inicio, fim = min(s.ts_ini for s in series), max(s.ts_fim for s in series)
        limites = limites_folds(inicio, fim, pq.n_folds, pq.dias_teste)
        if limites is None:
            return None
        n_folds = len(limites) - 1
        payout = min(s.payout for s in series)  # o pior payout do grupo: critério conservador
        p0, z = breakeven(payout), z_unilateral(pq.confianca)
        x_am = np.concatenate(lote["x"])
        y_am = np.concatenate(lote[f"y_{chave}"])
        saida_am = np.concatenate(lote["ts"]) + (k + 1) * tf  # instante em que o desfecho fica conhecido
        min_validacao = max(pq.min_trades_treino, MIN_VALIDACAO)

        h = Hipotese(ativo=rotulo_grupo(grupo), modo=modo, estrategia=NOME, payout=payout, breakeven=p0,
                     folds=n_folds)
        por_ativo: dict[str, list[int]] = {}
        # Diagnóstico: placar fora da amostra de CADA limiar, mesmo quando o modelo ficou de fora.
        # Não entra na aprovação; serve para mostrar a que distância ele ficou do necessário.
        diagnostico = np.zeros((len(LIMIARES), 2))
        validacao_final = np.zeros((len(LIMIARES), 2))
        for j in range(n_folds + 1):
            final = j == n_folds  # a última rodada treina o modelo que vai para o plano
            fim_treino = fim if final else int(limites[j])
            fim_janela = fim_treino if final else int(limites[j + 1])
            corte = inicio + FRACAO_AJUSTE * (fim_treino - inicio)
            ajuste = (saida_am <= corte) & (y_am >= 0)
            if int(ajuste.sum()) < MIN_LINHAS_AJUSTE or len(np.unique(y_am[ajuste])) < 2:
                continue
            modelo = treinar(x_am[ajuste], y_am[ajuste])
            validacao = np.zeros((len(LIMIARES), 2))
            teste: list[dict[str, list[int]]] = [{} for _ in LIMIARES]
            for s in series:
                with np.load(s.caminho) as d:
                    ts, ok = d["ts"], d["ok"]
                    saida = ts + (k + 1) * tf
                    linhas = np.flatnonzero(ok & d[f"va_{chave}"] & (saida > corte) & (saida <= fim_janela))
                    if linhas.size == 0:
                        continue
                    p = prever(modelo, d["x"][linhas].astype(np.float32))
                    trecho = slice(int(linhas[0]), int(linhas[-1]) + 1)
                    res_call, res_put = d[f"rc_{chave}"][trecho], d[f"rp_{chave}"][trecho]
                em_teste = saida[trecho] > fim_treino
                for i, limiar in enumerate(LIMIARES):
                    sinal = np.zeros(trecho.stop - trecho.start, dtype=np.int8)
                    sinal[linhas - trecho.start] = sinais(p, limiar)
                    sinal = sem_sobreposicao(sinal, k)
                    ganhou = ((sinal > 0) & (res_call > 0)) | ((sinal < 0) & (res_put > 0))
                    perdeu = ((sinal > 0) & (res_call < 0)) | ((sinal < 0) & (res_put < 0))
                    validacao[i] += (int((ganhou & ~em_teste).sum()), int((perdeu & ~em_teste).sum()))
                    if not final:
                        placar = teste[i].setdefault(s.ativo, [0, 0, 0])
                        placar[0] += int((ganhou & em_teste).sum())
                        placar[1] += int((perdeu & em_teste).sum())
                        placar[2] += int(((sinal != 0) & em_teste & ~ganhou & ~perdeu).sum())
            if final:
                validacao_final = validacao
            else:
                for i, placares in enumerate(teste):
                    diagnostico[i] += (sum(p[0] for p in placares.values()), sum(p[1] for p in placares.values()))
            total = validacao.sum(axis=1)
            nota = np.where(total >= min_validacao,
                            total * (wilson_inferior_vetor(validacao[:, 0], total, z) * (1.0 + payout) - 1.0),
                            -np.inf)
            i = int(np.argmax(nota))
            if not nota[i] > 0:
                continue  # nenhum limiar mostrou vantagem na validação: o modelo fica de fora
            if final:
                caminho = self.dir_modelos / f"{grupo}_{chave}.pkl"
                salvar_modelo(caminho, modelo, {"grupo": grupo, "modo": modo, "limiar": LIMIARES[i],
                                                "treinado_ate": int(corte), "linhas": int(ajuste.sum())})
                h.config = {"params": {"limiar": LIMIARES[i], "arquivo": str(caminho)},
                            "sessao": "todas", "vol": "qualquer"}
                h.treino_n = int(total[i])
                h.treino_taxa = float(validacao[i, 0] / total[i])
            else:
                h.folds_ativos += 1
                for ativo, (vit, der, emp) in teste[i].items():
                    h.vitorias += vit
                    h.derrotas += der
                    h.empates += emp
                    soma = por_ativo.setdefault(ativo, [0, 0])
                    soma[0] += vit
                    soma[1] += vit + der
        preencher_estatisticas(h, z, pq)
        ordenados = sorted(por_ativo.items(), key=lambda kv: -kv[1][1])
        h.detalhe = {
            "grupo": grupo, "ativos": sorted(s.ativo for s in series), "linhas_treino": int(len(y_am)),
            "por_ativo": [{"ativo": a, "vitorias": w, "n": q} for a, (w, q) in ordenados[:15] if q],
            "fora_da_amostra_por_limiar": _placar_por_limiar(diagnostico),
            "validacao_final_por_limiar": _placar_por_limiar(validacao_final),
        }
        return h


def _placar_por_limiar(placar: np.ndarray) -> list[dict]:
    """[{limiar, n, taxa}] para cada limiar testado (taxa None quando não houve operação)."""
    saida = []
    for limiar, (vitorias, derrotas) in zip(LIMIARES, placar):
        n = int(vitorias + derrotas)
        saida.append({"limiar": limiar, "n": n, "taxa": float(vitorias / n) if n else None})
    return saida


def diagnostico_amplo(detalhe: dict | None) -> dict | None:
    """Acerto fora da amostra no limiar com MAIS operações (o mais baixo, na prática).

    É de propósito que não se escolhe o limiar que mais acertou: com poucas operações o
    acerto oscila muito, e escolher o maior premiaria a sorte. O limiar mais amplo dá a
    estimativa mais firme de quanto o modelo acerta quando opera.
    """
    linhas = [d for d in (detalhe or {}).get("fora_da_amostra_por_limiar", []) if d["n"] > 0]
    return max(linhas, key=lambda d: d["n"]) if linhas else None
