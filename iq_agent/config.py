"""Leitura do config.toml e das credenciais do .env."""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


class ErroConfig(ValueError):
    """Configuração inválida ou incompleta."""


@dataclass
class Coleta:
    dias_historico: int = 30
    payout_minimo: float = 0.70
    max_ativos: int = 150
    incluir_otc: bool = True
    ativos: list[str] = field(default_factory=list)
    pausa_entre_pedidos: float = 0.2


@dataclass
class Pesquisa:
    modos: list[str] = field(default_factory=lambda: ["M1>1", "M1>5", "M5>5"])
    estrategias: list[str] = field(default_factory=list)
    n_folds: int = 5
    dias_teste: float = 3.0
    min_trades_treino: int = 150
    min_trades_teste: int = 100
    fdr_q: float = 0.10
    confianca: float = 0.95
    slippage_frac: float = 0.02
    payout_padrao: float = 0.80


@dataclass
class Risco:
    risco_por_operacao: float = 0.01
    kelly_fracao: float = 0.25
    valor_minimo: float = 1.0
    valor_maximo: float = 50.0
    perda_maxima_diaria: float = 0.05
    meta_diaria: float = 0.0
    max_operacoes_dia: int = 20
    max_perdas_seguidas: int = 3
    pausa_apos_perdas_min: int = 30


@dataclass
class Operacao:
    modo_exploracao: bool = False
    valor_exploracao: float = 1.0
    max_candidatos_exploracao: int = 3
    atraso_maximo_s: float = 6.0
    espera_fechamento_s: float = 1.0
    min_ops_para_suspender: int = 30


@dataclass
class Real:
    """Regras para a conta real. O padrão é bloqueado: quem libera é o usuário."""

    permitido: bool = False
    exigir_prontidao: bool = True
    valor_maximo: float = 5.0
    min_operacoes_demo: int = 100
    min_dias_demo: int = 10
    janela_dias: int = 60


@dataclass
class Ml:
    """Modelo de aprendizado de máquina local (ver iq_agent/ml.py)."""

    ativo: bool = True
    max_linhas_treino: int = 300_000  # linhas na memória para treinar cada grupo de ativos
    max_ativos_plano: int = 10  # se o modelo de um grupo for aprovado, em quantos ativos operar


@dataclass
class Config:
    coleta: Coleta = field(default_factory=Coleta)
    pesquisa: Pesquisa = field(default_factory=Pesquisa)
    risco: Risco = field(default_factory=Risco)
    operacao: Operacao = field(default_factory=Operacao)
    real: Real = field(default_factory=Real)
    ml: Ml = field(default_factory=Ml)
    raiz: Path = RAIZ
    db: Path = RAIZ / "data" / "iq_agent.db"

    @property
    def dir_modelos(self) -> Path:
        """Modelos treinados ficam ao lado do banco de dados (só no computador do usuário)."""
        return Path(self.db).parent / "modelos"

    @property
    def dir_relatorios(self) -> Path:
        return self.raiz / "reports"

    @property
    def dir_logs(self) -> Path:
        return self.raiz / "logs"


_SECOES = {"coleta": Coleta, "pesquisa": Pesquisa, "risco": Risco, "operacao": Operacao, "real": Real, "ml": Ml}
_MODO = re.compile(r"^M(\d+)>(\d+)$")


def interpretar_modo(modo: str) -> tuple[int, int]:
    """'M1>5' -> (timeframe em minutos, expiração em minutos)."""
    m = _MODO.match(modo.strip().upper())
    if not m:
        raise ErroConfig(f"Modo inválido: {modo!r}. Use o formato 'M1>1', 'M1>5' ou 'M5>5'.")
    tf, exp = int(m.group(1)), int(m.group(2))
    if tf not in (1, 5):
        raise ErroConfig(f"Modo {modo!r}: timeframe precisa ser M1 ou M5.")
    if exp % tf or not 1 <= exp <= 5:
        raise ErroConfig(
            f"Modo {modo!r}: a expiração precisa ser múltipla do timeframe e ficar entre 1 e 5 minutos "
            "(opções turbo)."
        )
    return tf, exp


def _entre(nome: str, valor: float, minimo: float, maximo: float) -> None:
    if not minimo <= valor <= maximo:
        raise ErroConfig(f"{nome} = {valor} está fora do permitido ({minimo} a {maximo}).")


def validar(cfg: Config) -> Config:
    c, p, r, o = cfg.coleta, cfg.pesquisa, cfg.risco, cfg.operacao
    _entre("coleta.dias_historico", c.dias_historico, 3, 120)
    _entre("coleta.payout_minimo", c.payout_minimo, 0.0, 1.0)
    _entre("coleta.max_ativos", c.max_ativos, 1, 200)
    for modo in p.modos:
        interpretar_modo(modo)
    if not p.modos:
        raise ErroConfig("pesquisa.modos não pode ficar vazio.")
    _entre("pesquisa.n_folds", p.n_folds, 2, 20)
    _entre("pesquisa.dias_teste", p.dias_teste, 0.25, 30)
    _entre("pesquisa.fdr_q", p.fdr_q, 0.001, 0.5)
    _entre("pesquisa.confianca", p.confianca, 0.5, 0.999)
    _entre("pesquisa.slippage_frac", p.slippage_frac, 0.0, 1.0)
    _entre("pesquisa.payout_padrao", p.payout_padrao, 0.1, 1.0)
    # Teto de 5% por operação: acima disso uma sequência ruim comum quebra a banca.
    _entre("risco.risco_por_operacao", r.risco_por_operacao, 0.0005, 0.05)
    _entre("risco.kelly_fracao", r.kelly_fracao, 0.01, 1.0)
    _entre("risco.perda_maxima_diaria", r.perda_maxima_diaria, 0.005, 0.5)
    _entre("risco.meta_diaria", r.meta_diaria, 0.0, 5.0)
    _entre("risco.max_operacoes_dia", r.max_operacoes_dia, 1, 500)
    _entre("risco.max_perdas_seguidas", r.max_perdas_seguidas, 1, 50)
    if r.valor_minimo <= 0 or r.valor_maximo < r.valor_minimo:
        raise ErroConfig("risco.valor_minimo precisa ser positivo e menor ou igual a risco.valor_maximo.")
    _entre("operacao.atraso_maximo_s", o.atraso_maximo_s, 1.0, 25.0)
    _entre("operacao.espera_fechamento_s", o.espera_fechamento_s, 0.0, 5.0)
    if o.valor_exploracao <= 0:
        raise ErroConfig("operacao.valor_exploracao precisa ser positivo.")
    if cfg.real.valor_maximo <= 0:
        raise ErroConfig("real.valor_maximo precisa ser positivo.")
    _entre("real.min_operacoes_demo", cfg.real.min_operacoes_demo, 0, 100000)
    _entre("real.min_dias_demo", cfg.real.min_dias_demo, 0, 365)
    _entre("real.janela_dias", cfg.real.janela_dias, 1, 365)
    _entre("ml.max_linhas_treino", cfg.ml.max_linhas_treino, 10_000, 5_000_000)
    _entre("ml.max_ativos_plano", cfg.ml.max_ativos_plano, 1, 50)
    return cfg


def carregar(caminho: Path | str | None = None) -> Config:
    """Lê o config.toml (ou usa os padrões se ele não existir)."""
    caminho = Path(caminho) if caminho else RAIZ / "config.toml"
    cfg = Config()
    if caminho.exists():
        with caminho.open("rb") as arq:
            bruto = tomllib.load(arq)
        for secao, valores in bruto.items():
            if secao not in _SECOES:
                raise ErroConfig(f"Seção desconhecida no config.toml: [{secao}]")
            alvo = getattr(cfg, secao)
            validos = {f.name: f for f in fields(alvo)}
            for chave, valor in valores.items():
                if chave not in validos:
                    raise ErroConfig(f"Opção desconhecida em [{secao}]: {chave}")
                atual = getattr(alvo, chave)
                if isinstance(atual, bool) != isinstance(valor, bool):
                    raise ErroConfig(f"[{secao}] {chave}: esperado {'true/false' if isinstance(atual, bool) else 'número'}.")
                if isinstance(atual, float) and isinstance(valor, int):
                    valor = float(valor)
                if not isinstance(valor, type(atual)):
                    raise ErroConfig(f"[{secao}] {chave}: tipo errado ({type(valor).__name__}).")
                setattr(alvo, chave, valor)
    return validar(cfg)


def credenciais() -> tuple[str, str]:
    """Lê IQ_EMAIL e IQ_SENHA do ambiente ou do arquivo .env (preenchido pelo usuário)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env")
    except ImportError:  # python-dotenv é opcional: variáveis de ambiente também servem
        pass
    email = os.environ.get("IQ_EMAIL", "").strip()
    senha = os.environ.get("IQ_SENHA", "")
    if not email or not senha or email == "seu-email@exemplo.com":
        raise ErroConfig(
            "Credenciais não encontradas. Use o painel (IQ-AGENT.bat), que pede o login na hora, ou copie "
            "o arquivo .env.example para .env e preencha IQ_EMAIL e IQ_SENHA você mesmo. O .env fica só no "
            "seu computador."
        )
    return email, senha
