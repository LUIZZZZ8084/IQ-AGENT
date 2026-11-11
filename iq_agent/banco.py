"""Banco local (SQLite): velas, payouts, resultados de pesquisa e diário de operações."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .tipos import Ativo
from .velas import Velas

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS velas (
    ativo TEXT NOT NULL, tf INTEGER NOT NULL, ts INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
    PRIMARY KEY (ativo, tf, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ativos (
    nome TEXT NOT NULL, tipo TEXT NOT NULL, ativo_id INTEGER NOT NULL,
    payout REAL NOT NULL, aberto INTEGER NOT NULL, aposta_min REAL, aposta_max REAL, ts INTEGER NOT NULL,
    PRIMARY KEY (nome, tipo)
);

CREATE TABLE IF NOT EXISTS pesquisas (
    run_id TEXT PRIMARY KEY, ts INTEGER NOT NULL,
    resumo TEXT NOT NULL, hipoteses TEXT NOT NULL, plano TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operacoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_sinal INTEGER NOT NULL, ts_ordem REAL, conta TEXT NOT NULL, tipo TEXT NOT NULL,
    ativo TEXT NOT NULL, modo TEXT NOT NULL, estrategia TEXT NOT NULL, config TEXT NOT NULL,
    direcao TEXT NOT NULL, valor REAL NOT NULL, payout REAL NOT NULL, expiracao INTEGER,
    opcao_id INTEGER, status TEXT NOT NULL, lucro REAL NOT NULL,
    saldo_antes REAL, saldo_depois REAL, atraso_s REAL, fonte TEXT, run_id TEXT, erro TEXT
);
CREATE INDEX IF NOT EXISTS idx_operacoes_ts ON operacoes (ts_sinal);
"""


class Banco:
    def __init__(self, caminho: Path | str, somente_leitura: bool = False):
        self.caminho = Path(caminho)
        if somente_leitura:  # para ensaios: lê o banco real sem poder alterar nada nele
            self.con = sqlite3.connect(f"{self.caminho.resolve().as_uri()}?mode=ro", uri=True)
            self.con.row_factory = sqlite3.Row
            return
        if str(caminho) != ":memory:":
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(caminho))
        self.con.row_factory = sqlite3.Row
        if str(caminho) != ":memory:":
            self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(_ESQUEMA)
        self.con.commit()

    def fechar(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------ velas
    def inserir_velas(self, ativo: str, tf: int, linhas: Iterable[tuple]) -> int:
        """Grava tuplas (ts, open, high, low, close, volume). Repetidas são substituídas."""
        dados = [(ativo, tf, int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]))
                 for l in linhas]
        if not dados:
            return 0
        self.con.executemany("INSERT OR REPLACE INTO velas VALUES (?,?,?,?,?,?,?,?)", dados)
        self.con.commit()
        return len(dados)

    def inserir_serie(self, ativo: str, v: Velas) -> int:
        return self.inserir_velas(ativo, v.tf, zip(v.ts.tolist(), v.open.tolist(), v.high.tolist(),
                                                   v.low.tolist(), v.close.tolist(), v.volume.tolist()))

    def carregar_velas(self, ativo: str, tf: int = 60, desde: int | None = None) -> Velas:
        cur = self.con.execute(
            "SELECT ts, open, high, low, close, volume FROM velas WHERE ativo=? AND tf=? AND ts>=? ORDER BY ts",
            (ativo, tf, desde or 0),
        )
        dados = np.array(cur.fetchall(), dtype=np.float64)
        if dados.size == 0:
            return Velas.vazia(tf)
        return Velas(dados[:, 0].astype(np.int64), dados[:, 1].copy(), dados[:, 2].copy(), dados[:, 3].copy(),
                     dados[:, 4].copy(), dados[:, 5].copy(), tf)

    def intervalo_velas(self, ativo: str, tf: int = 60) -> tuple[int | None, int | None, int]:
        linha = self.con.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM velas WHERE ativo=? AND tf=?", (ativo, tf)
        ).fetchone()
        return linha[0], linha[1], linha[2]

    def ativos_com_velas(self, tf: int = 60) -> list[str]:
        cur = self.con.execute("SELECT DISTINCT ativo FROM velas WHERE tf=? ORDER BY ativo", (tf,))
        return [l[0] for l in cur.fetchall()]

    def apagar_velas_antes(self, ts: int) -> int:
        cur = self.con.execute("DELETE FROM velas WHERE ts<?", (int(ts),))
        self.con.commit()
        return cur.rowcount

    # ----------------------------------------------------------------- ativos
    def salvar_ativos(self, ativos: Iterable[Ativo], ts: float | None = None) -> None:
        ts = int(ts if ts is not None else time.time())
        self.con.executemany(
            "INSERT OR REPLACE INTO ativos VALUES (?,?,?,?,?,?,?,?)",
            [(a.nome, a.tipo, a.id, a.payout, int(a.aberto), a.aposta_min, a.aposta_max, ts) for a in ativos],
        )
        self.con.commit()

    def ativos(self, tipo: str = "turbo") -> dict[str, Ativo]:
        cur = self.con.execute("SELECT * FROM ativos WHERE tipo=?", (tipo,))
        return {
            l["nome"]: Ativo(l["ativo_id"], l["nome"], l["tipo"], l["payout"], bool(l["aberto"]),
                             l["aposta_min"], l["aposta_max"])
            for l in cur.fetchall()
        }

    # --------------------------------------------------------------- pesquisa
    def salvar_pesquisa(self, run_id: str, ts: float, resumo: dict, hipoteses: list[dict], plano: dict) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO pesquisas VALUES (?,?,?,?,?)",
            (run_id, int(ts), json.dumps(resumo), json.dumps(hipoteses), json.dumps(plano)),
        )
        self.con.commit()

    def ultima_pesquisa(self) -> dict[str, Any] | None:
        l = self.con.execute("SELECT * FROM pesquisas ORDER BY ts DESC LIMIT 1").fetchone()
        if l is None:
            return None
        return {
            "run_id": l["run_id"], "ts": l["ts"], "resumo": json.loads(l["resumo"]),
            "hipoteses": json.loads(l["hipoteses"]), "plano": json.loads(l["plano"]),
        }

    # -------------------------------------------------------------- operações
    def registrar_operacao(self, **campos: Any) -> int:
        if isinstance(campos.get("config"), dict):
            campos["config"] = json.dumps(campos["config"], sort_keys=True)
        colunas = ", ".join(campos)
        marcas = ", ".join("?" for _ in campos)
        cur = self.con.execute(f"INSERT INTO operacoes ({colunas}) VALUES ({marcas})", list(campos.values()))
        self.con.commit()
        return int(cur.lastrowid)

    def operacoes(self, desde: float | None = None, conta: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM operacoes WHERE ts_sinal>=?", [int(desde or 0)]
        if conta:
            sql += " AND conta=?"
            args.append(conta)
        return [dict(l) for l in self.con.execute(sql + " ORDER BY ts_sinal, id", args).fetchall()]
