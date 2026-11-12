"""Trava de instância única: impede dois painéis, ou dois bots, sobre os mesmos dados.

Dois bots na mesma conta abririam operações ao mesmo tempo e um atrapalharia a
conferência de resultado do outro, que é feita pelo saldo.

A trava é um arquivo mantido aberto e bloqueado enquanto o programa roda. Se o programa
fechar de qualquer jeito (até por queda de energia), o sistema operacional solta a trava.
"""
from __future__ import annotations

import os
from pathlib import Path


class Trava:
    def __init__(self, caminho: Path | str):
        self.caminho = Path(caminho)
        self._arquivo = None

    def adquirir(self) -> bool:
        """Tenta pegar a trava. Devolve False se outro programa já está com ela."""
        if self._arquivo is not None:
            return True
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        arquivo = open(self.caminho, "a+")  # noqa: SIM115 - fica aberto de propósito enquanto a trava vale
        try:
            arquivo.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(arquivo.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(arquivo.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            arquivo.close()
            return False
        self._arquivo = arquivo
        return True

    def liberar(self) -> None:
        arquivo, self._arquivo = self._arquivo, None
        if arquivo is None:
            return
        try:
            arquivo.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(arquivo.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(arquivo.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            arquivo.close()

    def __enter__(self) -> "Trava":
        return self

    def __exit__(self, *exc) -> None:
        self.liberar()
