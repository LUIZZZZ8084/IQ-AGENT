"""Comandos de terminal: python -m iq_agent <comando>."""
from __future__ import annotations

import argparse
import sys
import webbrowser

from . import app
from .banco import Banco
from .config import ErroConfig, carregar
from .tipos import FRASE_CONTA_REAL, ErroIQ


def _saida(texto: str) -> None:
    print(texto, flush=True)


def _conectar(log, capturar: bool = False):
    cliente = app.novo_cliente(capturar=capturar)
    log("Conectando à IQ Option...")
    cliente.conectar()
    return cliente


def cmd_diagnostico(cfg, args, log) -> int:
    cliente = _conectar(log, capturar=True)
    try:
        return 0 if app.diagnosticar(cfg, cliente, log) else 1
    finally:
        cliente.fechar()


def cmd_diario(cfg, args, log) -> int:
    banco, cliente = Banco(cfg.db), _conectar(log)
    try:
        app.atualizar_mercado(cfg, banco, cliente, log)
        return 0
    finally:
        cliente.fechar()
        banco.fechar()


def cmd_coletar(cfg, args, log) -> int:
    from .coleta import coletar

    banco, cliente = Banco(cfg.db), _conectar(log)
    try:
        r = coletar(banco, cfg, cliente, log)
        log(f"Coleta concluída: {r['velas_novas']} velas novas em {r['ativos']} ativos ({r['falhas']} falhas).")
        return 0
    finally:
        cliente.fechar()
        banco.fechar()


def cmd_pesquisar(cfg, args, log) -> int:
    from .pesquisa import pesquisar

    import dataclasses
    import tempfile
    from pathlib import Path

    ensaio = getattr(args, "ensaio", False)
    if not Path(cfg.db).exists():
        log("Não há velas no banco. Rode 'diario' (ou 'coletar') primeiro para baixar o histórico real.")
        return 1
    banco = Banco(cfg.db, somente_leitura=ensaio)
    pasta = tempfile.TemporaryDirectory(prefix="iq_agent_ensaio_") if ensaio else None
    try:
        if not banco.ativos_com_velas(60):
            log("Não há velas no banco. Rode 'diario' (ou 'coletar') primeiro para baixar o histórico real.")
            return 1
        if ensaio:
            # o banco é aberto só para leitura e os modelos do ensaio vão para uma pasta temporária
            cfg = dataclasses.replace(cfg, db=Path(pasta.name) / "ensaio.db")
            log("Ensaio: nada é gravado. O plano do bot e os modelos em uso não mudam.")
        resultado = pesquisar(banco, cfg, log, salvar=not ensaio)
        r = resultado["resumo"]
        log(f"Pesquisa concluída: {r['hipoteses']} hipóteses, {r['testaveis']} com dados suficientes, "
            f"{r['aprovadas']} aprovada(s).")
        if ensaio:
            fortes = sorted((h for h in resultado["hipoteses"] if h["testavel"]), key=lambda h: h["p_valor"])[:8]
            log("As mais fortes (acerto fora da amostra / necessário):")
            for h in fortes:
                log(f"  {h['ativo']} {h['modo']} {h['estrategia']}: {h['taxa']:.1%} em {h['n']} operações / "
                    f"{h['breakeven']:.1%} | {h['situacao']}")
        return 0
    finally:
        banco.fechar()
        if pasta is not None:
            pasta.cleanup()


def cmd_relatorio(cfg, args, log) -> int:
    from .relatorio import gerar

    banco = Banco(cfg.db)
    try:
        caminho = gerar(banco, cfg)
    finally:
        banco.fechar()
    log(f"Relatório salvo em {caminho}")
    if args.abrir:
        webbrowser.open(caminho.as_uri())
    return 0


def cmd_prontidao(cfg, args, log) -> int:
    import time

    banco = Banco(cfg.db)
    try:
        pode, explicacao, itens = app.conferir_real(cfg, banco, time.time())
    finally:
        banco.fechar()
    for item in itens:
        log(f"[{'OK' if item.ok else '  '}] {item.nome}: {item.detalhe}")
    log(explicacao)
    return 0 if pode else 1


def cmd_operar(cfg, args, log) -> int:
    conta = "REAL" if args.conta == "real" else "PRACTICE"
    banco = Banco(cfg.db)
    cliente = None
    try:
        cliente = _conectar(log)
        if conta == "REAL":
            pode, explicacao, itens = app.conferir_real(cfg, banco, cliente.agora())
            for item in itens:
                log(f"[{'OK' if item.ok else '  '}] {item.nome}: {item.detalhe}")
            log(explicacao)
            if not pode:
                return 1
            if not sys.stdin or not sys.stdin.isatty():
                log("A conta real só pode ser liberada com você presente, digitando a confirmação.")
                return 1
            print("\nVocê está prestes a operar com DINHEIRO REAL. Perdas são definitivas.")
            digitado = input(f'Para confirmar, digite exatamente "{FRASE_CONTA_REAL}": ').strip()
            cliente.usar_conta_real(digitado)
        app.operar(cfg, banco, cliente, log, conta=conta, exploracao=args.exploracao, simular=args.simular,
                   minutos=args.minutos, continuo=args.continuo)
        return 0
    finally:
        if cliente is not None:
            cliente.fechar()
        banco.fechar()


def cmd_autoteste(cfg, args, log) -> int:
    log("Autoteste com mercado sintético em memória (não usa a corretora nem o banco real).")
    return 0 if app.autoteste(log) else 1


def cmd_painel(cfg, args, log) -> int:
    from .painel import abrir

    try:
        abrir(cfg)
    except Exception as exc:  # noqa: BLE001 - o painel roda sem terminal: o erro precisa aparecer na tela
        _caixa_erro(f"O painel fechou por um erro: {type(exc).__name__}: {exc}")
        raise
    return 0


def _caixa_erro(texto: str) -> None:
    """Mostra um erro em janela (o painel é aberto sem terminal)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        raiz = tk.Tk()
        raiz.withdraw()
        messagebox.showerror("IQ-AGENT", texto)
        raiz.destroy()
    except Exception:  # noqa: BLE001
        pass


def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iq_agent", description="IQ-AGENT: pesquisa e execução para opções turbo.")
    sub = p.add_subparsers(dest="comando", required=True)
    sub.add_parser("painel", help="abre a janela de controle").set_defaults(func=cmd_painel)
    sub.add_parser("diagnostico", help="confere login, saldos, ativos e velas").set_defaults(func=cmd_diagnostico)
    sub.add_parser("diario", help="baixa dados reais, pesquisa e gera o relatório").set_defaults(func=cmd_diario)
    sub.add_parser("coletar", help="só baixa os dados reais").set_defaults(func=cmd_coletar)
    pesq = sub.add_parser("pesquisar", help="só refaz a pesquisa com os dados já baixados")
    pesq.add_argument("--ensaio", action="store_true",
                      help="roda sem gravar nada: o plano do bot e os modelos em uso não mudam")
    pesq.set_defaults(func=cmd_pesquisar)
    rel = sub.add_parser("relatorio", help="gera o relatório HTML")
    rel.add_argument("--abrir", action="store_true", help="abre no navegador")
    rel.set_defaults(func=cmd_relatorio)
    sub.add_parser("prontidao", help="mostra os critérios para a conta real").set_defaults(func=cmd_prontidao)
    op = sub.add_parser("operar", help="roda o bot (conta demo por padrão)")
    op.add_argument("--conta", choices=["demo", "real"], default="demo")
    op.add_argument("--exploracao", action="store_true", help="opera candidatos não aprovados (só demo)")
    op.add_argument("--simular", action="store_true", help="mostra o que faria, sem enviar ordens")
    op.add_argument("--minutos", type=float, default=None, help="tempo máximo de execução")
    op.add_argument("--continuo", action="store_true",
                    help="na demo, segue rodando e atualiza o mercado sozinho a cada novo dia")
    op.set_defaults(func=cmd_operar)
    sub.add_parser("autoteste", help="testa o motor com mercado sintético").set_defaults(func=cmd_autoteste)
    return p


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        if fluxo is not None and hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")
    args = montar_parser().parse_args(argv)
    try:
        cfg = carregar()
    except ErroConfig as exc:
        print(f"Erro no config.toml: {exc}")
        if args.comando == "painel":
            _caixa_erro(f"Erro no config.toml: {exc}")
        return 2
    app.preparar_logs(cfg)
    log = app.com_hora(_saida)
    try:
        return args.func(cfg, args, log)
    except ErroConfig as exc:
        log(f"Configuração: {exc}")
        return 2
    except ErroIQ as exc:
        log(f"Erro: {exc}")
        return 1
    except KeyboardInterrupt:
        log("Interrompido.")
        return 130
