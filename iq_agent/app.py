"""Operações de alto nível, usadas pelos comandos de terminal e pelo painel."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Callable

from . import prontidao
from .banco import Banco
from .coleta import coletar
from .config import Config, credenciais
from .executor import Executor
from .iqclient import IQClient, formato
from .pesquisa import pesquisar
from .relatorio import gerar
from .tipos import PRATICA, REAL, Ativo, ErroIQ

VALIDADE_PLANO_S = 36 * 3600
PLANO_VELHO_S = 20 * 3600  # acima disso, o bot atualiza o mercado sozinho antes de operar
Log = Callable[[str], None]


def preparar_logs(cfg: Config) -> None:
    """Grava o registro técnico em logs/iq_agent.log (sem senhas nem tokens)."""
    raiz = logging.getLogger("iq_agent")
    if raiz.handlers:
        return
    cfg.dir_logs.mkdir(parents=True, exist_ok=True)
    arquivo = RotatingFileHandler(cfg.dir_logs / "iq_agent.log", maxBytes=2_000_000, backupCount=3,
                                  encoding="utf-8")
    arquivo.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    raiz.addHandler(arquivo)
    raiz.setLevel(logging.INFO)


def com_hora(saida: Log) -> Log:
    """Acrescenta o horário a cada linha e copia tudo para o arquivo de log."""
    registro = logging.getLogger("iq_agent.app")

    def log(texto: str) -> None:
        registro.info(texto)
        saida(f"{datetime.now():%H:%M:%S}  {texto}")

    return log


def novo_cliente(email: str | None = None, senha: str | None = None, capturar: bool = False) -> IQClient:
    """Cria a conexão. Sem argumentos, usa as credenciais do .env preenchido pelo usuário."""
    if not email or not senha:
        email, senha = credenciais()
    return IQClient(email, senha, capturar=capturar)


def atualizar_mercado(cfg: Config, banco: Banco, cliente, log: Log) -> dict:
    """Rotina diária: baixa o histórico real, refaz a pesquisa e gera o relatório."""
    log("Baixando dados reais da corretora (ativos, payouts e velas de 1 minuto)...")
    coleta = coletar(banco, cfg, cliente, log)
    log(f"Coleta concluída: {coleta['velas_novas']} velas novas em {coleta['ativos']} ativos.")
    log("Testando as estratégias no histórico (walk-forward)...")
    resultado = pesquisar(banco, cfg, log, agora=cliente.agora())
    r = resultado["resumo"]
    log(f"Pesquisa concluída: {r['hipoteses']} hipóteses, {r['testaveis']} com dados suficientes, "
        f"{r['aprovadas']} aprovada(s).")
    caminho = gerar(banco, cfg, agora=cliente.agora())
    log(f"Relatório salvo em {caminho}")
    return resultado


def plano_para_operar(banco: Banco, agora: float, conta: str, log: Log) -> dict:
    """Devolve o plano mais recente. Para a conta real, ele precisa ser recente."""
    pesquisa = banco.ultima_pesquisa()
    if pesquisa is None:
        raise ErroIQ("Ainda não existe plano. Atualize o mercado primeiro.")
    idade = agora - pesquisa["ts"]
    if idade > VALIDADE_PLANO_S:
        if conta == "REAL":
            raise ErroIQ(f"O plano tem {idade / 3600:.0f} h. Para a conta real, atualize o mercado antes.")
        log(f"Atenção: o plano tem {idade / 3600:.0f} h. O ideal é atualizar o mercado todo dia.")
    return pesquisa["plano"]


def conferir_real(cfg: Config, banco: Banco, agora: float) -> tuple[bool, str, list[prontidao.Item]]:
    """A conta real pode ser liberada agora? Devolve (pode, explicação, critérios)."""
    if not cfg.real.permitido:
        return False, ("A conta real está desligada. Para habilitar a chave, mude 'permitido = true' na seção "
                       "[real] do config.toml."), []
    pronto, itens = prontidao.avaliar(banco, cfg, agora)
    if pronto:
        return True, "Todos os critérios de prontidão foram cumpridos.", itens
    if cfg.real.exigir_prontidao:
        return False, "Os critérios de prontidão na demo ainda não foram cumpridos.", itens
    return True, ("ATENÇÃO: os critérios de prontidão NÃO foram cumpridos. A exigência está desligada no "
                  "config.toml, então a decisão é sua."), itens


def operar(cfg: Config, banco: Banco, cliente, log: Log, conta: str = "PRACTICE", exploracao: bool = False,
           simular: bool = False, minutos: float | None = None, parar=None, continuo: bool = False) -> dict:
    """Roda o executor na conta indicada. A conta real já precisa ter sido liberada na conexão.

    Se o plano não existe ou está velho, o mercado é atualizado antes. Com `continuo`
    (só demo), o bot refaz a pesquisa sozinho a cada novo dia e segue rodando.
    """
    if cliente.conta != conta:
        raise ErroIQ(f"A conexão está na conta {cliente.conta}; para operar em {conta}, faça a liberação antes.")
    pesquisa = banco.ultima_pesquisa()
    if pesquisa is None or cliente.agora() - pesquisa["ts"] > PLANO_VELHO_S:
        log("O plano do dia não existe ou está desatualizado: atualizando o mercado antes de operar.")
        atualizar_mercado(cfg, banco, cliente, log)
    plano = plano_para_operar(banco, cliente.agora(), conta, log)

    def renovar() -> dict:
        return atualizar_mercado(cfg, banco, cliente, log)["plano"]

    executor = Executor(cfg, banco, cliente, plano, exploracao=exploracao, simular=simular, log=log, conta=conta,
                        parar=parar, renovar_plano=renovar if continuo and conta == "PRACTICE" else None)
    resumo = executor.rodar(minutos)
    try:
        gerar(banco, cfg, agora=cliente.agora())
    except Exception as exc:  # noqa: BLE001 - o relatório não pode derrubar o fim da sessão
        log(f"Não consegui atualizar o relatório: {exc}")
    return resumo


def diagnosticar(cfg: Config, cliente: IQClient, log: Log) -> bool:
    """Confere cada etapa da conversa com a corretora e salva a ESTRUTURA das mensagens
    (sem valores) em logs/diagnostico.json, para depuração."""
    ok = True
    log(f"Login e autenticação: OK. Horário do servidor: {datetime.fromtimestamp(cliente.agora()):%d/%m %H:%M:%S}")
    saldos = cliente._buscar_saldos()
    tipos = sorted({s.tipo for s in saldos})
    log(f"Saldos encontrados: {len(saldos)} (tipos {tipos}; 4 = demo, 1 = real)")
    demo = next((s for s in saldos if s.tipo == PRATICA), None)
    if demo:
        log(f"  Conta demo: {demo.valor:.2f} {demo.moeda}")
    else:
        ok = False
        log("  FALHA: não achei a conta demo.")
    log(f"  Conta real: {'existe' if any(s.tipo == REAL for s in saldos) else 'não encontrada'}")
    ativos = cliente.ativos()
    turbo = [a for a in ativos if a.tipo == "turbo"]
    abertos = sorted((a for a in turbo if a.aberto), key=lambda a: -a.payout)
    log(f"Ativos: {len(ativos)} no total, {len(turbo)} turbo, {len(abertos)} turbo abertos agora.")
    if not abertos:
        ok = False
        log("  FALHA ou mercado fechado: nenhum ativo turbo aberto.")
    for a in abertos[:8]:
        log(f"  {a.nome}: payout {a.payout:.0%}, aposta mínima {a.aposta_min}")
    if abertos:
        alvo: Ativo = abertos[0]
        velas = cliente.velas(alvo.id, 60, 5)
        if velas:
            log(f"Velas de {alvo.nome}: {len(velas)} recebidas; última abre em "
                f"{datetime.fromtimestamp(velas[-1][0]):%H:%M:%S}, fechamento {velas[-1][4]}")
        else:
            ok = False
            log(f"  FALHA: nenhuma vela recebida de {alvo.nome}.")
    cfg.dir_logs.mkdir(parents=True, exist_ok=True)
    destino = cfg.dir_logs / "diagnostico.json"
    destino.write_text(json.dumps({nome: formato(msg) for nome, msg in cliente.amostras.items()}, indent=1,
                                  ensure_ascii=False), encoding="utf-8")
    log(f"Estrutura das mensagens salva em {destino} (só nomes de campos, sem valores).")
    log("Diagnóstico: tudo certo." if ok else "Diagnóstico: há falhas acima.")
    return ok


def autoteste(log: Log = print) -> bool:
    """Confere o motor com um mercado SINTÉTICO, em memória. Não toca no banco real.

    Verifica que: (1) em mercados aleatórios nada é aprovado; (2) um padrão plantado de
    propósito é encontrado; (3) o executor opera o plano e a contabilidade fecha.
    """
    from .simulado import ClienteSimulado
    from .sintetico import gerar_velas

    fim, dias = 1_794_000_000, 22
    corte = fim - 2 * 86400
    cfg = Config()
    banco = Banco(":memory:")
    series = {f"ALEATORIO{i + 1}": gerar_velas(dias * 1440, semente=100 + i, fim_ts=fim) for i in range(4)}
    series["PLANTADO"] = gerar_velas(dias * 1440, semente=7, fim_ts=fim, reversao_apos=3, forca=0.2)
    for nome, v in series.items():
        banco.inserir_serie(nome, v.fechadas_ate(corte))
    banco.salvar_ativos([Ativo(i + 1, nome, "turbo", 0.85, True) for i, nome in enumerate(sorted(series))], corte)

    inicio = time.time()
    r = pesquisar(banco, cfg, log=lambda _: None, agora=corte)
    aprovadas = [h for h in r["hipoteses"] if h["aprovado"]]
    falsas = [h for h in aprovadas if h["ativo"] != "PLANTADO"]
    achou = [h for h in aprovadas if h["ativo"] == "PLANTADO"]
    log(f"Pesquisa: {r['resumo']['hipoteses']} hipóteses em {time.time() - inicio:.1f}s")
    ok1 = not falsas
    log(f"[{'OK' if ok1 else 'FALHA'}] mercados aleatórios: {len(falsas)} aprovação(ões) falsa(s)")
    ok2 = bool(achou)
    for h in achou:
        log(f"       achou {h['modo']} {h['estrategia']} {h['config']['params']}: {h['taxa']:.1%} em {h['n']} operações")
    log(f"[{'OK' if ok2 else 'FALHA'}] padrão plantado {'encontrado' if ok2 else 'NÃO encontrado'}")

    cliente = ClienteSimulado(series, inicio_ts=corte)
    cliente.conectar()
    saldo_inicial = cliente.saldo_demo().valor
    resumo = Executor(cfg, banco, cliente, r["plano"], log=lambda _: None).rodar(minutos=2 * 1440 - 30)
    saldo_final = cliente.saldo_demo().valor
    diario = [o for o in banco.operacoes(conta="PRACTICE") if o["status"] in ("win", "loss", "equal")]
    ok3 = (resumo["operacoes"] > 0 and len(diario) == resumo["operacoes"]
           and abs((saldo_final - saldo_inicial) - resumo["lucro"]) < 0.05)
    log(f"[{'OK' if ok3 else 'FALHA'}] executor simulado: {resumo['operacoes']} operações, "
        f"{resumo['vitorias']} ganhos, resultado {resumo['lucro']:+.2f}, contabilidade "
        f"{'fecha' if ok3 else 'NÃO fecha'}")
    banco.fechar()
    return ok1 and ok2 and ok3
