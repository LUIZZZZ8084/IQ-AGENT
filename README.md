# IQ-AGENT

Bot de day trade para opções turbo (1 a 5 minutos) da IQ Option. Ele baixa o histórico
real da corretora, testa estratégias com rigor estatístico, monta o plano do dia e opera
sozinho, primeiro na conta demo e, quando você liberar, na conta real.

## Leia antes de usar

- **Opções binárias têm expectativa negativa.** Com payout de 85% é preciso acertar mais
  de 54% só para empatar. A maioria das estratégias conhecidas não passa disso quando
  testada direito. Na maior parte dos dias o resultado honesto da pesquisa é "não operar".
- **Demo não é real.** Resultado na demo não garante resultado com dinheiro de verdade.
- **API não oficial.** A IQ Option não publica API. O bot fala o mesmo protocolo do
  aplicativo: pode parar de funcionar se a corretora mudar algo, e o uso de automação
  pode violar os termos de uso e levar ao bloqueio da conta.
- Nada aqui é recomendação de investimento. O risco é de quem opera.

## Primeiros passos

1. Dê dois cliques em `instalar.bat` (cria o ambiente Python; só na primeira vez).
2. Dê dois cliques em `IQ-AGENT.bat`. Abre o painel de controle.
3. Clique em **Diagnóstico**. O painel pede o e-mail e a senha da IQ Option (ficam só na
   memória, até você fechar a janela) e confere login, saldos, ativos e velas.
4. Clique em **1. Atualizar mercado**. Na primeira vez leva cerca de 45 minutos: ele baixa
   30 dias de velas reais de 1 minuto de todos os ativos turbo abertos (cerca de 100) e
   roda a pesquisa. Nos dias seguintes baixa só o que falta e leva poucos minutos. O
   banco de dados ocupa em torno de 350 MB.
5. Clique em **Abrir relatório** para ver o plano do dia.
6. Clique em **2. Iniciar bot**. Ele opera na conta demo até você clicar em **Parar bot**.

As opções abaixo dos botões valem no momento da partida:

- **Exploração** (vem marcada): nos dias em que a pesquisa não aprova nenhuma estratégia,
  é ela que faz o bot operar na demo, com valor mínimo, nas melhores candidatas. Serve
  para treinar e conferir a execução; essas candidatas não têm vantagem comprovada.
- **Contínuo** (vem marcada): o bot fica rodando e refaz a pesquisa a cada novo dia.
- **Só simular** (vem desmarcada): com ela marcada o bot **não envia ordens**, só mostra
  o que faria. A faixa do topo fica cinza e o botão vira "Iniciar simulação".

Se preferir não digitar o login toda vez, copie `.env.example` para `.env` e preencha.
O `.env` nunca vai para o GitHub.

## Como funciona

| Etapa | O que faz |
|---|---|
| Coleta | Baixa da corretora a lista de ativos, os payouts e as velas reais de 1 minuto. |
| Pesquisa | Testa cada estratégia em cada ativo com walk-forward: a configuração é escolhida só com dados antigos e julgada só em dados que ela não viu. |
| Filtro estatístico | Uma hipótese só é aprovada se a taxa de acerto fora da amostra superar o breakeven com 95% de confiança e sobreviver à correção para múltiplos testes (Benjamini-Hochberg). |
| Plano do dia | Lista as estratégias aprovadas e, à parte, candidatas de exploração (sem vantagem comprovada, só demo). |
| Executor | A cada fechamento de vela lê os dados ao vivo, calcula o sinal com a mesma função do backtest, abre a operação, espera o resultado e anota no diário. |
| Relatório | Página HTML com o plano, o ranking, a leitura do cenário atual de cada ativo e o desempenho ao vivo comparado ao backtest. |

Todo o treino usa dados reais de mercado. O mercado sintético que existe no código serve
só para o autoteste do motor e nunca entra no banco de dados.

### Estratégias testadas

RSI, Bandas de Bollinger, cruzamento com média a favor da tendência, sequência de velas,
MHI, engolfo, estocástico, vela de força, MACD e um detector de padrões de cores das
últimas 3 ou 4 velas (a direção de cada padrão é aprendida do histórico). Cada uma é
combinada com filtros de horário (Ásia, Londres, Nova York, tarde/noite) e de
volatilidade (alta ou baixa).

Modos: velas de 1 min com expiração de 1 ou 5 min, e velas de 5 min com expiração de 5 min.

### Quanto investir

O valor de cada operação é calculado por regra, nunca "no olho":

- Kelly fracionado (1/4), usando a taxa de acerto **pessimista** (limite inferior do intervalo).
- Teto de 1% do saldo por operação (`risco_por_operacao`).
- Parada do dia ao perder 5% do saldo, ao bater 20 operações, ou pausa de 30 minutos
  depois de 3 perdas seguidas.
- **Sem martingale.** O valor nunca aumenta depois de uma perda.

Tudo isso é ajustável em `config.toml`.

## Conta demo e conta real

O bot **sempre começa na demo**. Para usar a conta real são necessárias três coisas:

1. Em `config.toml`, seção `[real]`, mudar para `permitido = true`.
2. Cumprir os critérios de prontidão na demo (botão **Ver critérios para a conta real**):
   por padrão, 100 operações de estratégias aprovadas em 10 dias diferentes, taxa de
   acerto acima do breakeven com confiança, resultado positivo e plano atual com
   estratégia aprovada. Para dispensar essa exigência por sua conta e risco, mude
   `exigir_prontidao = false`.
3. No painel, escolher **Real** e digitar a frase de confirmação. Isso vale só até
   fechar o painel.

Na conta real o bot só opera estratégias aprovadas (nunca exploração), respeita o teto
`valor_maximo` da seção `[real]` e para sozinho à meia-noite: cada dia exige uma nova
partida manual.

## Atualização diária

- **Pelo painel:** com a opção **Contínuo** marcada, o bot fica rodando na demo e refaz
  a coleta e a pesquisa sozinho a cada novo dia. Se o plano tiver mais de 20 horas ao
  iniciar, ele também atualiza antes de operar.
- **Pelo Agendador de Tarefas do Windows** (opcional, exige o `.env` preenchido):

```bash
schtasks /Create /TN "IQ-AGENT diario" /TR "\"%USERPROFILE%\IQ-AGENT\iq.bat\" diario" /SC DAILY /ST 08:00
```

## Comandos de terminal

Use `iq.bat` na pasta do projeto:

| Comando | O que faz |
|---|---|
| `iq painel` | Abre o painel. |
| `iq diagnostico` | Confere login, saldos, ativos e velas. |
| `iq diario` | Coleta dados reais, refaz a pesquisa e gera o relatório. |
| `iq relatorio --abrir` | Gera e abre o relatório. |
| `iq operar` | Roda o bot na demo. Opções: `--exploracao`, `--simular`, `--minutos N`, `--continuo`. |
| `iq operar --conta real` | Roda na conta real (pede a frase de confirmação). |
| `iq prontidao` | Mostra os critérios para a conta real. |
| `iq autoteste` | Testa o motor com mercado sintético, sem tocar na corretora. |

## Arquivos

- `config.toml`: todas as regras (coleta, pesquisa, risco, operação, conta real).
- `data/iq_agent.db`: velas, pesquisas e diário de operações (fica só no seu computador).
- `reports/relatorio.html`: último relatório.
- `logs/`: registro técnico. Não guarda senha nem token.
- `iq_agent/`: código. `tests/`: testes automáticos (`.venv\Scripts\python -m pytest`).

## Limitações conhecidas

- A conversa com a corretora foi escrita a partir do protocolo conhecido e validada só
  até onde dá sem login. **Rode o Diagnóstico primeiro**; se algo falhar, o arquivo
  `logs/diagnostico.json` mostra a estrutura das mensagens recebidas (sem valores).
- Só opções turbo (1 a 5 minutos). Digitais, blitz e forex/CFD não são suportados.
- Contas com verificação em duas etapas ainda não são suportadas.
- Uma operação por vez. Não opere manualmente na mesma conta enquanto o bot roda: a
  conferência do resultado usa o saldo.
- Se a corretora recusar uma ordem ou não confirmar a abertura, o bot anota o motivo,
  deixa o ativo 30 minutos sem novas tentativas e, depois de 5 problemas seguidos, para.
