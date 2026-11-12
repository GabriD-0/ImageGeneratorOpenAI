# ImageGeneratorOpenAI

Gera a arte do projeto (a ficha do cavaleiro em `prompt.txt`, copiada do ImageGenerator) pela **API de imagens da OpenAI** e ajuda a
descobrir **qual combinação de modelo e qualidade tem o melhor custo-benefício**, medindo o custo real de cada chamada.

Só biblioteca padrão do Python 3.10+ (testado no 3.12). O Pillow (`pip install -r requirements.txt`) é opcional: com ele, a comparação também confere se o
fundo chroma key `#00B140` saiu certo.

## Primeiros passos

```bash
cp .env.example .env            # coloque a OPENAI_API_KEY (ou exporte a variável no shell)

python -m openai_images estimate            # quanto custa cada combinação (não chama a API)
python -m openai_images generate --dry-run  # mostra o pedido, o prompt final e o custo
python -m openai_images compare --dry-run   # mostra o plano da comparação e o custo estimado
```

Só os comandos `generate` e `compare` (sem `--dry-run`) chamam a API e **geram cobrança**. Ambos têm trava de gasto (`--max-cost`, `--budget`).
O guia da OpenAI avisa que pode ser exigida a verificação da organização ("API Organization Verification", no painel de desenvolvedor) antes de usar os modelos GPT Image.

## Como descobrir o melhor custo-benefício

O custo é medível, mas a qualidade da arte é uma opinião. O fluxo separa as duas coisas:

```bash
python -m openai_images compare --yes                 # 7 candidatas, 1 imagem cada (~US$ 0,24 estimado)
#   abra runs/<data>/sheet.html e olhe as imagens lado a lado, com custo, tempo e checagens
#   dê notas de 1 a 5 em runs/<data>/ratings.csv
python -m openai_images rank runs/<data>              # ranking
```

Regra do `rank`: vence a combinação **mais barata** cuja nota média é >= 4 (`--min-rating`) e cuja altura atende o mínimo (1600 px). Também
mostra quanto custa subir para a de maior nota. Nota por dólar aparece só como referência, porque premiaria o que é barato e ruim.
Use `--samples 3` para ver a consistência de cada combinação (a mesma configuração varia de uma imagem para outra).

Checagens objetivas feitas em cada imagem: altura >= 1600 px e fração da borda que é o verde chroma key. Elas só avisam; quem decide é a nota.

## Comandos

| Comando | O que faz |
|---|---|
| `estimate` | Tabela de custo por modelo, qualidade e tamanho. `--size` (repetível), `--models`, `--all` (inclui o que não atende a altura mínima). |
| `generate` | Uma chamada. `--model`, `--quality`, `--size`, `--n`, `--format`, `--background`, `--complexity` (só 2.6), `--max-cost`, `--dry-run`. Grava a imagem e um `.json` com `usage`, custo real, tempo e `request-id`. |
| `compare` | Várias combinações. `--configs modelo:qualidade,...`, `--samples`, `--budget`, `--yes`, `--dry-run`. Grava imagens, `results.json`, `sheet.html` e `ratings.csv` em `runs/<data>/`. |
| `rank` | Lê as notas e escreve `ranking.md`. |
| `batch submit` | Envia N imagens de uma combinação pela **Batch API** (metade do preço, resultado em até 24 h). `--model`, `--quality`, `--size`, `--n`, `--max-cost`, `--yes`, `--dry-run`. Grava `requests.jsonl` e `batch.json` em `batches/<data>/`. |
| `batch status` | Andamento do batch (`batches/<data>`). |
| `batch fetch` | Baixa as imagens, roda as checagens e calcula o custo real (já com o desconto) em `fetch.json`. |

## Batch API (metade do preço)

Mesma geração, mas assíncrona: você envia um arquivo com os pedidos, a OpenAI processa em segundo plano (até 24 h, muitas vezes bem antes) e você
baixa o resultado depois. O `gpt-image-2.5-sunburst` e o `-flare` aceitam; o preço do Batch é metade do normal (conferido na página de preços).

```bash
# 1. veja o que seria enviado e quanto custa (não precisa de chave, não envia nada)
python -m openai_images batch submit --model gpt-image-2.5-sunburst --quality xhigh --n 4 --dry-run

# 2. envie (cada imagem vira um pedido; US$ 0,039 cada no sunburst xhigh 1088x1664, contra US$ 0,078 no normal)
python -m openai_images batch submit --model gpt-image-2.5-sunburst --quality xhigh --n 4 --yes

# 3. acompanhe e, quando estiver "completed", baixe
python -m openai_images batch status batches/<data>
python -m openai_images batch fetch  batches/<data>
```

- **Você paga pelos pedidos concluídos** (num batch expirado, só por eles). O `--max-cost` (padrão US$ 0,50) recusa o envio se a estimativa do batch passar disso.
- **Um pedido por imagem** (`n=1` em cada linha): o `custom_id` identifica cada resultado, porque a ordem do arquivo de saída não segue a do de entrada.
- **`fetch` pode ser repetido** e funciona também com batch expirado (só os pedidos concluídos são cobrados; os demais aparecem como "sem resultado").
  Linhas do resultado com `custom_id` que não saiu deste envio são ignoradas com aviso.
- **Os resultados ficam 30 dias** no servidor da OpenAI depois que o batch termina.
- Não existe Batch para o `gpt-image-2.6-*` neste projeto (`batch submit` recusa).
- O Batch é o oposto de "tempo real": serve para produzir a arte final barato, depois de escolher a combinação com o `compare`.

## Análise de custo (estimada, antes de gastar)

Por imagem de **1088x1664** (altura > 1600 px, a mesma resolução do ImageGenerator), incluindo ~310 tokens de prompt. Fórmula da calculadora oficial
da OpenAI; rode `estimate` para a tabela completa.

| Qualidade | gpt-image-2.5 (flare ou sunburst) | gpt-image-2 | gpt-image-2.6 |
|---|---|---|---|
| low | US$ 0,006 | US$ 0,006 | US$ 0,021 |
| medium | US$ 0,0125 | US$ 0,044 | US$ 0,038 |
| high | US$ 0,044 | US$ 0,174 | US$ 0,044 |
| xhigh | US$ 0,078 | n/d | US$ 0,078 |
| max | US$ 0,174 | n/d | US$ 0,174 |

O que isso já sugere (a confirmar com `compare`, porque qualidade não vem da tabela):

- **Flare e Sunburst custam o mesmo.** A diferença é qualidade e velocidade (a OpenAI descreve o Flare como o mais rápido e o Sunburst como o mais
  capaz), então escolher entre eles é uma questão de olhar o resultado.
- **O `gpt-image-2` sai mais caro que o 2.5 para o mesmo rótulo de qualidade.** O `medium` dele custa o mesmo que o `high` do 2.5, e o `high` custa o mesmo que
  o `max`. Só compensa se a imagem for melhor; o plano padrão inclui `gpt-image-2:medium` justamente para comparar com `2.5:high` ao mesmo preço.
- **O 2.6 é mais caro em `low` e `medium`** (a calculadora multiplica os tokens por 4,2 e 3,3) e igual em `high` ou mais. Ele não consta na referência da
  Images API, só no guia, na página de modelos e na de preços; por isso fica fora do plano padrão. Para testá-lo:
  `--configs gpt-image-2.6-flare:high,gpt-image-2.5-flare:high`.
- **`gpt-image-1.5` e `gpt-image-1-mini` não atendem ao requisito**: só geram 1024x1024, 1024x1536 e 1536x1024 (máximo de 1536 px, abaixo dos 1600 do projeto).
  O `gpt-image-1` foi desligado em 23/10/2026.
- **A Batch API custa metade** (ver a coluna `US$ batch` do `estimate`; não existe para o 2.6). Para arte que não precisa ficar pronta na hora, é o
  maior corte de custo disponível; use `batch submit` (seção "Batch API" acima).
- Em tamanhos livres o custo **não é monotônico**: 1088x1664 sai um pouco mais barato que 1088x1632 em `low` e `high`, por arredondamento da fórmula.

## Decisões de projeto

- **Negative prompt:** a API de imagens não tem esse campo. Por padrão (`--negative append`) o bloco `---negative---` do `prompt.txt` vai no fim do prompt como
  `Avoid: ...`; `--negative ignore` o descarta. O `--dry-run` mostra o texto exato enviado.
- **Fundo:** o prompt pede chroma key verde. Os modelos 2.5/2.6 também aceitam `--background transparent` (PNG ou WebP), que dispensaria o chroma key; teste
  com `generate --background transparent` e retire o trecho do verde do prompt.
- **Cliente HTTP próprio, sem o SDK:** o corpo é um JSON simples, e parâmetros novos da referência (`quality="xhigh"`, `complexity`) são enviados como estão,
  sem depender da versão do SDK. Repete só 429 (exceto falta de saldo) e 5xx; **não repete timeout de leitura**, porque a imagem pode já ter sido gerada e cobrada.
- **Custo real:** vem do `usage` da resposta (tokens de texto, de imagem e de saída x preço por token). A estimativa serve só para planejar e travar gastos.

## Estrutura

```text
prompt.txt                 <- cópia do prompt do ImageGenerator
openai_images/
  pricing.py               <- preços por token, fórmula de tokens, validação de tamanho, custo a partir do usage
  client.py                <- POST /v1/images/generations, Files e Batches (urllib), repetição segura, erros legíveis
  checks.py                <- altura mínima e fundo chroma key
  compare.py               <- executa a matriz de combinações com trava de orçamento
  batch.py                 <- Batch API: submit, status e fetch (pedidos .jsonl, custo real com desconto)
  report.py                <- tabelas, sheet.html, ratings.csv e ranking
  cli.py  __main__.py      <- estimate | generate | compare | rank | batch
tests/                     <- 59 testes, com um servidor falso da Images API e da Batch API (python -m unittest discover -s tests -t .)
outputs/  runs/  batches/  <- imagens, comparações e batches (criados sob demanda)
```

## O que foi verificado (e o que não foi)

Verificado:

- A fórmula de tokens reproduz o exemplo da própria calculadora da OpenAI (2.6, low, 1024²: 823 tokens, US$ 0,02469) e a tabela oficial de preço por imagem do
  `gpt-image-2` nos três tamanhos (testes em `tests/test_pricing.py`).
- O cliente, a comparação e o ranking funcionam de ponta a ponta contra um **servidor local que imita a API** (sucesso, 429 com repetição, falta de saldo, 400, 401,
  timeout sem repetição, orçamento, fundo fora do chroma key). Uma checagem de mutação confirmou que os testes de repetição e de timeout falham quando a regra é quebrada.

**Não verificado: nenhuma chamada real foi feita à API da OpenAI** (não há chave nesta sessão, e cada chamada cobra). Portanto:

- Os custos acima são **estimativas**. O `usage` real pode diferir e o `compare` mostra estimado e real lado a lado.
- Não sei se a API aceita `1088x1664` em todos os modelos listados, se o 2.6 está liberado na sua conta, nem se há exigência de verificação da organização para você.
  Os limites de tamanho vêm da calculadora e da referência (múltiplos de 16, 655.360 a 8.294.400 pixels, lado maior até 3840, proporção até 3:1).
- A qualidade relativa dos modelos (Flare x Sunburst, 2.5 x gpt-image-2) não está medida: é o que o `compare` e as suas notas respondem.
- **Batch:** o formato dos pedidos, os endpoints e o desconto de 50% seguem a documentação da OpenAI (guia do Batch e página de preços), e o fluxo
  `submit → status → fetch` foi testado contra o servidor falso, que é escrito por nós. Não sei se o upload multipart, o `quality="xhigh"` ou o tamanho
  `1088x1664` são aceitos pelo Batch real. Teste primeiro com `batch submit --quality low --n 1` (≈ US$ 0,003) antes de gastar no `xhigh`.
- O preço de saída de `chatgpt-image-latest` e o do `gpt-image-1` (desligado) não foram incluídos.

## Fontes (consultadas em 04/11/2026)

- Referência da Images API: https://developers.openai.com/api/reference/resources/images
- Guia de geração de imagens (inclui a calculadora de tokens): https://developers.openai.com/api/docs/guides/image-generation
- Preços: https://developers.openai.com/api/docs/pricing
- Modelos: https://developers.openai.com/api/docs/models
