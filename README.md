# text-purifier 2.0

Serviço local que extrai **todo** o texto de processos e documentos jurídicos em PDF e devolve Markdown
limpo, com o **índice de saúde da leitura de cada página** e um **relatório de processamento** ao final.

## Guia para usuário não técnico

- Veja `docs/guia-usuario.md`

## Como garante que todas as folhas são lidas

Cada página passa por várias leituras independentes que depois são comparadas:

| Etapa | O que faz |
|---|---|
| Abertura robusta | Aceita PDF com restrições (cópia/impressão bloqueadas), PDF com senha (campo *Senha*), PDF danificado (reparo automático com qpdf). Cria cópia de trabalho descriptografada quando necessário. Confere o número de páginas em 4 motores. |
| Camada de texto | Lida por **3 motores** (PyMuPDF, PDFium, pypdf); o melhor resultado é escolhido. Detecta fonte sem mapeamento Unicode (texto "cifrado" de PDFs protegidos). |
| Renderização + OCR | **Toda página** é renderizada (300–600 DPI, conforme a digitalização) e passa por OCR Tesseract com várias tentativas: orientação (páginas giradas), binarização, correção de inclinação, realce de contraste (escrita fraca/manuscrita). Se o renderizador principal falhar, usa o PDFium. |
| Validação cruzada | Compara texto nativo × OCR. Camada de texto invisível/divergente do conteúdo visual é descartada; trechos em imagem dentro de páginas digitais são acrescentados. |
| Cobertura de tinta | Mede quanto da "tinta" da página virou texto — aponta manuscritos, carimbos e imagens não lidos. |
| IA local (opcional) | Páginas com saúde baixa são enviadas a um **modelo de visão** do Ollama (ex.: `qwen2.5vl:7b`) para transcrição de manuscritos/imagens, com verificações contra alucinação. Um **modelo de texto** (ex.: `qwen2.5:14b`) pode revisar erros de OCR, com salvaguardas (não pode alterar números, tamanho ou marcadores). |
| Extras | Valores de campos de formulário e anotações/comentários do PDF são incluídos. Páginas em branco são confirmadas por renderização e OCR. |

## Saída

Cada página começa com a etiqueta de origem e a saúde da leitura:

```markdown
[PAGINA_AUTOS:00045 ou PAGINA_FISICA:00002 | LEITURA:94% (ALTA)]
```

Faixas: **ALTA** ≥ 90% · **MÉDIA** ≥ 70% · **BAIXA** ≥ 50% · **CRÍTICA** < 50% (limites configuráveis).
A saúde combina: confiança do motor de leitura (40%), qualidade linguística (25%), cobertura da tinta (20%) e
concordância entre leituras independentes (15%). Transcrições por IA são limitadas a 85% e sempre sinalizadas.

Ao final do documento, a seção **INFORMAÇÕES SOBRE O PROCESSAMENTO** traz:

- **Resumo**: saúde média/mínima, distribuição, páginas por método (texto, OCR, IA, em branco).
- **Quem e quando**: operador informado, usuário/máquina, IP de origem, início, término, duração, ID.
- **Documento de origem**: tamanho, **SHA-256**, versão, metadados, criptografia e restrições contornadas,
  senha, reparo, campos de assinatura digital, formulários, contagem de páginas por motor.
- **Como foi processado**: método, versões do Tesseract e das bibliotecas, DPI, idiomas, IA usada e modelos,
  anonimização, cabeçalhos/rodapés removidos (com exemplos).
- **Leituras comprometidas**: cada página abaixo do limite, com o motivo (manuscrito, baixa confiança, etc.).
- **Uso da IA por página**, **divergências** (ex.: saltos na numeração de folhas = possíveis folhas ausentes,
  anexos embutidos não processados) e **erros técnicos**.
- **Detalhamento por página** (tabela) e **SHA-256 do conteúdo extraído**.

## Requisitos

- Docker (recomendado — já inclui Tesseract com português), **ou**
- Python 3.11+ e [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) com o idioma português
  (no Windows, instale em `C:\Program Files\Tesseract-OCR` ou defina `TESSERACT_CMD`).

## Configuração (`.env`)

Copie `.env.example` para `.env`. Principais variáveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `LLM_URL` | `http://192.168.15.6:11434` | Servidor Ollama |
| `LLM_TEXT_MODEL` | automático (prefere ~14b) | Modelo para revisão de OCR (`LLM_MODEL` também é aceito) |
| `LLM_VISION_MODEL` | automático (prefere ~7b com visão) | Modelo para transcrever páginas manuscritas/imagem |
| `LLM_VISION_THRESHOLD` | `75` | Páginas com saúde abaixo disso vão para a IA de visão |
| `LLM_VISION_MAX_PAGES` | `0` | Limite de páginas por documento enviadas à IA (0 = sem limite) |
| `ALWAYS_OCR` | `true` | OCR em todas as páginas (validação cruzada). `false` = mais rápido |
| `OCR_DPI` / `OCR_MAX_DPI` | `300` / `600` | Resolução do OCR |
| `HEALTH_WARN` / `HEALTH_CRITICAL` | `70` / `50` | Limites de leitura comprometida/crítica |
| `REMOVE_REPEATED_HEADERS` | `true` | Padrão da opção de remover cabeçalhos/rodapés |
| `WORKERS` | nº de CPUs | Processos paralelos |

### IA no servidor 192.168.15.6

O Ollama precisa aceitar conexões da rede. No computador 192.168.15.6:

1. Defina a variável de ambiente `OLLAMA_HOST=0.0.0.0` e reinicie o Ollama.
2. Libere a porta TCP **11434** no firewall.
3. Para manuscritos, instale um modelo com visão: `ollama pull qwen2.5vl:7b`.

Confira em `http://localhost:8000/status` (mostra Tesseract, modelos encontrados e os escolhidos).

## Uso

Docker:

```powershell
.\scripts\docker_up.ps1        # build + run em http://localhost:8000
.\scripts\docker_down.ps1      # parar (-RemoveImage para remover a imagem)
```

Python local:

```powershell
.\scripts\setup_python.ps1
.\scripts\run_local.ps1
```

Linha de comando (lote, sem servidor):

```powershell
python -m purifier.cli processo.pdf -o saida.md --vision --operador "Nome"
python -m purifier.cli pasta\*.pdf -o resultados\ --anon
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Interface web |
| GET | `/status` | Diagnóstico (Tesseract, IA, bibliotecas) |
| POST | `/jobs` | Inicia processamento assíncrono → `{"job_id"}` |
| GET | `/jobs/{id}` | Progresso (etapas, páginas lidas, IA em tempo real, tempo restante e término previsto) e resumo de saúde |
| GET | `/jobs/{id}/result` | Markdown final |
| POST | `/process` | Processamento síncrono (compatível com a versão anterior) |

Campos `multipart/form-data` de `/jobs` e `/process`: `file` (obrigatório), `anon`, `llm` (revisão de OCR por IA),
`llm_vision` (transcrição por IA de visão), `remove_repeated`, `page_start`, `page_end`, `password`, `operador`.

O nome do arquivo gerado segue o padrão `nome-do-arquivode00001a00013.md`.

## Estrutura

```
main.py                 API FastAPI (jobs, progresso)
purifier/pdfsource.py   abertura, senha, restrições, reparo, inventário
purifier/extract.py     leitura de uma página (3 motores + render + OCR + saúde)
purifier/ocr.py         Tesseract com múltiplas tentativas
purifier/llm.py         Ollama: descoberta de modelos, visão, correção
purifier/postprocess.py cabeçalhos/rodapés repetidos, numeração de folhas
purifier/report.py      Markdown e relatório de processamento
purifier/pipeline.py    orquestração
purifier/cli.py         linha de comando
```
