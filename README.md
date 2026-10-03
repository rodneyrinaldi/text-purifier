# text-purifier

Serviço local para processar PDFs e retornar texto limpo em Markdown.

## Guia para usuário não técnico

- Veja `docs/guia-usuario.md`

## O que o projeto faz

- recebe um PDF
- limpa ruído (incluindo padrões de cabeçalho e rodapé repetidos)
- extrai texto com Docling
- tenta OCR quando o texto extraído é insuficiente
- insere aviso em Markdown por página quando o OCR não consegue converter a imagem
- permite anonimização opcional (`anon`)
- permite melhoria opcional com IA local via HTTP (`llm`)
- permite filtrar por página inicial e final
- gera arquivo `.md` com nome no padrão: `nome-do-arquivo00001a00013.md`
- inclui tags de rastreabilidade por página original do PDF
- retorna sempre Markdown

## Requisitos

- Python 3.11+
- pip
- Docker (opcional)

## Configuração

Crie o arquivo `.env` a partir do exemplo:

```bash
copy .env.example .env
```

No `.env`:

- `LLM_URL`: endpoint HTTP da IA local
- `LLM_MODEL`: nome do modelo
- `MIN_NATIVE_TEXT_CHARS` (opcional): mínimo de caracteres para confiar no texto nativo do PDF (padrão `25`)
- `MIN_EXTRACTED_TEXT_CHARS` (opcional): mínimo de caracteres para aceitar a extração sem OCR (padrão `80`)
- `MIN_ALPHA_RATIO` (opcional): proporção mínima de letras no texto para considerar que não é ruído (padrão `0.30`)

Exemplo:

```env
LLM_URL=http://localhost:11434/api/generate
LLM_MODEL=llama3.1
```

## Scripts (PowerShell)

Todos os comandos abaixo devem ser executados na raiz do projeto.

Alternativa no Windows: você também pode usar os arquivos `.bat` equivalentes na pasta `scripts`.

1. Instalar dependências no Python atual:

```powershell
.\scripts\setup_python.ps1
```

```bat
scripts\setup_python.bat
```

2. Rodar a API localmente:

```powershell
.\scripts\run_local.ps1
```

```bat
scripts\run_local.bat
```

3. Subir container Docker (build + run):

```powershell
.\scripts\docker_up.ps1
```

```bat
scripts\docker_up.bat
```

4. Derrubar container Docker:

```powershell
.\scripts\docker_down.ps1
```

```bat
scripts\docker_down.bat
```

5. Derrubar container e remover imagem:

```powershell
.\scripts\docker_down.ps1 -RemoveImage
```

```bat
scripts\docker_down.bat -RemoveImage
```

## Execução local

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Abra no navegador:

- http://localhost:8000

## Uso pela interface

1. Envie um arquivo PDF.
2. Marque `anon` para anonimização opcional.
3. Marque `llm` para melhoria opcional com IA local.
4. Informe página inicial e página final somente se quiser limitar o intervalo.
5. Clique em **Processar**.
6. O resultado em Markdown aparece na tela e é baixado automaticamente.

Se os campos de página ficarem em branco, o sistema processa do início ao fim do PDF.

## Endpoint HTTP

### GET /

Retorna a interface HTML.

### POST /process

`multipart/form-data` com campos:

- `file`: PDF (obrigatório)
- `anon`: `true/false` (opcional)
- `llm`: `true/false` (opcional)
- `page_start`: número da página inicial (opcional)
- `page_end`: número da página final (opcional)

Resposta:

- `text/markdown`
- header `Content-Disposition` com nome no padrão: `nome-do-arquivo00001a00013.md`

Cada página processada gera um bloco precedido por uma tag de origem clara para IA:

```markdown
[ORIGEM_PAGINA_PDF:00005]
```

Quando uma página não puder ser convertida em texto por OCR, o bloco dessa página inclui o aviso:

```markdown
**[Aviso: não foi possível converter a imagem desta página em texto.]**
```

## Execução com Docker

1. Build da imagem:

```bash
docker build -t text-purifier .
```

2. Run do container:

```bash
docker run -d --name text-purifier -p 8000:8000 --env-file .env text-purifier
```

Depois da primeira execução, para voltar a usar sem recriar, basta iniciar o container existente:

```bash
docker start text-purifier
```

Acesse:

- http://localhost:8000

## Observações

- Quando o OCR falha em uma página, o sistema mantém o fluxo e inclui no Markdown:

```markdown
**[Aviso: não foi possível converter a imagem desta página em texto.]**
```

- Se `llm` estiver ativo e a chamada HTTP falhar, o texto original é retornado.
- Se a anonimização por modelo spaCy não estiver disponível, o sistema mantém máscaras básicas por regex.
- Em ambiente de container dedicado, não é necessário usar ambiente virtual.







# text-purifier

Procedimento para refatorar o ambiente.

docker rm -f text-purifier
docker rmi -f text-purifier
docker build --no-cache -t text-purifier .
docker run -d --name text-purifier -p 8000:8000 --env-file .env text-purifier


