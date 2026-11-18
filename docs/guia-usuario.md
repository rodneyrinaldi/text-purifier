# Guia simples para uso (não técnico)

Este guia é para usar o sistema sem precisar entender programação.

## O que este sistema faz

Você envia um arquivo PDF e o sistema devolve um arquivo `.md` (Markdown) com o texto:

- limpo (remove ruídos comuns de cabeçalho/rodapé)
- com leitura de todas as páginas, inclusive PDFs protegidos, digitalizados e manuscritos
- com a saúde (qualidade) da leitura de cada página
- com relatório de processamento ao final
- com opção de anonimização e de uso da IA local

## Antes de começar

Escolha uma forma de uso:

- Forma 1: rodar com Python local
- Forma 2: rodar com Docker (mais simples para evitar ajustes de ambiente)

## Forma 1: usar com Python local

1. Abra o PowerShell na pasta do projeto.
2. Rode o comando abaixo uma única vez (ou quando atualizar dependências):

```powershell
.\scripts\setup_python.ps1
```

3. Inicie o sistema:

```powershell
.\scripts\run_local.ps1
```

4. No navegador, abra:

- http://localhost:8000

## Forma 2: usar com Docker

1. Abra o PowerShell na pasta do projeto.
2. Suba o sistema em container:

```powershell
.\scripts\docker_up.ps1
```

3. No navegador, abra:

- http://localhost:8000

4. Para parar o sistema:

```powershell
.\scripts\docker_down.ps1
```

5. Para parar e também remover a imagem Docker:

```powershell
.\scripts\docker_down.ps1 -RemoveImage
```

## Como usar a tela

Na página principal (no topo aparece se o OCR e a IA local estão disponíveis):

1. Selecione o PDF (pode ser protegido; se pedir senha, preencha o campo **Senha do PDF**).
2. Opções:
   - **Remover cabeçalhos/rodapés repetidos** (marcado por padrão; a primeira ocorrência é mantida).
   - **Anonimizar dados pessoais**.
   - **IA: transcrever páginas com leitura comprometida** — usa a IA local para ler manuscritos e imagens.
   - **IA: revisar erros de OCR** — corrige letras trocadas sem alterar números.
3. Informe `Página inicial` e `Página final` apenas se quiser limitar o trecho.
4. Opcional: informe seu nome em **Operador** (aparece no relatório).
5. Clique em `Processar` e acompanhe o painel de status: etapa atual, tempo decorrido, tempo restante e horário previsto de término (a estimativa é refinada a cada página; aparece como "preliminar" até se saber quantas páginas irão para a IA), lista de etapas, o mapa de saúde das páginas já lidas e, durante o uso da IA, a página em andamento e a velocidade de escrita.

Resultado:

- o texto aparece na tela e um arquivo `.md` é baixado automaticamente
- um resumo mostra a saúde de cada página (verde = boa, amarelo = média, laranja = baixa, vermelho = crítica)

## Como ler o resultado

Cada página começa assim:

`[PAGINA_AUTOS:00045 ou PAGINA_FISICA:00002 | LEITURA:94% (ALTA)]`

- `PAGINA_FISICA`: posição da página no arquivo PDF.
- `PAGINA_AUTOS`: número da folha nos autos ("fls."), quando encontrado.
- `LEITURA`: saúde da leitura da página. Abaixo de 70% a página é marcada como **leitura comprometida**
  e deve ser conferida no original.

No final do arquivo há a seção **INFORMAÇÕES SOBRE O PROCESSAMENTO**: quem processou, quando, como, as
páginas que precisam de conferência, erros, possíveis folhas ausentes e a identificação (SHA-256) do PDF.

## Dicas rápidas

- Páginas manuscritas são lidas por OCR; para melhor resultado, marque a opção de IA de visão.
- Texto marcado como **Transcrição assistida por IA** deve sempre ser conferido com o original.
- Se deixar os campos de página em branco, o sistema processa do início ao fim automaticamente.

## Problemas comuns

- A página não abre em `http://localhost:8000`:
  verifique se o comando de inicialização ainda está rodando no terminal.
- Erro de Docker:
  abra o Docker Desktop e tente novamente.
- Erro de Python:
  confirme que Python está instalado e rode `setup_python.ps1` novamente.

## Alternativa para Prompt de Comando (.bat)

Se preferir `.bat` em vez de `.ps1`:

```bat
scripts\setup_python.bat
scripts\run_local.bat
scripts\docker_up.bat
scripts\docker_down.bat
```