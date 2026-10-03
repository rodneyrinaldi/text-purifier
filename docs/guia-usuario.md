# Guia simples para uso (não técnico)

Este guia é para usar o sistema sem precisar entender programação.

## O que este sistema faz

Você envia um arquivo PDF e o sistema devolve um arquivo `.md` (Markdown) com o texto:

- limpo (remove ruídos comuns de cabeçalho/rodapé)
- com tentativa de OCR quando necessário
- com opção de anonimização (`anon`)
- com opção de melhoria por IA local (`llm`)

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

Na página principal:

1. Selecione o PDF.
2. Opcional: marque `anon` para tentar ocultar dados pessoais.
3. Opcional: marque `llm` para tentar melhorar o texto usando IA local.
4. Informe `Página inicial` e `Página final` apenas se quiser limitar o trecho.
5. Clique em `Processar`.

Resultado:

- o texto aparece na tela
- um arquivo `.md` é baixado automaticamente
- se deixar as páginas em branco, o sistema lê o PDF inteiro

## Significado das opções

- `anon`: tenta anonimizar informações pessoais (ex.: e-mail, telefone, CPF e entidades reconhecidas).
- `llm`: envia o texto para um serviço de IA local (se configurado) para melhorar a escrita.

## Dicas rápidas

- Se não precisar de IA local, deixe `llm` desmarcado.
- Se quiser processar o documento inteiro, use da primeira até a última página.
- Se deixar os campos de página em branco, o sistema processa do início ao fim automaticamente.
- Se o OCR falhar em uma página com imagem, o arquivo pode incluir o aviso:
  `**[Aviso: não foi possível converter a imagem desta página em texto.]**`

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