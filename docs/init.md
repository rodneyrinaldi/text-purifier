Gere o código completo seguindo as instruções definidas no arquivo de configuração do projeto.

Crie:
- Dockerfile
- requirements.txt
- main.py
- .env.example
- static/index.html

O sistema deve:
- receber um PDF
- limpar cabeçalhos e rodapés
- extrair texto limpo
- tentar OCR quando houver imagens
- inserir aviso em Markdown quando o OCR falhar
- permitir anonimização opcional
- permitir uso opcional da IA local via HTTP
- retornar sempre em Markdown

Use a função improve_with_local_llm(text) conforme definido nas instruções gerais.

Não adicione nada além do necessário.
