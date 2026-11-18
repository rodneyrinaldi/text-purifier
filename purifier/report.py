"""Montagem do Markdown final: blocos por página e seção de informações sobre o processamento."""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import APP_NAME, VERSION, config
from .quality import health_level


def build_output_filename(original_name: str, page_start: int, page_end: int) -> str:
    stem = Path(original_name or "arquivo").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "arquivo"
    return f"{safe_stem}de{page_start:05d}a{page_end:05d}.md"


def page_header(p: Dict[str, Any]) -> str:
    physical = f"{p['page']:05d}"
    autos = p.get("autos")
    if autos and autos != physical:
        ident = f"PAGINA_AUTOS:{autos} ou PAGINA_FISICA:{physical}"
    elif autos:
        ident = f"PAGINA_AUTOS:{autos}"
    else:
        ident = f"PAGINA_FISICA:{physical}"
    return f"[{ident} | LEITURA:{p['health']}% ({health_level(p['health'])})]"


def page_block(p: Dict[str, Any]) -> str:
    parts = [page_header(p)]
    notices: List[str] = []
    if p.get("blank"):
        notices.append("*[Página em branco — confirmada por renderização e OCR]*")
    elif not p.get("text"):
        notices.append("**[Aviso: não foi possível extrair texto desta página. Conteúdo visual sem texto "
                       "reconhecível (manuscrito, imagem, carimbo ou arquivo danificado). Consultar o original.]**")
    elif p["health"] < config.HEALTH_WARN:
        notices.append("**[Aviso: leitura comprometida; o texto pode conter erros ou estar incompleto. "
                       "Conferir com o original.]**")
    if p.get("vision_used"):
        notices.append(f"**[Transcrição assistida por IA (modelo {p['vision_model']}) — conferir com o original.]**")
    if p.get("llm_corrected"):
        notices.append(f"*[Texto de OCR revisado por IA (modelo {p['text_model']}) com salvaguardas.]*")
    if notices:
        parts.extend(["", *notices])
    if p.get("text"):
        parts.extend(["", p["text"].strip()])
    return "\n".join(parts).strip()


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return str(n)


def _fmt_pdf_date(value: str) -> str:
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", value or "")
    if not m:
        return value
    y, mo, d, h, mi, s = (g or "00" for g in m.groups())
    return f"{d}/{mo}/{y} {h}:{mi}:{s}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def build_report(ctx: Dict[str, Any], pages: List[Dict[str, Any]],
                 anonymize: Optional[Callable[[str], str]] = None) -> str:
    """Seção final com: quem, quando, como, saúde, leituras comprometidas, erros e divergências."""
    an = anonymize or (lambda s: s)
    src = ctx["source"]
    opts = ctx["options"]
    tess = ctx["tesseract"]
    llm = ctx.get("llm") or {}
    L: List[str] = ["---", "", "## INFORMAÇÕES SOBRE O PROCESSAMENTO", ""]

    # --- resumo -----------------------------------------------------------------
    healths = [p["health"] for p in pages]
    content = [p for p in pages if not p.get("blank")]
    avg = round(sum(p["health"] for p in content) / len(content)) if content else 100
    compromised = [p for p in pages if p["health"] < config.HEALTH_WARN]
    critical = [p for p in pages if p["health"] < config.HEALTH_CRITICAL]
    blank = [p for p in pages if p.get("blank")]
    ocr_pages = [p for p in pages if p["method"].startswith("OCR")]
    vision_pages = [p for p in pages if p.get("vision_used")]
    error_pages = [p for p in pages if p.get("errors")]

    L += ["### Resumo", ""]
    L.append(f"- **Resultado geral:** {len(pages)} página(s) processada(s); saúde média da leitura "
             f"**{avg}%** (mínima {min(healths) if healths else 0}%).")
    L.append(f"- **Leituras comprometidas (< {config.HEALTH_WARN}%):** {len(compromised)}"
             f" — das quais críticas (< {config.HEALTH_CRITICAL}%): {len(critical)}.")
    L.append(f"- **Páginas lidas por OCR:** {len(ocr_pages)} · **pela camada de texto:** "
             f"{len(pages) - len(ocr_pages) - len(blank) - len(vision_pages)} · **transcritas com IA:** "
             f"{len(vision_pages)} · **em branco:** {len(blank)}.")
    L.append(f"- **Páginas com erros técnicos:** {len(error_pages)}.")
    bands = [("ALTA (≥ 90%)", 90, 101), (f"MÉDIA ({config.HEALTH_WARN}–89%)", config.HEALTH_WARN, 90),
             (f"BAIXA ({config.HEALTH_CRITICAL}–{config.HEALTH_WARN - 1}%)", config.HEALTH_CRITICAL, config.HEALTH_WARN),
             (f"CRÍTICA (< {config.HEALTH_CRITICAL}%)", -1, config.HEALTH_CRITICAL)]
    L.append("- **Distribuição da saúde:** " + " · ".join(
        f"{label}: {sum(lo <= h < hi for h in healths)}" for label, lo, hi in bands))
    L.append("")

    # --- quem / quando ---------------------------------------------------------------
    L += ["### Quem e quando", ""]
    L.append(f"- **Operador informado:** {opts.get('operator') or 'não informado'}")
    L.append(f"- **Usuário do sistema / máquina:** {ctx['who']['user']} @ {ctx['who']['host']} ({ctx['who']['os']})")
    if ctx["who"].get("client_ip"):
        L.append(f"- **Origem da requisição (IP):** {ctx['who']['client_ip']}")
    L.append(f"- **Início:** {ctx['started']}")
    L.append(f"- **Término:** {ctx['finished']}")
    L.append(f"- **Duração:** {ctx['duration']}")
    L.append(f"- **Identificador do processamento:** `{ctx['job_id']}`")
    L.append("")

    # --- arquivo -------------------------------------------------------------------
    L += ["### Documento de origem", ""]
    L.append(f"- **Arquivo:** {an(ctx['filename'])}")
    L.append(f"- **Tamanho:** {_fmt_bytes(src['size_bytes'])}")
    L.append(f"- **SHA-256 (integridade/cadeia de custódia):** `{src['sha256']}`")
    L.append(f"- **Versão PDF:** {src.get('pdf_version') or 'desconhecida'}")
    L.append(f"- **Páginas no arquivo:** {src['page_count']} · **intervalo processado:** "
             f"{opts['page_start']} a {opts['page_end']}")
    L.append("- **Contagem por motor:** " + ", ".join(f"{k}={v}" for k, v in src["page_counts"].items()))
    meta = src.get("metadata") or {}
    labels = {"title": "Título", "author": "Autor", "subject": "Assunto", "keywords": "Palavras-chave",
              "creator": "Aplicativo de criação", "producer": "Gerador do PDF",
              "creationDate": "Criado em", "modDate": "Modificado em"}
    for key, label in labels.items():
        if meta.get(key):
            value = _fmt_pdf_date(meta[key]) if key.endswith("Date") else an(meta[key])
            L.append(f"- **{label}:** {_cell(value)}")
    if src.get("encrypted"):
        L.append(f"- **Proteção:** criptografado ({src.get('encryption') or 'padrão'})"
                 + (f"; restrições: {', '.join(src['restrictions'])}" if src["restrictions"] else "")
                 + ". Restrições contornadas para leitura integral"
                 + (" (cópia de trabalho descriptografada com qpdf)." if src.get("decrypted_copy") else "."))
    else:
        L.append("- **Proteção:** sem criptografia/restrições.")
    if src.get("needs_password"):
        L.append(f"- **Senha de abertura:** exigida; {'informada e aceita' if src.get('password_used') else 'senha vazia aceita'}.")
    if src.get("repaired"):
        L.append("- **Estrutura:** o arquivo apresentava erros estruturais e foi reparado automaticamente.")
    if src.get("signature_fields") or src.get("signed"):
        L.append(f"- **Assinatura digital:** {src.get('signature_fields', 0)} campo(s) de assinatura detectado(s). "
                 "A validade criptográfica das assinaturas NÃO foi verificada por este processamento.")
    if src.get("has_forms"):
        L.append("- **Formulário:** o PDF contém campos de formulário (valores incluídos nas páginas).")
    L.append("")

    # --- como ----------------------------------------------------------------------
    L += ["### Como foi processado", ""]
    L.append(f"- **Sistema:** {APP_NAME} {VERSION}")
    L.append("- **Método:** leitura da camada de texto por 3 motores independentes (PyMuPDF, PDFium, pypdf); "
             "renderização de cada página e OCR com múltiplas tentativas (orientação, binarização, deskew, "
             "realce de contraste); validação cruzada entre as leituras; renderizador alternativo (PDFium) "
             "quando necessário; medição da cobertura da tinta da página convertida em texto.")
    L.append(f"- **Modo OCR:** {'em todas as páginas (validação cruzada)' if config.ALWAYS_OCR else 'somente quando necessário'}"
             f"; resolução base {config.OCR_DPI} DPI (até {config.OCR_MAX_DPI} DPI conforme a digitalização); "
             f"idiomas: {ctx.get('ocr_lang')}.")
    if tess.get("available"):
        L.append(f"- **Tesseract:** {tess['version']}"
                 + (f" — idiomas ausentes: {', '.join(tess['missing_langs'])}" if tess.get("missing_langs") else ""))
    else:
        L.append(f"- **Tesseract:** INDISPONÍVEL ({tess.get('error', 'não encontrado')}). Páginas digitalizadas "
                 "não puderam ser lidas por OCR.")
    L.append("- **Bibliotecas:** " + ", ".join(f"{k} {v}" for k, v in ctx["versions"].items()))
    L.append(f"- **Workers paralelos:** {ctx.get('workers')}")
    if opts.get("use_vision") or opts.get("use_llm"):
        if llm.get("available"):
            L.append(f"- **IA local (Ollama {llm.get('version') or ''}):** {llm['url']} — "
                     f"modelo de visão: {llm.get('vision_model') or 'nenhum'}; modelo de texto: {llm.get('text_model') or 'nenhum'}.")
            L.append(f"  - Transcrição por visão: {'ativada' if opts.get('use_vision') else 'desativada'} "
                     f"(páginas com saúde < {config.LLM_VISION_THRESHOLD}%); correção de OCR: "
                     f"{'ativada' if opts.get('use_llm') else 'desativada'}.")
            for note in llm.get("notes", []):
                L.append(f"  - {note}")
        else:
            L.append(f"- **IA local:** solicitada, mas INDISPONÍVEL em {llm.get('url')} ({llm.get('error')}). "
                     "Processamento concluído sem IA.")
    else:
        L.append("- **IA local:** não utilizada.")
    L.append(f"- **Anonimização:** {'aplicada (regex + NER)' if opts.get('anon') else 'não aplicada'}.")
    removed = ctx.get("removed_patterns") or []
    if opts.get("remove_repeated"):
        L.append(f"- **Cabeçalhos/rodapés repetidos:** {len(removed)} padrão(ões) removido(s) "
                 "(mantida a primeira ocorrência de cada).")
        for r in removed[:15]:
            L.append(f"  - \"{_cell(an(r['example']))[:160]}\" — removido {r['removed']}x")
    else:
        L.append("- **Cabeçalhos/rodapés repetidos:** mantidos (remoção desativada).")
    L.append("- **Índice de saúde por página:** combina confiança do motor de leitura (40%), qualidade "
             "linguística do texto (25%), cobertura da tinta convertida em texto (20%) e concordância entre "
             "leituras independentes (15%). Transcrições por IA têm saúde limitada a 85%.")
    L.append("")

    # --- leituras comprometidas -------------------------------------------------------
    L += ["### Leituras comprometidas e páginas a conferir (revisar no original)", ""]
    review = [p for p in pages if p["health"] < config.HEALTH_WARN or p.get("vision_used")]
    if review:
        for p in review:
            autos = f" (fls. {int(p['autos'])})" if p.get("autos") else ""
            reasons = "; ".join(p.get("flags") or []) or "baixa confiança geral"
            tag = "transcrita por IA" if p.get("vision_used") and p["health"] >= config.HEALTH_WARN else health_level(p["health"])
            L.append(f"- **Página {p['page']:05d}{autos} — {p['health']}% ({tag}):** {reasons}.")
    else:
        L.append("- Nenhuma. Todas as páginas foram lidas com saúde ≥ "
                 f"{config.HEALTH_WARN}% sem uso de transcrição por IA.")
    L.append("")

    # --- IA ----------------------------------------------------------------------
    ai_events = ctx.get("ai_events") or []
    if ai_events:
        L += ["### Uso da IA por página", ""]
        for ev in ai_events:
            L.append(f"- Página {ev['page']:05d}: {ev['kind']} — {ev['result']}")
        L.append("")

    # --- divergências ---------------------------------------------------------------
    notes = list(src.get("warnings") or []) + list(ctx.get("gaps") or []) + list(ctx.get("warnings") or [])
    L += ["### Divergências e observações", ""]
    L += [f"- {an(n)}" for n in notes] or ["- Nenhuma."]
    L.append("")

    # --- erros ---------------------------------------------------------------------
    L += ["### Erros técnicos", ""]
    errs = [f"- {e}" for e in (src.get("errors") or []) + (ctx.get("errors") or [])]
    for p in error_pages:
        for e in p["errors"][:6]:
            errs.append(f"- Página {p['page']:05d}: {_cell(e)[:300]}")
    L += errs or ["- Nenhum erro registrado."]
    L.append("")

    # --- tabela por página -------------------------------------------------------------
    L += ["### Detalhamento por página", "",
          "| Pág. física | Fls. autos | Saúde | Método | Palavras | Conf. OCR | Cobertura | Concordância | DPI | Tempo |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in pages:
        L.append("| {pg:05d} | {autos} | {h}% {lvl} | {m} | {w} | {c} | {cov} | {ag} | {dpi} | {t}s |".format(
            pg=p["page"], autos=int(p["autos"]) if p.get("autos") else "—", h=p["health"],
            lvl=health_level(p["health"]), m=_cell(p["method"]), w=p["words"],
            c=f"{p['ocr_conf']:.0f}%" if p.get("ocr_conf") is not None else "—",
            cov=f"{p['coverage'] * 100:.0f}%" if p.get("coverage") is not None else "—",
            ag=f"{p['agreement'] * 100:.0f}%" if p.get("agreement") is not None else "—",
            dpi=p.get("dpi") or "—", t=p.get("elapsed", 0)))
    L.append("")
    L.append(f"- **SHA-256 do conteúdo extraído (acima desta seção):** `{ctx.get('content_sha256')}`")
    L.append("")
    L.append("*Relatório gerado automaticamente. O índice de saúde é uma estimativa técnica da qualidade da "
             "leitura e não substitui a conferência humana com o documento original.*")
    return "\n".join(L)
