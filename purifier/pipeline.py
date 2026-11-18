"""Orquestração do processamento completo de um PDF."""

import getpass
import hashlib
import logging
import multiprocessing
import platform
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config, extract, llm
from .anonymize import anonymize_text
from .ocr import effective_lang, tesseract_status
from .pdfsource import PdfOpenError, cleanup, prepare_pdf
from .postprocess import numbering_gaps, remove_repeated_edges
from .progress import Tracker
from .quality import clean_text, detect_autos_page_number, text_metrics
from .report import build_output_filename, build_report, page_block

logger = logging.getLogger("text_purifier.pipeline")

_EXECUTOR: Optional[ProcessPoolExecutor] = None
_EXEC_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()   # um processamento por vez (evita disputa de CPU/GPU)
_TESS: Optional[Dict[str, Any]] = None


class InputError(Exception):
    """Erro de entrada do usuário (intervalo inválido, senha, etc.)."""


def _executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    with _EXEC_LOCK:
        if _EXECUTOR is None:
            method = "spawn" if sys.platform == "win32" else "forkserver"
            _EXECUTOR = ProcessPoolExecutor(max_workers=config.WORKERS,
                                            mp_context=multiprocessing.get_context(method))
        return _EXECUTOR


def shutdown_executor() -> None:
    global _EXECUTOR
    with _EXEC_LOCK:
        if _EXECUTOR is not None:
            try:
                _EXECUTOR.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _EXECUTOR = None


def tesseract_info(refresh: bool = False) -> Dict[str, Any]:
    global _TESS
    if _TESS is None or refresh:
        _TESS = tesseract_status()
    return _TESS


def library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"Python": platform.python_version()}
    try:
        import pymupdf
        versions["PyMuPDF"] = pymupdf.VersionBind
    except Exception:
        pass
    try:
        import pypdfium2
        versions["pypdfium2"] = str(getattr(pypdfium2, "__version__", "")) or str(pypdfium2.version.PYPDFIUM_INFO)
    except Exception:
        pass
    try:
        import pypdf
        versions["pypdf"] = pypdf.__version__
    except Exception:
        pass
    try:
        import pikepdf
        versions["pikepdf/qpdf"] = f"{pikepdf.__version__}/{pikepdf.__libqpdf_version__}"
    except Exception:
        pass
    try:
        import cv2
        versions["OpenCV"] = cv2.__version__
    except Exception:
        pass
    return versions


def _extract_all(ctx: Dict[str, Any], pages: List[int], tracker: Tracker) -> List[Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    def done(n: int) -> None:
        r = results[n]
        tracker.page_read(n, r["health"], bool(r.get("vision_candidate")) and not r.get("blank"),
                          r["method"].startswith("OCR") and bool(r.get("text")))

    total = len(pages)
    pending = list(pages)
    pool_attempted = config.WORKERS > 1 and total > 1
    if pool_attempted:
        try:
            ex = _executor()
            futures = {ex.submit(extract.process_page, ctx, n): n for n in pages}
            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    results[n] = fut.result()
                    done(n)
                except BrokenProcessPool:
                    logger.error("Pool de processos quebrou na página %s; reprocessando localmente", n)
                    shutdown_executor()
                    break
                except Exception:
                    logger.exception("Worker falhou na página %s", n)
        except Exception:
            logger.exception("Pool de processos indisponível; processando localmente")
            shutdown_executor()
        pending = [n for n in pages if n not in results]

    for n in pending:  # sequencial (fallback ou WORKERS=1)
        res = extract.process_page(ctx, n)
        if pool_attempted:
            res["flags"].append("reprocessada fora do pool paralelo após falha de worker")
        results[n] = res
        done(n)
    return [results[n] for n in pages]


def _now() -> datetime:
    return datetime.now().astimezone()


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S %Z (UTC%z)")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def run(pdf_path: str, filename: str, options: Dict[str, Any], tracker: Optional[Tracker] = None,
        client_ip: Optional[str] = None, job_id: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """Processa o PDF e devolve (markdown, nome_do_arquivo_saida, resumo).

    `tracker` recebe o andamento (etapas, páginas, IA em tempo real) e calcula a estimativa de tempo."""
    tracker = tracker or Tracker(options)
    job_id = job_id or uuid.uuid4().hex[:12]
    with _RUN_LOCK:
        tracker.begin()
        try:
            return _run(pdf_path, filename, options, tracker, client_ip, job_id)
        except Exception as exc:
            tracker.fail(str(exc) if isinstance(exc, InputError) else f"Falha: {type(exc).__name__}")
            raise


def _run(pdf_path: str, filename: str, options: Dict[str, Any], tracker: Tracker,
         client_ip: Optional[str], job_id: str) -> Tuple[str, str, Dict[str, Any]]:
    t0 = time.time()
    started = _now()
    tracker.stage("prepare", "Abrindo e validando o PDF (criptografia, reparo, contagem de páginas)")
    try:
        source = prepare_pdf(pdf_path, options.get("password") or "")
    except PdfOpenError as exc:
        raise InputError(str(exc)) from exc

    try:
        page_count = source["page_count"]
        start = options.get("page_start") or 1
        end = options.get("page_end") or page_count
        if start < 1 or end < 1 or start > end:
            raise InputError("Intervalo de páginas inválido.")
        if end > page_count:
            raise InputError(f"O PDF possui apenas {page_count} página(s).")
        options = {**options, "page_start": start, "page_end": end}
        pages_idx = list(range(start, end + 1))
        tracker.set_pages(len(pages_idx), start)

        tess = tesseract_info()
        if not tess.get("available"):
            tess = tesseract_info(refresh=True)
        warnings: List[str] = []
        errors: List[str] = []
        if not tess.get("available"):
            errors.append("Tesseract OCR indisponível: páginas digitalizadas/manuscritas não puderam ser lidas por OCR.")
        elif tess.get("missing_langs"):
            warnings.append(f"Idiomas de OCR ausentes: {', '.join(tess['missing_langs'])} (qualidade reduzida).")

        llm_status: Dict[str, Any] = {}
        use_vision, use_llm = bool(options.get("use_vision")), bool(options.get("use_llm"))
        if use_vision or use_llm:
            tracker.stage("ai_connect", f"Conectando à IA local ({config.LLM_URL})")
            llm_status = llm.discover()
            if not llm_status.get("available"):
                warnings.append(f"IA local indisponível ({llm_status.get('error')}); processamento seguiu sem IA.")
                if use_vision:
                    tracker.skip("vision", "IA indisponível")
                if use_llm:
                    tracker.skip("text", "IA indisponível")
            else:
                if use_vision and not llm_status.get("vision_model"):
                    tracker.skip("vision", "sem modelo de visão no servidor")
                if use_llm and not llm_status.get("text_model"):
                    tracker.skip("text", "sem modelo de texto no servidor")

        ctx = {
            "work_path": source["work_path"], "pdfium_path": source["pdfium_path"],
            "password": options.get("password") or "", "decrypted_copy": source["decrypted_copy"],
            "pymupdf_pages": source["pymupdf_pages"], "tesseract_ok": bool(tess.get("available")),
            "ocr_lang": effective_lang(tess),
        }

        # 1) leitura de todas as páginas ---------------------------------------------
        tracker.stage("read", f"Lendo páginas — 0/{len(pages_idx)}", total=len(pages_idx))
        pages = _extract_all(ctx, pages_idx, tracker)
        tracker.finish_stage("read")

        # 2) transcrição por visão das páginas comprometidas -----------------------------
        ai_events: List[Dict[str, Any]] = []
        vision_model = llm_status.get("vision_model") if llm_status.get("available") else None
        if use_vision and llm_status.get("available") and not vision_model:
            warnings.append("Transcrição por visão solicitada, mas não há modelo de visão no servidor Ollama.")
        if use_vision and vision_model:
            candidates = [p for p in pages if p.get("vision_candidate") and not p.get("blank")]
            skipped = []
            if config.LLM_VISION_MAX_PAGES and len(candidates) > config.LLM_VISION_MAX_PAGES:
                candidates.sort(key=lambda p: p["health"])
                skipped = candidates[config.LLM_VISION_MAX_PAGES:]
                candidates = sorted(candidates[: config.LLM_VISION_MAX_PAGES], key=lambda p: p["page"])
                warnings.append(f"Limite LLM_VISION_MAX_PAGES={config.LLM_VISION_MAX_PAGES}: "
                                f"{len(skipped)} página(s) comprometida(s) não enviadas à IA: "
                                + ", ".join(f"{p['page']:05d}" for p in skipped[:50]))
            tracker.stage("vision", f"IA de visão ({vision_model}): {len(candidates)} página(s) comprometida(s)",
                          total=len(candidates))
            for i, p in enumerate(candidates, 1):
                ocr_len = len(p.get("ocr_text") or p.get("text") or "")
                tracker.unit_begin("vision", p["page"], int(ocr_len / 3.0),
                                   f"IA de visão transcrevendo a página {p['page']} ({i}/{len(candidates)})")
                try:
                    render = (lambda side, n=p["page"], rot=p.get("rotation") or 0:
                              extract.page_image_jpeg(ctx, n, side, rot))
                    t = llm.transcribe_page(vision_model, render, p.get("ocr_text") or "", p.get("ocr_conf"),
                                            p.get("ocr_numbers"), on_tokens=tracker.llm_activity,
                                            on_retry=tracker.llm_retry)
                except Exception as exc:
                    t = {"accepted": False, "reason": f"erro: {type(exc).__name__}: {exc}", "health": 0}
                finally:
                    tracker.unit_end("vision")
                if t["accepted"] and t["health"] > p["health"]:
                    old = p["health"]
                    p["text"] = clean_text(t["text"])
                    p["health"] = t["health"]
                    p["method"] = f"IA visão ({vision_model})"
                    p["vision_used"], p["vision_model"] = True, vision_model
                    tracker.update_health(p["page"], p["health"])
                    p["agreement"] = None if t.get("agreement") is None else round(t["agreement"], 3)
                    p["coverage"] = None
                    p["words"], q = text_metrics(p["text"])
                    p["quality"] = round(q, 3)
                    p["autos"] = p.get("autos") or detect_autos_page_number(p["text"])
                    p["flags"].append(f"transcrita por IA de visão (saúde {old}% → {p['health']}%)")
                    extra = ""
                    if t.get("missing_numbers"):
                        nums = ", ".join(t["missing_numbers"][:10])
                        p["flags"].append(f"números lidos pelo OCR ausentes na transcrição da IA: {nums} (conferir)")
                        extra = f"; números do OCR ausentes: {nums}"
                    ai_events.append({"page": p["page"], "kind": "transcrição por visão",
                                      "result": f"aceita; saúde {old}% → {p['health']}%{extra}"})
                else:
                    reason = t["reason"] if not t["accepted"] else "não superou a leitura por OCR"
                    p["flags"].append(f"IA de visão não aproveitada ({reason})")
                    ai_events.append({"page": p["page"], "kind": "transcrição por visão", "result": reason})
            accepted = sum(1 for p in candidates if p.get("vision_used"))
            tracker.finish_stage("vision", f"{accepted} de {len(candidates)} transcrição(ões) aceita(s)"
                                 if candidates else "nenhuma página precisou")

        # 3) correção de OCR por modelo de texto -----------------------------------------
        text_model = llm_status.get("text_model") if llm_status.get("available") else None
        if use_llm and text_model:
            targets = [p for p in pages if p["method"].startswith("OCR") and p.get("text")]
            tracker.stage("text", f"IA de texto ({text_model}): {len(targets)} página(s) lidas por OCR",
                          total=len(targets))
            for i, p in enumerate(targets, 1):
                tracker.unit_begin("text", p["page"], int(len(p["text"]) / 3.2),
                                   f"IA de texto revisando o OCR da página {p['page']} ({i}/{len(targets)})")
                try:
                    c = llm.correct_text(text_model, p["text"], on_tokens=tracker.llm_activity)
                finally:
                    tracker.unit_end("text")
                if c["applied"]:
                    p["text"] = c["text"]
                    p["llm_corrected"], p["text_model"] = True, text_model
                ai_events.append({"page": p["page"], "kind": "correção de OCR", "result": c["reason"]})
            applied = sum(1 for p in targets if p.get("llm_corrected"))
            tracker.finish_stage("text", f"{applied} de {len(targets)} correção(ões) aplicada(s)"
                                 if targets else "nenhuma página lida por OCR")

        # 4) cabeçalhos/rodapés repetidos ---------------------------------------------
        removed = []
        if options.get("remove_repeated"):
            tracker.stage("post", "Removendo cabeçalhos e rodapés repetidos")
            removed = remove_repeated_edges(pages)
            tracker.finish_stage("post", f"{len(removed)} padrão(ões) removido(s)")

        # 5) anonimização --------------------------------------------------------------
        if options.get("anon"):
            tracker.stage("anon", "Anonimizando dados pessoais", total=len(pages))
            for i, p in enumerate(pages, 1):
                if p.get("text"):
                    p["text"] = anonymize_text(p["text"])
                tracker.step("anon", i, f"Anonimizando — {i}/{len(pages)}")

        # 6) montagem ----------------------------------------------------------------
        tracker.stage("build", "Montando o documento e o relatório de processamento")
        body = "\n\n".join(page_block(p) for p in pages).strip()
        finished = _now()
        report_ctx = {
            "job_id": job_id, "filename": filename, "source": source, "options": options,
            "tesseract": tess, "llm": llm_status, "ocr_lang": ctx["ocr_lang"],
            "versions": library_versions(), "workers": config.WORKERS,
            "who": {"user": _safe_user(), "host": platform.node() or "desconhecido",
                    "os": f"{platform.system()} {platform.release()}", "client_ip": client_ip},
            "started": _fmt_dt(started), "finished": _fmt_dt(finished),
            "duration": _fmt_duration(time.time() - t0),
            "removed_patterns": removed, "gaps": numbering_gaps(pages), "warnings": warnings,
            "errors": errors, "ai_events": ai_events,
            "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
        report = build_report(report_ctx, pages, anonymize_text if options.get("anon") else None)
        markdown = f"{body}\n\n{report}\n"
        output_name = build_output_filename(filename, start, end)

        content_pages = [p for p in pages if not p.get("blank")]
        summary = {
            "job_id": job_id, "pages": len(pages),
            "avg_health": round(sum(p["health"] for p in content_pages) / len(content_pages)) if content_pages else 100,
            "min_health": min((p["health"] for p in pages), default=0),
            "compromised": [p["page"] for p in pages if p["health"] < config.HEALTH_WARN or p.get("vision_used")],
            "duration": report_ctx["duration"], "output_name": output_name,
            "health_by_page": [{"page": p["page"], "health": p["health"]} for p in pages],
        }
        tracker.complete()
        return markdown, output_name, summary
    finally:
        extract.close_cache()
        cleanup(source)


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "desconhecido"
