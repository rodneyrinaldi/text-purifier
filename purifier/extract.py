"""Processamento de UMA página (executado em processos worker).

Estratégia de máxima confiabilidade:
  1. Lê a camada de texto com três motores independentes (PyMuPDF, PDFium, pypdf).
  2. Renderiza a página (PyMuPDF; PDFium como renderizador alternativo) e executa OCR
     com várias tentativas (orientação, binarização, deskew, contraste).
  3. Compara as leituras (validação cruzada), escolhe/mescla a melhor e mede quanto
     da "tinta" da página foi efetivamente convertida em texto (cobertura).
  4. Calcula a saúde da leitura (0-100%) e registra tudo que for relevante.
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

from . import config
from .ocr import best_ocr, ocr_score, rotate
from .quality import (agreement, clean_text, detect_autos_page_number, is_good_text,
                      keep_readable_lines, merge_texts, text_metrics, text_score, _BAD_CHARS)

logger = logging.getLogger("text_purifier.extract")

_CACHE: Dict[str, Any] = {"key": None, "doc": None, "pypdf": None, "pdfium": None}


# --------------------------------------------------------------------------- #
# Acesso aos documentos (cache por processo)
# --------------------------------------------------------------------------- #

def _open(ctx: Dict[str, Any]) -> None:
    key = ctx["work_path"] + "|" + ctx["pdfium_path"]
    if _CACHE["key"] == key:
        return
    for name in ("doc", "pdfium"):
        try:
            if _CACHE[name] is not None:
                _CACHE[name].close()
        except Exception:
            pass
    _CACHE.update(key=key, doc=None, pypdf=None, pdfium=None)
    if ctx.get("pymupdf_pages"):
        try:
            doc = pymupdf.open(ctx["work_path"])
            if doc.needs_pass:
                doc.authenticate(ctx.get("password") or "")
            _CACHE["doc"] = doc
        except Exception:
            logger.exception("PyMuPDF não abriu %s no worker", ctx["work_path"])


def _mupdf_page(ctx: Dict[str, Any], n: int):
    _open(ctx)
    doc = _CACHE["doc"]
    if doc is None or n > doc.page_count:
        return None
    return doc[n - 1]


def _pdfium_doc(ctx: Dict[str, Any]):
    _open(ctx)
    if _CACHE["pdfium"] is None:
        import pypdfium2 as pdfium

        pw = None if ctx.get("decrypted_copy") else (ctx.get("password") or None)
        _CACHE["pdfium"] = pdfium.PdfDocument(ctx["pdfium_path"], password=pw)
    return _CACHE["pdfium"]


def _pypdf_reader(ctx: Dict[str, Any]):
    _open(ctx)
    if _CACHE["pypdf"] is None:
        from pypdf import PdfReader

        reader = PdfReader(ctx["pdfium_path"], strict=False)
        if reader.is_encrypted:
            reader.decrypt(ctx.get("password") or "")
        _CACHE["pypdf"] = reader
    return _CACHE["pypdf"]


# --------------------------------------------------------------------------- #
# Leituras da camada de texto
# --------------------------------------------------------------------------- #

def _native_readings(ctx: Dict[str, Any], n: int, page, errors: List[str]) -> Dict[str, str]:
    readings: Dict[str, str] = {}
    if page is not None:
        try:
            flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
            readings["PyMuPDF"] = page.get_text("text", sort=True, flags=flags) or ""
        except Exception as exc:
            errors.append(f"PyMuPDF (texto): {type(exc).__name__}: {exc}")
    try:
        pdoc = _pdfium_doc(ctx)
        if n <= len(pdoc):
            textpage = pdoc[n - 1].get_textpage()
            readings["PDFium"] = textpage.get_text_bounded() or ""
            textpage.close()
    except Exception as exc:
        errors.append(f"PDFium (texto): {type(exc).__name__}: {exc}")
    try:
        reader = _pypdf_reader(ctx)
        if n <= len(reader.pages):
            readings["pypdf"] = reader.pages[n - 1].extract_text() or ""
    except Exception as exc:
        errors.append(f"pypdf (texto): {type(exc).__name__}: {exc}")
    return readings


def _extras(page) -> Tuple[str, str]:
    """Valores de campos de formulário e conteúdo de anotações (não aparecem na camada de texto)."""
    forms: List[str] = []
    notes: List[str] = []
    if page is None:
        return "", ""
    try:
        for w in page.widgets() or []:
            value = w.field_value
            if value in (None, "", "Off", False):
                continue
            label = (w.field_label or w.field_name or "campo").strip()
            forms.append(f"{label}: {value}")
    except Exception:
        pass
    try:
        for annot in page.annots() or []:
            content = (annot.info or {}).get("content", "").strip()
            if content:
                author = (annot.info or {}).get("title", "").strip()
                notes.append(f"{annot.type[1]}{f' ({author})' if author else ''}: {content}")
    except Exception:
        pass
    return "\n".join(forms), "\n".join(notes)


def _native_word_boxes(page, scale: float) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if page is None:
        return boxes
    try:
        mat = page.rotation_matrix
        for w in page.get_text("words"):
            r = pymupdf.Rect(w[:4]) * mat
            boxes.append((r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
    except Exception:
        pass
    return boxes


# --------------------------------------------------------------------------- #
# Renderização
# --------------------------------------------------------------------------- #

def _choose_dpi(page, ctx_page_rect: Optional[Tuple[float, float]]) -> int:
    dpi = config.OCR_DPI
    if page is not None:
        try:
            area = abs(page.rect) or 1
            for info in page.get_image_info():
                bbox = pymupdf.Rect(info["bbox"]) & page.rect
                if bbox.is_empty or abs(bbox) / area < 0.3 or bbox.width <= 0:
                    continue
                native = info.get("width", 0) / (bbox.width / 72.0)
                dpi = max(dpi, min(int(native), config.OCR_MAX_DPI))
        except Exception:
            pass
        w, h = page.rect.width, page.rect.height
    elif ctx_page_rect:
        w, h = ctx_page_rect
    else:
        return dpi
    # limita o número de pixels (páginas gigantes)
    while dpi > 100 and (w * dpi / 72) * (h * dpi / 72) > config.OCR_MAX_PIXELS:
        dpi -= 25
    return dpi


def _render_mupdf(page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, annots=True)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
    return np.ascontiguousarray(arr)


def _render_pdfium(ctx: Dict[str, Any], n: int, dpi: int) -> np.ndarray:
    pdoc = _pdfium_doc(ctx)
    bitmap = pdoc[n - 1].render(scale=dpi / 72.0, grayscale=True, draw_annots=True)
    return np.ascontiguousarray(np.array(bitmap.to_pil().convert("L")))


def render_page(ctx: Dict[str, Any], n: int, dpi: int, prefer: str = "PyMuPDF") -> Tuple[np.ndarray, str]:
    """Renderiza em tons de cinza; usa o outro motor se o preferido falhar."""
    engines = ["PyMuPDF", "PDFium"] if prefer == "PyMuPDF" else ["PDFium", "PyMuPDF"]
    last_exc: Optional[Exception] = None
    for engine in engines:
        try:
            if engine == "PyMuPDF":
                page = _mupdf_page(ctx, n)
                if page is None:
                    continue
                return _render_mupdf(page, dpi), engine
            return _render_pdfium(ctx, n, dpi), engine
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"nenhum motor conseguiu renderizar a página: {last_exc}")


def ink_mask(gray: np.ndarray, target_dpi: int, dpi: int) -> Tuple[np.ndarray, float]:
    """Máscara de 'tinta' (qualquer marca visível) em resolução reduzida."""
    scale = min(1.0, target_dpi / float(dpi))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    bg = float(np.percentile(small, 90)) if small.size else 255.0
    thr = bg - max(40.0, 0.22 * bg)
    mask = (small < thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask, scale


def coverage(mask: np.ndarray, boxes: List[Tuple[float, float, float, float]], scale: float) -> float:
    """Fração da tinta da página que está dentro de áreas efetivamente lidas como texto."""
    total = int(mask.sum())
    if total < 30:
        return 1.0
    covered = np.zeros_like(mask)
    pad = 3
    H, W = mask.shape
    for x0, y0, x1, y1 in boxes:
        a, b = max(0, int(x0 * scale) - pad), max(0, int(y0 * scale) - pad)
        c, d = min(W, int(x1 * scale) + pad), min(H, int(y1 * scale) + pad)
        if c > a and d > b:
            covered[b:d, a:c] = 1
    return float((mask & covered).sum()) / total


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #

def _ocr_lines_outside(ocr: Dict[str, Any], native_boxes, min_conf: float = 70.0) -> str:
    """Linhas do OCR em regiões sem texto nativo (ex.: imagem escaneada colada na página)."""
    if not native_boxes:
        return ""
    out = []
    for line in ocr.get("lines", []):
        if line["conf"] < min_conf:
            continue
        x0, y0, x1, y1 = line["bbox"]
        area = max(1.0, (x1 - x0) * (y1 - y0))
        inter = 0.0
        for a, b, c, d in native_boxes:
            iw, ih = min(x1, c) - max(x0, a), min(y1, d) - max(y0, b)
            if iw > 0 and ih > 0:
                inter += iw * ih
        if inter / area < 0.15:
            words, quality = text_metrics(line["text"])
            if words >= 2 and quality >= 0.6:
                out.append(line["text"])
    return "\n".join(out)


def process_page(ctx: Dict[str, Any], n: int) -> Dict[str, Any]:
    t0 = time.time()
    res: Dict[str, Any] = {
        "page": n, "text": "", "autos": None, "health": 0, "method": "falha",
        "engines": [], "words": 0, "quality": 0.0, "ocr_conf": None, "coverage": None,
        "agreement": None, "dpi": None, "render": None, "rotation": 0, "ink": None,
        "blank": False, "flags": [], "errors": [], "ocr_text": "", "native_chars": 0,
        "vision_candidate": False, "elapsed": 0.0, "ocr_numbers": [],
    }
    try:
        _process(ctx, n, res)
    except Exception as exc:
        res["errors"].append(f"Falha geral: {type(exc).__name__}: {exc}")
        res["errors"].append(traceback.format_exc(limit=3).strip().splitlines()[-1])
        res["flags"].append("página não processada integralmente")
        res["health"] = 0
        res["vision_candidate"] = True
        logger.exception("Falha ao processar a página %s", n)
    res["elapsed"] = round(time.time() - t0, 2)
    return res


def _process(ctx: Dict[str, Any], n: int, res: Dict[str, Any]) -> None:
    errors, flags = res["errors"], res["flags"]
    page = _mupdf_page(ctx, n)
    if page is None:
        flags.append("página acessível apenas pelo motor PDFium")

    # 1) camada de texto (3 motores) -------------------------------------------
    readings = _native_readings(ctx, n, page, errors)
    native_name, native = "", ""
    if readings:
        # PyMuPDF preserva melhor a ordem de leitura; outro motor só vence se for claramente melhor
        scores = {k: text_score(v) for k, v in readings.items()}
        native_name = "PyMuPDF" if "PyMuPDF" in readings else max(scores, key=scores.get)
        best = max(scores, key=scores.get)
        if scores[best] > scores[native_name] * 1.15 + 5:
            native_name = best
        native = readings[native_name]
    native_words, native_q = text_metrics(native)
    res["native_chars"] = len(native.strip())
    if native.strip():
        bad = len(_BAD_CHARS.findall(native))
        if bad >= 5 or (native_words < config.MIN_WORDS and len(native.strip()) > 80):
            flags.append("camada de texto com caracteres inválidos (fonte sem mapeamento/PDF protegido)")
    native_ok = is_good_text(native)
    forms, notes = _extras(page)

    # 2) renderização -----------------------------------------------------------
    rect = None
    try:
        pdoc = _pdfium_doc(ctx) if page is None else None
        if pdoc is not None:
            rect = tuple(pdoc[n - 1].get_size())
    except Exception:
        pass
    dpi = _choose_dpi(page, rect)
    gray, render_engine = render_page(ctx, n, dpi)
    res["dpi"], res["render"] = dpi, render_engine
    mask, mscale = ink_mask(gray, 100, dpi)
    ink = float(mask.mean()) if mask.size else 0.0
    res["ink"] = round(ink, 4)

    # Renderização em branco mas com conteúdo conhecido: tenta o outro renderizador
    if ink < config.BLANK_INK_RATIO and (native_words > 10 or (page is not None and page.get_images())):
        try:
            alt_gray, alt_engine = render_page(ctx, n, dpi, prefer="PDFium" if render_engine == "PyMuPDF" else "PyMuPDF")
            alt_mask, _ = ink_mask(alt_gray, 100, dpi)
            if alt_mask.mean() > ink:
                gray, render_engine, mask, ink = alt_gray, alt_engine, alt_mask, float(alt_mask.mean())
                res["render"], res["ink"] = render_engine, round(ink, 4)
                flags.append(f"renderização alternativa ({alt_engine}) utilizada")
        except Exception as exc:
            errors.append(f"Renderização alternativa: {exc}")

    # 3) OCR --------------------------------------------------------------------
    ocr: Optional[Dict[str, Any]] = None
    tess_ok = ctx.get("tesseract_ok", False)
    need_ocr = config.ALWAYS_OCR or not native_ok or ink >= config.BLANK_INK_RATIO
    if tess_ok and need_ocr:
        ocr = best_ocr(gray, ctx["ocr_lang"], n)
        errors.extend(ocr.get("errors", []))
        # leitura fraca: segunda opinião com o outro renderizador
        if (not native_ok and ink >= config.BLANK_INK_RATIO and ocr["conf"] < 65):
            try:
                alt_gray, alt_engine = render_page(ctx, n, dpi, prefer="PDFium" if render_engine == "PyMuPDF" else "PyMuPDF")
                if alt_engine != render_engine:
                    alt = best_ocr(alt_gray, ctx["ocr_lang"], n)
                    if ocr_score(alt) > ocr_score(ocr) * 1.1:
                        ocr, gray, render_engine = alt, alt_gray, alt_engine
                        mask, mscale = ink_mask(gray, 100, dpi)
                        res["render"] = render_engine
                        flags.append(f"OCR refeito com renderizador {alt_engine} (melhor resultado)")
            except Exception as exc:
                errors.append(f"OCR alternativo: {exc}")
    elif not tess_ok:
        flags.append("OCR indisponível (Tesseract não encontrado)")

    ocr_text = ocr["text"] if ocr else ""
    res["ocr_text"] = ocr_text
    if ocr:
        from .llm import number_tokens
        res["ocr_numbers"] = [n for line in ocr.get("lines", []) if line["conf"] >= 75
                              for n in number_tokens(line["text"])]
    if ocr:
        res["ocr_conf"] = round(ocr["conf"], 1)
        res["rotation"] = ocr.get("angle", 0)
        if ocr.get("angle"):
            flags.append(f"página com orientação girada ({ocr['angle']}°) corrigida no OCR")
    ocr_ok = bool(ocr) and ocr["words"] >= config.MIN_WORDS and ocr["quality"] >= config.MIN_QUALITY and ocr["conf"] >= 60
    agree = agreement(native, ocr_text) if (native.strip() and ocr_text.strip()) else None
    res["agreement"] = None if agree is None else round(agree, 3)

    # 4) decisão e mescla ---------------------------------------------------------
    native_boxes = _native_word_boxes(page, dpi / 72.0)
    text, method, conf_component = "", "falha", 0.0

    if native_ok and not (ocr_ok and agree is not None and agree < 0.35 and ocr["conf"] >= 80):
        method = f"camada de texto ({native_name})"
        res["engines"].append(native_name)
        text = native
        conf_component = min(1.0, native_q / 0.85)
        if ocr:
            res["engines"].append("Tesseract (validação)")
            extra = _ocr_lines_outside(ocr, native_boxes) if ocr.get("angle", 0) == 0 else ""
            if extra:
                text = f"{text}\n\n{extra}"
                method += " + OCR de imagens"
                flags.append("trechos em imagem dentro da página lidos por OCR")
        if ocr_ok and agree is not None and agree < 0.35:
            flags.append("camada de texto diverge do conteúdo visual (conferir)")
            conf_component *= 0.75
    elif ocr and ocr_text.strip():
        method = "OCR (Tesseract)"
        res["engines"].append("Tesseract")
        text = ocr_text
        conf_component = ocr["conf"] / 100.0
        divergent = agree is not None and agree < 0.35
        if native.strip() and not divergent:
            readable = keep_readable_lines(native)
            if readable:
                text = merge_texts(text, readable)
                res["engines"].append(f"{native_name} (complemento)")
        if native_ok and divergent:
            flags.append("camada de texto (possivelmente invisível) diverge do conteúdo visual e foi descartada")
        elif native.strip():
            flags.append("camada de texto ilegível; usado OCR")
    elif native.strip():
        readable = keep_readable_lines(native)
        if readable:
            method = f"camada de texto parcial ({native_name})"
            res["engines"].append(native_name)
            text = readable
            conf_component = 0.4 * native_q

    text = clean_text(text)
    if forms:
        text = f"{text}\n\n[CAMPOS DE FORMULÁRIO]\n{forms}".strip()
    if notes:
        text = f"{text}\n\n[ANOTAÇÕES/COMENTÁRIOS NO PDF]\n{notes}".strip()

    res["autos"] = detect_autos_page_number(native) or detect_autos_page_number(ocr_text)
    words, quality = text_metrics(text)
    res["words"], res["quality"] = words, round(quality, 3)

    # 5) página em branco (confirmada por renderização + OCR) ----------------------
    if words < config.MIN_WORDS and ink < config.BLANK_INK_RATIO and not forms and not notes:
        res.update(text="", method="página em branco", blank=True, health=100, coverage=1.0, ocr_conf=None)
        if ocr is None and tess_ok is False:
            res["health"] = 90
        return

    # 6) cobertura: quanto da tinta foi convertida em texto ------------------------
    boxes: List[Tuple[float, float, float, float]] = []
    cov_mask = mask
    if ocr:
        angle = ocr.get("angle", 0)
        cov_mask = rotate(mask, angle) if angle else mask
        boxes.extend((b[0], b[1], b[2], b[3]) for b in ocr["boxes"] if b[4] >= 40)
        if angle == 0:
            boxes.extend(native_boxes)
    else:
        boxes.extend(native_boxes)
    cov = coverage(cov_mask, boxes, mscale)
    res["coverage"] = round(cov, 3)

    # 7) saúde da leitura -----------------------------------------------------------
    if agree is not None:
        a_comp = min(1.0, agree / 0.85)
    elif ocr and native.strip():
        a_comp = 0.8
    else:
        a_comp = 0.7
    health = 100 * (0.40 * conf_component + 0.25 * min(1.0, quality / 0.9) + 0.20 * cov + 0.15 * a_comp)
    if words < config.MIN_WORDS:
        health = min(health, 25)
        flags.append("conteúdo visual sem texto reconhecível (manuscrito, imagem, carimbo ou baixa qualidade)")
    elif cov < 0.75 and ink > 0.01:
        flags.append(f"{round((1 - cov) * 100)}% da tinta da página não foi convertida em texto "
                     f"(possível manuscrito, carimbo, assinatura ou imagem)")
    if ocr and method.startswith("OCR") and ocr["conf"] < 70:
        flags.append(f"OCR com confiança baixa ({ocr['conf']:.0f}%)")

    res["health"] = int(max(0, min(100, round(health))))
    res["text"] = text
    res["method"] = method
    res["vision_candidate"] = res["health"] < config.LLM_VISION_THRESHOLD


# --------------------------------------------------------------------------- #
# Imagem para o modelo de visão (executado no processo principal)
# --------------------------------------------------------------------------- #

def page_image_jpeg(ctx: Dict[str, Any], n: int, max_side: int, rotation: int = 0) -> bytes:
    page = _mupdf_page(ctx, n)
    if page is not None:
        longest = max(page.rect.width, page.rect.height) or 1
    else:
        w, h = _pdfium_doc(ctx)[n - 1].get_size()
        longest = max(w, h) or 1
    dpi = max(72, min(300, int(max_side / (longest / 72.0))))
    gray, _ = render_page(ctx, n, dpi)
    if rotation:
        gray = rotate(gray, rotation)
    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("falha ao codificar imagem da página")
    return buf.tobytes()


def close_cache() -> None:
    for name in ("doc", "pdfium"):
        try:
            if _CACHE[name] is not None:
                _CACHE[name].close()
        except Exception:
            pass
    _CACHE.update(key=None, doc=None, pypdf=None, pdfium=None)
