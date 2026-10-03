import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # evita oversubscription do Tesseract em paralelo

import io
import re
import logging
import tempfile
import unicodedata
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pymupdf
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from PIL import Image
from pypdf import PdfReader


load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("r2_text_purify")

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "static" / "index.html"

OCR_WARNING = "**[Aviso: não foi possível converter a imagem desta página em texto por problemas de codificação no PDF.]**"
PAGE_PROCESSING_WARNING = "**[Aviso: não foi possível processar esta página completamente.]**"
LOW_CONFIDENCE_WARNING = "**[Aviso: texto extraído com baixa confiança; pode conter erros ou estar incompleto.]**"
BLANK_PAGE_NOTE = "*[Página em branco]*"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _available_cpus() -> int:
    """CPUs efetivas, respeitando limite de cgroup (docker --cpus)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 2


MIN_WORDS = _env_int("MIN_WORDS", 6)                      # palavras válidas mínimas p/ página "boa"
MIN_QUALITY = _env_float("MIN_QUALITY", 0.55)             # razão palavras válidas / tokens
TRUST_NATIVE_WORDS = _env_int("TRUST_NATIVE_WORDS", 150)  # acima disso, texto nativo dispensa OCR
IMAGE_COVER_OCR = _env_float("IMAGE_COVER_OCR", 0.30)     # cobertura de imagem que força OCR
BLANK_INK_RATIO = _env_float("BLANK_INK_RATIO", 0.003)
OCR_DPI = _env_int("OCR_DPI", 300)
OCR_LANG = os.getenv("OCR_LANG", "por+eng")
OCR_TIMEOUT = _env_int("OCR_TIMEOUT", 180)
WORKERS = max(1, _env_int("WORKERS", _available_cpus()))
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 300)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 120)
SPACY_MODEL = os.getenv("SPACY_MODEL", "pt_core_news_sm")

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

cv2.setNumThreads(1)


# --------------------------------------------------------------------------- #
# Qualidade de texto
# --------------------------------------------------------------------------- #

_BAD_CHARS = re.compile(r"[-�□\x00-\x08\x0b\x0c\x0e-\x1f]")
_FONT_GARBAGE = re.compile(r"(?:(?<=\s)|^)/(?:\d{1,3}|C[0-9A-Fa-f]{2})(?=\s|/|$)", re.M)
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
_VOWEL = re.compile(r"[aeiouyáéíóúâêôãõàüAEIOUYÁÉÍÓÚÂÊÔÃÕÀÜ]")
_NUMERIC = re.compile(r"[\d.,/\-:%ºª°R$]+")
_STRIP = ".,;:!?()[]{}<>\"'«»“”‘’`´-–—_/\\|*•·"
STOPWORDS = frozenset(
    "a o e é de da do das dos em no na nos nas um uma para por com não que se ao à "
    "os as ou mais como foi ser são pelo pela seu sua sobre nº the of and to in".split()
)


def text_metrics(text: str) -> Tuple[int, float]:
    tokens = (text or "").split()
    if not tokens:
        return 0, 0.0

    words = junk = stops = 0
    for tok in tokens:
        core = tok.strip(_STRIP)
        if not core:
            continue
        low = core.lower()
        if len(core) >= 2 and _WORD.fullmatch(core) and _VOWEL.search(core):
            words += 1
            stops += low in STOPWORDS
        elif len(core) == 1 and low in STOPWORDS:
            words += 1
            stops += 1
        elif _NUMERIC.fullmatch(core):
            continue
        else:
            junk += 1

    bad = len(_BAD_CHARS.findall(text)) + len(_FONT_GARBAGE.findall(text))
    denom = words + junk + bad
    quality = words / denom if denom else 0.0

    # Texto com fonte "cifrada" tem cara de palavra mas não tem stopwords
    if words >= 40 and stops / words < 0.04:
        quality *= 0.5

    return words, quality


def text_score(text: str) -> float:
    words, quality = text_metrics(text)
    return words * quality


def is_good_text(text: str, min_words: int = MIN_WORDS) -> bool:
    words, quality = text_metrics(text)
    return words >= min_words and quality >= MIN_QUALITY


def keep_readable_lines(text: str) -> str:
    kept = []
    for line in (text or "").split("\n"):
        words, quality = text_metrics(line)
        if words >= 2 and quality >= 0.5:
            kept.append(line.strip())
    return "\n".join(kept)


_PAGE_LINE = re.compile(r"(?i)(?:p[áa]g(?:ina)?\.?|page)\s*\d+(?:\s*(?:de|of|/)\s*\d+)?")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _BAD_CHARS.sub("", text)
    text = re.sub(r"([a-zà-ÿ])-\n([a-zà-ÿ])", r"\1\2", text)

    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if not _PAGE_LINE.fullmatch(line)]

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if lines and lines[0].isdigit() and len(lines[0]) <= 4:
        lines.pop(0)
    if lines and lines[-1].isdigit() and len(lines[-1]) <= 4:
        lines.pop()

    out, last_blank = [], False
    for line in lines:
        blank = not line
        if blank and last_blank:
            continue
        out.append(line)
        last_blank = blank
    return "\n".join(out).strip()


def merge_texts(primary: str, secondary: str) -> str:
    """Acrescenta ao primário as linhas legíveis do secundário que ele não contém."""
    if not secondary.strip():
        return primary
    if not primary.strip():
        return secondary

    base_tokens = set(re.findall(r"\w+", primary.lower()))
    extra = []
    for line in secondary.split("\n"):
        tokens = re.findall(r"\w+", line.lower())
        if len(tokens) < 2:
            continue
        words, quality = text_metrics(line)
        if words < 2 or quality < 0.5:
            continue
        overlap = sum(t in base_tokens for t in tokens) / len(tokens)
        if overlap < 0.7:
            extra.append(line.strip())

    return f"{primary}\n\n" + "\n".join(extra) if extra else primary


# --------------------------------------------------------------------------- #
# Extração por página (executa em processos worker)
# --------------------------------------------------------------------------- #

_DOC_CACHE: Dict[str, object] = {"path": None, "data": None, "doc": None, "pypdf": None}


def _get_doc(pdf_path: str):
    if _DOC_CACHE["path"] != pdf_path:
        if _DOC_CACHE["doc"] is not None:
            try:
                _DOC_CACHE["doc"].close()
            except Exception:
                pass
        data = Path(pdf_path).read_bytes()  # em memória: não prende handle
        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.needs_pass:
            doc.authenticate("")
        _DOC_CACHE.update(path=pdf_path, data=data, doc=doc, pypdf=None)
    return _DOC_CACHE["doc"]


def _pypdf_text(pdf_path: str, page_number: int) -> str:
    try:
        _get_doc(pdf_path)
        if _DOC_CACHE["pypdf"] is None:
            _DOC_CACHE["pypdf"] = PdfReader(io.BytesIO(_DOC_CACHE["data"]))
        return _DOC_CACHE["pypdf"].pages[page_number - 1].extract_text() or ""
    except Exception:
        logger.debug("pypdf falhou na página %s", page_number, exc_info=True)
        return ""


def _native_text(page) -> str:
    try:
        flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
        return page.get_text("text", sort=True, flags=flags) or ""
    except Exception:
        logger.debug("PyMuPDF falhou na extração nativa", exc_info=True)
        return ""


def _image_coverage(page) -> float:
    area = abs(page.rect)
    if not area:
        return 0.0
    covered = 0.0
    try:
        for info in page.get_image_info():
            rect = pymupdf.Rect(info["bbox"]) & page.rect
            if not rect.is_empty:
                covered += abs(rect)
    except Exception:
        return 1.0  # na dúvida, força OCR
    return min(1.0, covered / area)


def _render_gray(page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
    return np.ascontiguousarray(arr)


def _is_blank(page) -> bool:
    gray = _render_gray(page, 40)
    return gray.size == 0 or float((gray < 200).mean()) < BLANK_INK_RATIO


# ---- OCR ------------------------------------------------------------------ #

def _tesseract(arr: np.ndarray, psm: int, page_number: int) -> str:
    try:
        return pytesseract.image_to_string(
            Image.fromarray(arr),
            lang=OCR_LANG,
            config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
            timeout=OCR_TIMEOUT,
        )
    except Exception:
        logger.warning("Tesseract falhou na página %s (psm %s)", page_number, psm, exc_info=True)
        return ""


def _osd_rotation(arr: np.ndarray) -> Optional[int]:
    """Rotação (0/90/180/270) sugerida pelo OSD do Tesseract; None se indeterminado."""
    try:
        osd = pytesseract.image_to_osd(
            Image.fromarray(arr), config="--psm 0 -c min_characters_to_try=5", timeout=60
        )
        match = re.search(r"Rotate:\s*(\d+)", osd)
        return int(match.group(1)) % 360 if match else None
    except Exception:
        return None


def _rotate(arr: np.ndarray, degrees_cw: int) -> np.ndarray:
    if degrees_cw == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if degrees_cw == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if degrees_cw == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return arr


def _binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Corrige inclinação leve (±5°) por perfil de projeção horizontal."""
    inv = 255 - binary
    small = cv2.resize(inv, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    if h < 20 or w < 20:
        return binary

    center = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-5.0, 5.01, 0.5):
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score

    if abs(best_angle) < 0.25:
        return binary
    H, W = binary.shape
    m = cv2.getRotationMatrix2D((W / 2, H / 2), best_angle, 1.0)
    return cv2.warpAffine(binary, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def _ocr_page(page, page_number: int) -> str:
    gray = _render_gray(page, OCR_DPI)
    candidates: List[str] = []

    def attempt(arr: np.ndarray, psm: int) -> bool:
        text = _tesseract(arr, psm, page_number)
        candidates.append(text)
        return is_good_text(text)

    # 1) imagem crua: resolve a maioria dos escaneados limpos
    if attempt(gray, 3):
        return candidates[-1]

    # 2) binarização + deskew, com orientação pelo OSD e fallback em força bruta
    clean = _deskew(_binarize(gray))
    rotation = _osd_rotation(gray)
    order = [0, 90, 270, 180]
    if rotation:
        order = [rotation, (360 - rotation) % 360] + order
    order = list(dict.fromkeys(order))

    for i, angle in enumerate(order):
        arr = _rotate(clean, angle)
        if attempt(arr, 3):
            break
        if i == 0 and attempt(arr, 6):  # psm 6: bloco único (formulários, colunas quebradas)
            break

    return max(candidates, key=text_score) if candidates else ""


# ---- Página --------------------------------------------------------------- #

def detect_autos_page_number(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:8] + lines[-8:]:
        if len(line) > 60:
            continue
        match = re.search(r"(?i)\b(?:fls?|folhas?)\.?\s*[:\-]?\s*(\d{1,6})\b", line)
        if match:
            return match.group(1).zfill(5)
    return None


def process_page(pdf_path: str, page_number: int) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {"page": page_number, "text": "", "warning": None, "autos": None}
    try:
        page = _get_doc(pdf_path)[page_number - 1]

        native = _native_text(page)
        if not is_good_text(native):
            alt = _pypdf_text(pdf_path, page_number)
            if text_score(alt) > text_score(native):
                native = alt

        result["autos"] = detect_autos_page_number(native)
        native_words, _ = text_metrics(native)
        native_ok = is_good_text(native)
        coverage = _image_coverage(page)

        if native_ok and (native_words >= TRUST_NATIVE_WORDS or coverage < IMAGE_COVER_OCR):
            result["text"] = clean_text(native)
            return result

        if native_words < MIN_WORDS and _is_blank(page):
            result["text"] = BLANK_PAGE_NOTE
            return result

        ocr = _ocr_page(page, page_number)
        result["autos"] = result["autos"] or detect_autos_page_number(ocr)

        primary, secondary = (ocr, native) if text_score(ocr) >= text_score(native) else (native, ocr)
        merged = merge_texts(primary, secondary)

        if is_good_text(merged):
            result["text"] = clean_text(merged)
            return result

        readable = clean_text(keep_readable_lines(merged))
        if readable:
            result["text"] = readable
            result["warning"] = LOW_CONFIDENCE_WARNING
        else:
            result["warning"] = OCR_WARNING
        return result

    except Exception:
        logger.exception("Falha ao processar a página %s", page_number)
        result["warning"] = PAGE_PROCESSING_WARNING
        return result


_EXECUTOR: Optional[ProcessPoolExecutor] = None


def _executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # forkserver: evita deadlock de fork com threads do uvicorn
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=WORKERS, mp_context=multiprocessing.get_context("forkserver")
        )
    return _EXECUTOR


def _reset_executor() -> None:
    global _EXECUTOR
    if _EXECUTOR is not None:
        try:
            _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _EXECUTOR = None


def extract_pages(pdf_path: str, pages: List[int]) -> List[Dict[str, Optional[str]]]:
    futures = []
    try:
        ex = _executor()
        futures = [ex.submit(process_page, pdf_path, p) for p in pages]
    except Exception:
        logger.exception("Pool de processos indisponível; processando localmente")
        _reset_executor()

    results, reset_done = [], False
    for i, page_number in enumerate(pages):
        try:
            if not futures:
                raise RuntimeError("sem pool")
            results.append(futures[i].result())
        except Exception:
            if futures and not reset_done:
                logger.exception("Worker falhou; reprocessando localmente a partir da página %s", page_number)
                _reset_executor()
                reset_done = True
            results.append(process_page(pdf_path, page_number))
    return results


# --------------------------------------------------------------------------- #
# Anonimização
# --------------------------------------------------------------------------- #

_NLP = None
_NLP_FAILED = False
_CNJ = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
_INSTITUTIONAL = re.compile(
    r"(?i)\b(tribunal|minist[ée]rio|vara|ju[íi]zo|fazenda|defensoria|procuradoria|poder|"
    r"estado|munic[íi]pio|comarca|foro|supremo|superior|c[âa]mara|turma|se[çc][ãa]o|cart[óo]rio|"
    r"receita|uni[ãa]o|inss|detran|oab|tjsp|stj|stf)\b"
)


def get_nlp_model():
    global _NLP, _NLP_FAILED
    if _NLP is None and not _NLP_FAILED:
        try:
            import spacy  # import tardio: workers de OCR não carregam o spaCy

            _NLP = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])
        except Exception:
            _NLP_FAILED = True
            logger.exception("spaCy indisponível; usando apenas regex para nomes")
    return _NLP


def mask_regex_patterns(text: str) -> str:
    protected: List[str] = []

    def _protect(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    masked = _CNJ.sub(_protect, text)  # número de processo não é dado pessoal

    masked = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]", masked)
    masked = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]", masked)
    masked = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "[CNPJ]", masked)
    masked = re.sub(r"(?<![\d.\-/])\d{14}(?![\d.\-/])", "[CNPJ]", masked)
    masked = re.sub(r"(?<![\d.\-/])\d{11}(?![\d.\-/])", "[CPF]", masked)
    masked = re.sub(
        r"(?<![\w.\-/])(?:\+?55[\s\-]?)?(?:\(?\d{2}\)?[\s\-]?)?9?\d{4}[\s\-]?\d{4}(?![\w.\-/])",
        "[TELEFONE]",
        masked,
    )

    masked = re.sub(
        r"(?i)\b(RG\s*(?:n[ºo°.]?\s*)?[:\-]?\s*)([0-9][0-9A-Za-z.\-]{4,19})",
        r"\1[RG]",
        masked,
    )
    masked = re.sub(
        r"(?i)\b(IE|INSCRI[ÇC][AÃ]O\s+ESTADUAL)\s*(?:n[ºo°.]?\s*)?[:\-]?\s*([0-9][0-9A-Za-z.\-/]{5,24})",
        r"\1 [IE]",
        masked,
    )

    masked = re.sub(r"\b[A-Z]{3}-?\d{4}\b", "[PLACA]", masked)
    masked = re.sub(r"\b[A-Z]{3}\d[A-Z]\d{2}\b", "[PLACA]", masked)
    masked = re.sub(
        r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])[A-HJ-NPR-Z0-9]{17}\b",
        "[CHASSI]",
        masked,
    )

    masked = re.sub(
        r"(?i)\b(N[ÚU]MERO\s+SQL|SQL\s+DO\s+IM[ÓO]VEL|INSCRI[ÇC][AÃ]O\s+IMOBILI[ÁA]RIA|MATR[ÍI]CULA\s+DO\s+IM[ÓO]VEL)\s*[:\-]?\s*([0-9A-Za-z.\-/]{4,40})",
        r"\1: [DADO_IMOVEL]",
        masked,
    )
    masked = re.sub(
        r"(?im)^(\s*(?:ENDERE[ÇC]O|LOGRADOURO|RUA|AVENIDA|AV\.|TRAVESSA|ALAMEDA)\s*[:\-]\s*)(.+)$",
        r"\1[ENDERECO]",
        masked,
    )
    masked = re.sub(
        r"(?i)\b(ENDERE[ÇC]O\s*[:\-]\s*)(.*?)(?=\s*(?:,\s*(?:CPF|RG|CNPJ|IE|EMPRESA|RAZ[ÃA]O\s+SOCIAL|NOME|PLACA|CHASSI|MATR[ÍI]CULA|N[ÚU]MERO\s+SQL)\b|;|$))",
        r"\1[ENDERECO]",
        masked,
    )
    masked = re.sub(
        r"(?i)\b(?:RUA|AVENIDA|AV\.|TRAVESSA|ALAMEDA|LOGRADOURO)\s+[A-Z0-9À-ÿ\s.,\-]{3,120}?,\s*(?:n[ºo°.]?\s*)?\d+[A-Z0-9\-]*",
        "[ENDERECO]",
        masked,
    )
    masked = re.sub(r"\b\d{5}-\d{3}\b", "[CEP]", masked)

    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], masked)


def mask_labeled_fields(text: str) -> str:
    masked = re.sub(
        r"(?im)\b(NOME(?:\s+COMPLETO)?|PROPRIET[ÁA]RIO|REQUERENTE|REQUERIDO|AUTOR|R[ÉE]U|REPRESENTANTE|EXEQUENTE|EXECUTADO)\s*[:\-]\s*([^\n;]+)",
        r"\1: [NOME_PESSOA]",
        text,
    )
    masked = re.sub(
        r"(?im)\b(EMPRESA|RAZ[ÃA]O\s+SOCIAL|FIRMA)\s*[:\-]\s*([^\n;]+)",
        r"\1: [EMPRESA]",
        masked,
    )
    masked = re.sub(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'\-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ&][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'\-&]*)*\s+(?:LTDA\.?|S/A|S\.A\.|EIRELI|ME|EPP)(?=\W|$)",
        "[EMPRESA]",
        masked,
    )
    return masked


def mask_generic_names(text: str) -> str:
    return re.sub(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{2,}(?:\s+(?:da|de|do|das|dos|e))?(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{2,})+\b",
        "[NOME_PESSOA]",
        text,
    )


def anonymize_text(text: str) -> str:
    anonymized = mask_labeled_fields(mask_regex_patterns(text))

    nlp = get_nlp_model()
    if nlp is None:
        return mask_generic_names(anonymized)

    try:
        doc = nlp(anonymized)
        spans = []
        for ent in doc.ents:
            if "[" in ent.text or "]" in ent.text:
                continue
            if ent.label_ in {"PER", "PERSON"}:
                spans.append((ent.start_char, ent.end_char, "[NOME_PESSOA]"))
            elif ent.label_ == "ORG" and not _INSTITUTIONAL.search(ent.text):
                spans.append((ent.start_char, ent.end_char, "[EMPRESA]"))
        for start, end, repl in sorted(spans, key=lambda s: s[0], reverse=True):
            anonymized = anonymized[:start] + repl + anonymized[end:]
    except Exception:
        logger.exception("Falha no NER; aplicando regex genérica")
        anonymized = mask_generic_names(anonymized)

    return anonymized


# --------------------------------------------------------------------------- #
# LLM local (Ollama /api/generate)
# --------------------------------------------------------------------------- #

LLM_PROMPT = (
    "Corrija somente erros evidentes de OCR no texto abaixo (letras trocadas, palavras partidas, "
    "espaçamento). Não resuma, não omita, não acrescente, não comente. Preserve números, datas e "
    "marcadores entre colchetes. Devolva apenas o texto corrigido.\n\n"
)


def improve_with_local_llm(text: str) -> str:
    llm_url = os.getenv("LLM_URL", "").strip()
    llm_model = os.getenv("LLM_MODEL", "").strip()
    if not llm_url or not llm_model or not text.strip():
        return text

    payload = {
        "model": llm_model,
        "prompt": LLM_PROMPT + text,
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            response = client.post(llm_url, json=payload)
            response.raise_for_status()
            out = (response.json().get("response") or "").strip()
    except Exception:
        logger.exception("Falha na chamada ao LLM")
        return text

    # Rejeita se o modelo resumiu/cortou conteúdo
    if len(out) < 0.8 * len(text.strip()):
        logger.warning("Saída do LLM descartada (encurtou o texto)")
        return text
    return out


# --------------------------------------------------------------------------- #
# Montagem da saída
# --------------------------------------------------------------------------- #

def build_output_filename(original_name: str, page_start: int, page_end: int) -> str:
    stem = Path(original_name or "arquivo").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "arquivo"
    return f"{safe_stem}de{page_start:05d}a{page_end:05d}.md"


def build_page_block(page_number: int, text: str, warning: Optional[str] = None,
                     autos_num: Optional[str] = None) -> str:
    physical_num_str = f"{page_number:05d}"
    if autos_num is None:
        autos_num = detect_autos_page_number(text)

    if autos_num:
        if autos_num != physical_num_str:
            header = f"[PAGINA_AUTOS:{autos_num} ou PAGINA_FISICA:{physical_num_str}]"
        else:
            header = f"[PAGINA_AUTOS:{autos_num}]"
    else:
        header = f"[PAGINA_FISICA:{physical_num_str}]"

    parts = [header]
    if warning:
        parts.extend(["", warning])
    if text:
        parts.extend(["", text.strip()])
    return "\n".join(parts).strip()


def build_summary_warning_section(failed_pages: List[Dict[str, Optional[str]]]) -> str:
    if not failed_pages:
        return ""

    lines = [
        "---",
        "## ⚠️ RELATÓRIO DE PÁGINAS COM CONTEÚDO OMITIDO OU ILEGÍVEL",
        f"**Total de ocorrências com texto não exibido:** {len(failed_pages)}",
        "",
        "### Páginas afetadas:",
    ]
    for item in failed_pages:
        p_fisica = item["physical_page"]
        p_autos = item.get("autos_page")
        detail = f" — {item['kind']}" if item.get("kind") else ""
        if p_autos:
            lines.append(f"- **Página Física:** {p_fisica} (Página nos Autos: {p_autos}){detail}")
        else:
            lines.append(f"- **Página Física:** {p_fisica}{detail}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def _save_upload(file: UploadFile) -> str:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        size, head = 0, b""
        while chunk := file.file.read(1 << 20):
            if not head:
                head = chunk[:1024]
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail=f"Arquivo excede {MAX_UPLOAD_MB} MB.")
            tmp.write(chunk)
        tmp.close()
        if b"%PDF" not in head:
            raise HTTPException(status_code=415, detail="Arquivo inválido. Envie um PDF.")
        return tmp.name
    except BaseException:
        tmp.close()
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        raise


@app.on_event("shutdown")
def _shutdown() -> None:
    _reset_executor()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))


@app.get("/health", response_class=PlainTextResponse)
def health() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.post("/process", response_class=PlainTextResponse)
def process_pdf(
    file: UploadFile = File(...),
    anon: bool = Form(False),
    llm: bool = Form(False),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
) -> PlainTextResponse:
    filename = file.filename or "arquivo.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Arquivo inválido. Envie um PDF.")

    output_name = build_output_filename(filename, 1, 1)
    tmp_path: Optional[str] = None

    try:
        tmp_path = _save_upload(file)

        with pymupdf.open(tmp_path) as doc:
            if doc.needs_pass and not doc.authenticate(""):
                raise HTTPException(status_code=400, detail="PDF protegido por senha.")
            page_count = doc.page_count

        resolved_page_start = page_start if page_start is not None else 1
        resolved_page_end = page_end if page_end is not None else page_count

        if resolved_page_start < 1 or resolved_page_end < 1 or resolved_page_start > resolved_page_end:
            raise HTTPException(status_code=400, detail="Intervalo de páginas inválido.")
        if resolved_page_end > page_count:
            raise HTTPException(status_code=400, detail=f"O PDF possui apenas {page_count} página(s).")

        output_name = build_output_filename(filename, resolved_page_start, resolved_page_end)
        pages = list(range(resolved_page_start, resolved_page_end + 1))
        results = extract_pages(tmp_path, pages)

        page_blocks, failed_pages = [], []
        for res in results:
            page_number = res["page"]
            page_text = res["text"] or ""
            warning = res["warning"]
            autos_num = res["autos"]

            if warning:
                failed_pages.append({
                    "physical_page": f"{page_number:05d}",
                    "autos_page": autos_num,
                    "kind": "baixa confiança" if warning == LOW_CONFIDENCE_WARNING else "sem texto",
                })

            if page_text and page_text != BLANK_PAGE_NOTE:
                if anon:
                    page_text = anonymize_text(page_text)
                if llm:
                    page_text = improve_with_local_llm(page_text)

            page_blocks.append(build_page_block(page_number, page_text, warning, autos_num or ""))

        cleaned = "\n\n".join(block for block in page_blocks if block).strip()
        summary_section = build_summary_warning_section(failed_pages)
        if summary_section:
            cleaned = f"{cleaned}\n\n{summary_section}"

        return PlainTextResponse(
            cleaned,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falha no processamento do PDF: %s", exc)
        return PlainTextResponse(
            f"{OCR_WARNING}\n\nNão foi possível processar o PDF.",
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
