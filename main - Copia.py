import os
import re
import tempfile
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import httpx
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pypdf import PdfReader, PdfWriter

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


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

_NLP = None


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


MIN_NATIVE_TEXT_CHARS = _env_int("MIN_NATIVE_TEXT_CHARS", 25)
MIN_EXTRACTED_TEXT_CHARS = _env_int("MIN_EXTRACTED_TEXT_CHARS", 80)
MIN_ALPHA_RATIO = _env_float("MIN_ALPHA_RATIO", 0.30)


def build_output_filename(original_name: str, page_start: int, page_end: int) -> str:
    stem = Path(original_name or "arquivo").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "arquivo"
    return f"{safe_stem}de{page_start:05d}a{page_end:05d}.md"


def is_corrupted_pdf_text(text: str) -> bool:
    """
    Verifica se o texto extraído da camada nativa contém lixo de codificação de fontes
    (ex: '/0 /1 /2', '/C45/C78', excesso de quadros '□', etc.).
    """
    if not text:
        return False

    slash_digits = len(re.findall(r"/\d+", text))
    slash_c_hex = len(re.findall(r"/C[0-9a-fA-F]{2}", text))
    square_symbols = text.count("□") + text.count("\ufffd")

    total_corrupted_tokens = slash_digits + slash_c_hex + square_symbols

    if total_corrupted_tokens > 5:
        return True

    return False


def extract_valid_human_sentences(text: str) -> str:
    """
    Filtra um texto corrompido extraindo apenas as linhas ou frases
    que sejam perfeitamente legíveis para humanos (ex: avisos do TJSP, assinaturas digitais).
    """
    if not text:
        return ""

    lines = text.split("\n")
    valid_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_corrupted_pdf_text(stripped):
            continue

        compact = re.sub(r"\s+", "", stripped)
        if len(compact) < 15:
            continue

        letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", compact)
        alpha_ratio = len(letters) / max(len(compact), 1)

        if alpha_ratio >= 0.50:
            valid_lines.append(stripped)

    return "\n".join(valid_lines)


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [line.strip() for line in text.split("\n")]

    filtered = []
    for line in raw_lines:
        if not line:
            filtered.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"(?i)page\s+\d+(\s+of\s+\d+)?", line):
            continue
        if re.fullmatch(r"(?i)p[áa]gina\s+\d+(\s+de\s+\d+)?", line):
            continue
        filtered.append(line)

    non_empty = [line for line in filtered if line]
    counts = Counter(non_empty)
    repeated_short = {
        line for line, count in counts.items() if count > 1 and len(line) <= 80
    }

    cleaned = []
    for line in filtered:
        if line in repeated_short:
            continue
        cleaned.append(line)

    collapsed = []
    last_blank = False
    for line in cleaned:
        blank = line == ""
        if blank and last_blank:
            continue
        collapsed.append(line)
        last_blank = blank

    return "\n".join(collapsed).strip()


def is_text_sufficient(text: str, min_chars: int) -> bool:
    if is_corrupted_pdf_text(text):
        return False

    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < min_chars:
        return False

    letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", compact)
    alpha_ratio = len(letters) / max(len(compact), 1)
    return alpha_ratio >= MIN_ALPHA_RATIO


def get_nlp_model():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("pt_core_news_sm")
    return _NLP


def mask_regex_patterns(text: str) -> str:
    masked = text

    masked = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", masked)
    masked = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "[TELEFONE]", masked)

    masked = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]", masked)
    masked = re.sub(r"\b\d{11}\b", "[CPF]", masked)
    masked = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "[CNPJ]", masked)
    masked = re.sub(r"\b\d{14}\b", "[CNPJ]", masked)

    masked = re.sub(
        r"(?i)\b(RG\s*(?:n[ºo°.]?\s*)?[:\-]?\s*)([0-9A-Za-z\.\-]{5,20})",
        r"\1[RG]",
        masked,
    )
    masked = re.sub(
        r"(?i)\b(IE|INSCRI[ÇC][AÃ]O\s+ESTADUAL)\s*(?:n[ºo°.]?\s*)?[:\-]?\s*([0-9A-Za-z\.\-/]{6,25})",
        r"\1 [IE]",
        masked,
    )

    masked = re.sub(r"\b[A-Z]{3}-?\d{4}\b", "[PLACA]", masked)
    masked = re.sub(r"\b[A-Z]{3}\d[A-Z0-9]\d{2}\b", "[PLACA]", masked)
    masked = re.sub(r"\b[A-HJ-NPR-Z0-9]{17}\b", "[CHASSI]", masked)

    masked = re.sub(
        r"(?i)\b(N[ÚU]MERO\s+SQL|SQL\s+DO\s+IM[ÓO]VEL|INSCRI[ÇC][AÃ]O\s+IMOBILI[ÁA]RIA|MATR[ÍI]CULA\s+DO\s+IM[ÓO]VEL)\s*[:\-]?\s*([0-9A-Za-z\.\-/]{4,40})",
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
        r"(?i)\b(?:RUA|AVENIDA|AV\.|TRAVESSA|ALAMEDA|LOGRADOURO)\s+[A-Z0-9\s\.,\-]{3,120},\s*\d+[A-Z0-9\-]*",
        "[ENDERECO]",
        masked,
    )

    return masked


def mask_name_and_company_fallback(text: str) -> str:
    masked = text

    masked = re.sub(
        r"(?im)\b(NOME(?:\s+COMPLETO)?|PROPRIET[ÁA]RIO|REQUERENTE|AUTOR|R[ÉE]U|REPRESENTANTE)\s*[:\-]\s*([^\n;]+)",
        r"\1: [NOME_PESSOA]",
        masked,
    )
    masked = re.sub(
        r"(?im)\b(EMPRESA|RAZ[ÃA]O\s+SOCIAL|FIRMA)\s*[:\-]\s*([^\n;]+)",
        r"\1: [EMPRESA]",
        masked,
    )
    masked = re.sub(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'\-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'\-]+)*\s+(?:LTDA\.?|S/A|SA|EIRELI|MEI|ME)\b",
        "[EMPRESA]",
        masked,
    )
    masked = re.sub(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{2,}(?:\s+(?:da|de|do|das|dos|e))?(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{2,})+\b",
        "[NOME_PESSOA]",
        masked,
    )

    return masked


def anonymize_text(text: str) -> str:
    anonymized = mask_regex_patterns(text)
    entities_found = False

    try:
        nlp = get_nlp_model()
        doc = nlp(anonymized)
        entities = sorted(
            [
                (ent.start_char, ent.end_char, ent.label_)
                for ent in doc.ents
                if ent.label_ in {"PER", "PERSON", "ORG", "GPE", "LOC"}
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        entities_found = bool(entities)
        for start, end, label in entities:
            if label in {"PER", "PERSON"}:
                replacement = "[NOME_PESSOA]"
            elif label == "ORG":
                replacement = "[EMPRESA]"
            else:
                replacement = "[LOCAL]"
            anonymized = anonymized[:start] + replacement + anonymized[end:]
    except Exception:
        pass

    if not entities_found:
        anonymized = mask_name_and_company_fallback(anonymized)

    return anonymized


def improve_with_local_llm(text: str) -> str:
    llm_url = os.getenv("LLM_URL", "").strip()
    llm_model = os.getenv("LLM_MODEL", "").strip()

    if not llm_url or not llm_model:
        return text

    payload = {"model": llm_model, "prompt": text}

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(llm_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", text)
    except Exception:
        return text


def extract_markdown_with_docling(pdf_path: str, do_ocr: bool) -> str:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    if do_ocr:
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["por", "eng"])

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def get_pdf_page_count(pdf_path: str) -> int:
    return len(PdfReader(pdf_path).pages)


def write_single_page_pdf(pdf_path: str, page_number: int) -> str:
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        writer.write(tmp)
        return tmp.name


def extract_native_text_with_pypdf(pdf_path: str, page_number: int) -> str:
    try:
        reader = PdfReader(pdf_path)
        page = reader.pages[page_number - 1]
        return (page.extract_text() or "").strip()
    except Exception as exc:
        logger.exception("Falha na extração nativa (pypdf) da página %s: %s", page_number, exc)
        return ""


def extract_page_text(pdf_path: str, page_number: int) -> Tuple[str, Optional[str]]:
    page_pdf_path = write_single_page_pdf(pdf_path, page_number)

    try:
        raw_native_text = extract_native_text_with_pypdf(pdf_path, page_number)
        native_text = clean_markdown(raw_native_text)

        if is_text_sufficient(native_text, MIN_NATIVE_TEXT_CHARS):
            return native_text, None

        extracted_text = ""
        try:
            extracted_text = clean_markdown(extract_markdown_with_docling(page_pdf_path, do_ocr=False))
        except Exception as exc:
            logger.exception("Falha na extração sem OCR na página %s: %s", page_number, exc)

        if is_text_sufficient(extracted_text, MIN_EXTRACTED_TEXT_CHARS):
            return extracted_text, None

        ocr_text = ""
        try:
            ocr_text = clean_markdown(extract_markdown_with_docling(page_pdf_path, do_ocr=True))
        except Exception as exc:
            logger.exception("Falha na extração com OCR na página %s: %s", page_number, exc)

        if is_text_sufficient(ocr_text, MIN_NATIVE_TEXT_CHARS):
            return ocr_text, None

        human_text = extract_valid_human_sentences(raw_native_text) or extract_valid_human_sentences(ocr_text)

        if human_text:
            return human_text, OCR_WARNING

        return "", OCR_WARNING
    except Exception:
        return "", PAGE_PROCESSING_WARNING
    finally:
        try:
            os.remove(page_pdf_path)
        except Exception:
            pass


def detect_autos_page_number(text: str) -> Optional[str]:
    if not text:
        return None

    pattern = r"(?i)\b(?:fls?\.?|folhas?|p[áa]g\.?|p[áa]ginas?)\s*[:\-]?\s*(\d+[a-zA-Z]?)\b"
    match = re.search(pattern, text)
    if match:
        val = match.group(1).zfill(5)
        return val
    return None


def build_page_block(page_number: int, text: str, warning: Optional[str] = None) -> str:
    autos_num = detect_autos_page_number(text)
    physical_num_str = f"{page_number:05d}"

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


def build_summary_warning_section(failed_pages: List[Dict[str, str]]) -> str:
    """
    Constrói a seção de aviso final com a contagem total de ocorrências não
    exibidas/processadas completamente e a lista das páginas correspondentes.
    """
    if not failed_pages:
        return ""

    total_occurrences = len(failed_pages)
    lines = [
        "---",
        "## ⚠️ RELATÓRIO DE PÁGINAS COM CONTEÚDO OMITIDO OU ILEGÍVEL",
        f"**Total de ocorrências com texto não exibido:** {total_occurrences}",
        "",
        "### Páginas afetadas:",
    ]

    for item in failed_pages:
        p_fisica = item["physical_page"]
        p_autos = item.get("autos_page")
        if p_autos:
            lines.append(f"- **Página Física:** {p_fisica} (Página nos Autos: {p_autos})")
        else:
            lines.append(f"- **Página Física:** {p_fisica}")

    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))


@app.post("/process", response_class=PlainTextResponse)
async def process_pdf(
    file: UploadFile = File(...),
    anon: bool = Form(False),
    llm: bool = Form(False),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
) -> PlainTextResponse:
    if not file.filename.lower().endswith(".pdf"):
        return PlainTextResponse("Arquivo inválido. Envie um PDF.", media_type="text/markdown")

    output_name = build_output_filename(file.filename or "arquivo.pdf", 1, 1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        page_count = get_pdf_page_count(tmp_path)
        resolved_page_start = page_start if page_start is not None else 1
        resolved_page_end = page_end if page_end is not None else page_count

        if resolved_page_start < 1 or resolved_page_end < 1 or resolved_page_start > resolved_page_end:
            raise HTTPException(status_code=400, detail="Intervalo de páginas inválido.")

        if resolved_page_end > page_count:
            raise HTTPException(status_code=400, detail=f"O PDF possui apenas {page_count} página(s).")

        output_name = build_output_filename(file.filename, resolved_page_start, resolved_page_end)
        page_blocks = []
        failed_pages = []

        for page_number in range(resolved_page_start, resolved_page_end + 1):
            page_text, warning = extract_page_text(tmp_path, page_number)

            if warning:
                autos_num = detect_autos_page_number(page_text)
                failed_pages.append({
                    "physical_page": f"{page_number:05d}",
                    "autos_page": autos_num
                })

            if page_text and anon:
                page_text = anonymize_text(page_text)

            if page_text and llm:
                page_text = improve_with_local_llm(page_text)

            page_blocks.append(build_page_block(page_number, page_text, warning))

        cleaned = "\n\n".join(block for block in page_blocks if block).strip()

        # Adiciona o resumo de avisos/páginas ilegíveis ao final do arquivo
        summary_section = build_summary_warning_section(failed_pages)
        if summary_section:
            cleaned = f"{cleaned}\n\n{summary_section}"

        return PlainTextResponse(
            cleaned,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
        )
    except Exception as exc:
        logger.exception("Falha no processamento do PDF: %s", exc)
        return PlainTextResponse(
            f"{OCR_WARNING}\n\nNão foi possível processar o PDF.",
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


        