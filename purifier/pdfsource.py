"""Abertura robusta do PDF: senha, restrições, reparo, descriptografia e inventário."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

logger = logging.getLogger("text_purifier.pdf")
logging.getLogger("pypdf").setLevel(logging.ERROR)

_PERMS = [
    ("PDF_PERM_PRINT", "imprimir"),
    ("PDF_PERM_MODIFY", "modificar"),
    ("PDF_PERM_COPY", "copiar/extrair texto"),
    ("PDF_PERM_ANNOTATE", "anotar"),
    ("PDF_PERM_FORM", "preencher formulários"),
    ("PDF_PERM_ACCESSIBILITY", "extração p/ acessibilidade"),
    ("PDF_PERM_ASSEMBLE", "montar/reorganizar"),
    ("PDF_PERM_PRINT_HQ", "impressão em alta qualidade"),
]


class PdfOpenError(Exception):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_pymupdf(path: str, password: str, info: Dict[str, Any]) -> Optional["pymupdf.Document"]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        info["errors"].append(f"PyMuPDF não conseguiu abrir o arquivo ({type(exc).__name__}: {exc}).")
        return None
    if doc.needs_pass:
        info["needs_password"] = True
        ok = False
        for candidate in ([password] if password else []) + [""]:
            if doc.authenticate(candidate):
                ok = True
                info["password_used"] = bool(candidate)
                break
        if not ok:
            doc.close()
            info["errors"].append("PyMuPDF: senha de abertura necessária e não fornecida/inválida.")
            return None
    return doc


def _decrypted_copy(path: str, password: str, info: Dict[str, Any]) -> Optional[str]:
    """Gera cópia sem criptografia/restrições e com estrutura reparada (qpdf via pikepdf)."""
    try:
        import pikepdf
    except Exception:
        info["warnings"].append("pikepdf indisponível: não foi gerada cópia descriptografada/reparada.")
        return None
    for candidate in ([password] if password else []) + [""]:
        try:
            with pikepdf.open(path, password=candidate) as pdf:
                fd, out = tempfile.mkstemp(suffix=".work.pdf")
                os.close(fd)
                pdf.save(out)  # sem parâmetro 'encryption' => criptografia removida
                info["page_counts"]["pikepdf/qpdf"] = len(pdf.pages)
                if candidate:
                    info["password_used"] = True
                return out
        except pikepdf.PasswordError:
            continue
        except Exception as exc:
            info["warnings"].append(f"qpdf/pikepdf não conseguiu reparar/descriptografar ({type(exc).__name__}: {exc}).")
            return None
    info["warnings"].append("qpdf/pikepdf: senha de abertura necessária.")
    return None


def prepare_pdf(path: str, password: str = "") -> Dict[str, Any]:
    """Abre, valida e inventaria o PDF. Retorna dicionário serializável (enviado aos workers)."""
    info: Dict[str, Any] = {
        "original_path": path,
        "work_path": path,
        "pdfium_path": path,
        "temp_files": [],
        "size_bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "page_count": 0,
        "page_counts": {},
        "pymupdf_pages": 0,
        "encrypted": False,
        "encryption": None,
        "needs_password": False,
        "password_used": False,
        "restrictions": [],
        "metadata": {},
        "pdf_version": None,
        "repaired": False,
        "decrypted_copy": False,
        "signature_fields": 0,
        "signed": False,
        "embedded_files": [],
        "has_forms": False,
        "warnings": [],
        "errors": [],
    }

    doc = _try_pymupdf(path, password, info)
    if doc is not None:
        info["encrypted"] = bool(doc.is_encrypted or (doc.metadata or {}).get("encryption"))
        info["encryption"] = (doc.metadata or {}).get("encryption")
        info["repaired"] = bool(getattr(doc, "is_repaired", False))

    need_copy = doc is None or info["encrypted"] or info["needs_password"] or info["repaired"]
    if need_copy:
        copy = _decrypted_copy(path, password, info)
        if copy:
            info["temp_files"].append(copy)
            info["decrypted_copy"] = True
            info["pdfium_path"] = copy
            if doc is None:
                doc = _try_pymupdf(copy, "", info)
                if doc is not None:
                    info["work_path"] = copy
                    info["warnings"].append("Arquivo original danificado: usada cópia reparada pelo qpdf.")

    if doc is not None:
        try:
            meta = doc.metadata or {}
            info["pdf_version"] = meta.get("format")
            info["metadata"] = {
                k: v for k, v in meta.items()
                if v and k in {"title", "author", "subject", "keywords", "creator", "producer",
                               "creationDate", "modDate"}
            }
            perms = doc.permissions
            if info["encrypted"] and perms is not None and perms >= 0:
                for attr, label in _PERMS:
                    flag = getattr(pymupdf, attr, None)
                    if flag is not None and not (perms & flag):
                        info["restrictions"].append(label)
            info["pymupdf_pages"] = doc.page_count
            info["page_counts"]["PyMuPDF"] = doc.page_count
            try:
                info["has_forms"] = bool(doc.is_form_pdf)
                info["signed"] = doc.get_sigflags() > 0
            except Exception:
                pass
            try:
                sig_type = getattr(pymupdf, "PDF_WIDGET_TYPE_SIGNATURE", 6)
                count = 0
                for page in doc:
                    for widget in page.widgets() or []:
                        if widget.field_type == sig_type:
                            count += 1
                info["signature_fields"] = count
            except Exception:
                pass
            try:
                info["embedded_files"] = list(doc.embfile_names())
            except Exception:
                pass
        finally:
            doc.close()

    # PDFium (motor do Chrome): segunda opinião de contagem e renderização
    try:
        import pypdfium2 as pdfium

        pdoc = pdfium.PdfDocument(info["pdfium_path"], password=None if info["decrypted_copy"] else (password or None))
        info["page_counts"]["PDFium"] = len(pdoc)
        pdoc.close()
    except Exception as exc:
        info["warnings"].append(f"PDFium não abriu o arquivo ({type(exc).__name__}).")

    try:
        from pypdf import PdfReader

        reader = PdfReader(info["pdfium_path"], strict=False)
        if reader.is_encrypted:
            reader.decrypt(password or "")
        info["page_counts"]["pypdf"] = len(reader.pages)
    except Exception as exc:
        info["warnings"].append(f"pypdf não abriu o arquivo ({type(exc).__name__}).")

    renderable = [info["page_counts"].get("PyMuPDF", 0), info["page_counts"].get("PDFium", 0)]
    info["page_count"] = max(renderable)
    if info["page_count"] == 0:
        cleanup(info)
        if info["needs_password"] and not info["password_used"]:
            raise PdfOpenError("PDF protegido por senha de abertura. Informe a senha para processar.")
        raise PdfOpenError("Não foi possível abrir o PDF com nenhum dos motores disponíveis: "
                           + " | ".join(info["errors"] + info["warnings"]))

    counts = set(v for v in info["page_counts"].values() if v)
    if len(counts) > 1:
        info["warnings"].append(
            "Divergência na contagem de páginas entre motores: "
            + ", ".join(f"{k}={v}" for k, v in info["page_counts"].items())
            + f". Processadas {info['page_count']} páginas (maior contagem renderizável)."
        )
    if info["embedded_files"]:
        info["warnings"].append(
            f"O PDF contém {len(info['embedded_files'])} arquivo(s) anexado(s) internamente que NÃO "
            f"fazem parte das páginas e não foram processados: {', '.join(info['embedded_files'][:20])}"
        )
    return info


def cleanup(info: Dict[str, Any]) -> None:
    for tmp in info.get("temp_files", []):
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
