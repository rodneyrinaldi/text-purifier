"""Anonimização de dados pessoais (regex + NER spaCy opcional)."""

import logging
import re
from typing import List

from . import config

logger = logging.getLogger("text_purifier.anon")
SPACY_MODEL = config.SPACY_MODEL

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
