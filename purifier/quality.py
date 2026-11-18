"""Métricas de qualidade de texto, limpeza e comparação entre leituras."""

import re
import unicodedata
from typing import List, Optional, Set, Tuple

from . import config

_BAD_CHARS = re.compile(r"[�□\x00-\x08\x0b\x0c\x0e-\x1f-]")
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
    """(palavras válidas, razão de qualidade 0..1)."""
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

    # Fonte "cifrada" (sem mapa Unicode) gera tokens com cara de palavra, mas sem stopwords
    if words >= 40 and stops / words < 0.04:
        quality *= 0.5

    return words, quality


def text_score(text: str) -> float:
    words, quality = text_metrics(text)
    return words * quality


def is_good_text(text: str, min_words: int = config.MIN_WORDS) -> bool:
    words, quality = text_metrics(text)
    return words >= min_words and quality >= config.MIN_QUALITY


def keep_readable_lines(text: str) -> str:
    kept = []
    for line in (text or "").split("\n"):
        words, quality = text_metrics(line)
        if words >= 2 and quality >= 0.5:
            kept.append(line.strip())
    return "\n".join(kept)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def token_set(text: str) -> Set[str]:
    """Conjunto normalizado de palavras (>= 3 letras) para comparação entre leituras."""
    return {
        _strip_accents(w.lower())
        for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", text or "")
    }


def agreement(a: str, b: str) -> Optional[float]:
    """Concordância entre duas leituras independentes (0..1); None se não comparável."""
    ta, tb = token_set(a), token_set(b)
    if len(ta) < 8 or len(tb) < 8:
        return None
    return len(ta & tb) / min(len(ta), len(tb))


_PAGE_LINE = re.compile(r"(?i)(?:p[áa]g(?:ina)?\.?|page)\s*:?\s*\d+(?:\s*(?:de|of|/)\s*\d+)?")


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

    out: List[str] = []
    last_blank = False
    for line in lines:
        blank = not line
        if blank and last_blank:
            continue
        out.append(line)
        last_blank = blank
    return "\n".join(out).strip()


def merge_texts(primary: str, secondary: str, min_novelty: float = 0.7) -> str:
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
        if overlap < min_novelty:
            extra.append(line.strip())

    return f"{primary}\n\n" + "\n".join(extra) if extra else primary


_AUTOS = re.compile(r"(?i)\b(?:f[li1|]s?|folhas?)\.?\s*(?:n[ºo°.]?\s*)?[:\-]?\s*(\d{1,6})\b")
_AUTOS_EOL = re.compile(r"(?i)\b(?:f[li1|]s?|folhas?)\.?\s*[:\-]?\s*(\d{1,6})\s*\.?\s*$")


def detect_autos_page_number(text: str) -> Optional[str]:
    """Número da folha nos autos ("fls. 123"), procurado nas bordas e depois no restante da página."""
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    edges = lines[:8] + lines[-8:]
    # 1) linhas curtas nas bordas (carimbo/numeração da folha)
    for line in edges:
        if len(line) <= 60:
            match = _AUTOS.search(line)
            if match and int(match.group(1)) > 0:
                return match.group(1).zfill(5)
    # 2) tarja lateral de assinatura digital (".... fls. 12" no fim da linha)
    for line in edges:
        match = _AUTOS_EOL.search(line)
        if match and int(match.group(1)) > 0:
            return match.group(1).zfill(5)
    return None


def repetition_ratio(text: str) -> float:
    """Proporção de linhas repetidas (detecta laços de repetição de modelos de IA)."""
    lines = [l.strip() for l in (text or "").split("\n") if len(l.strip()) > 3]
    if len(lines) < 6:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def health_level(health: int) -> str:
    if health >= 90:
        return "ALTA"
    if health >= config.HEALTH_WARN:
        return "MÉDIA"
    if health >= config.HEALTH_CRITICAL:
        return "BAIXA"
    return "CRÍTICA"
