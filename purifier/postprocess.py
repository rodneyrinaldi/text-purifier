"""Pós-processamento entre páginas: cabeçalhos/rodapés repetidos e sequência de folhas dos autos."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .quality import token_set

EDGE_LINES = 3


def _norm(line: str) -> str:
    line = re.sub(r"\d+", "#", line.lower())
    return re.sub(r"\s+", " ", line).strip(" .-–—|")


def remove_repeated_edges(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove linhas repetidas no topo/rodapé de várias páginas, mantendo a 1ª ocorrência.

    Retorna a lista de padrões removidos (para o relatório de processamento).
    """
    candidates = [p for p in pages if p.get("text") and not p.get("blank")]
    if len(candidates) < 3:
        return []
    threshold = max(3, int(0.5 * len(candidates) + 0.5))

    counts: Counter = Counter()
    example: Dict[str, str] = {}
    for p in candidates:
        lines = [l for l in p["text"].split("\n") if l.strip()]
        edge = set()
        for l in lines[:EDGE_LINES] + lines[-EDGE_LINES:]:
            key = _norm(l)
            if len(key) >= 4:
                edge.add(key)
                example.setdefault(key, l.strip())
        counts.update(edge)

    repeated = {k for k, c in counts.items() if c >= threshold and len(k) <= 300}
    if not repeated:
        return []

    # Proteção: páginas quase idênticas (mesmo documento juntado mais de uma vez) não são
    # tratadas como cabeçalho/rodapé — as linhas repetidas ali fazem parte do conteúdo.
    tokens = {id(p): token_set(p["text"]) for p in candidates}
    duplicate_pages = set()
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:i + 40]:
            ta, tb = tokens[id(a)], tokens[id(b)]
            if len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.85:
                duplicate_pages.update((id(a), id(b)))

    removed: Dict[str, int] = defaultdict(int)
    first_seen: set = set()
    for p in candidates:
        if id(p) in duplicate_pages:
            continue
        lines = p["text"].split("\n")
        non_empty = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(non_empty[:EDGE_LINES] + non_empty[-EDGE_LINES:])
        keep = []
        for i, l in enumerate(lines):
            key = _norm(l)
            if i in edge_idx and key in repeated:
                if key not in first_seen:
                    first_seen.add(key)
                    keep.append(l)
                else:
                    removed[key] += 1
                continue
            keep.append(l)
        p["text"] = "\n".join(keep).strip()

    return [
        {"example": example[k], "pages": counts[k], "removed": removed[k]}
        for k in sorted(removed, key=lambda k: -removed[k])
    ]


def numbering_gaps(pages: List[Dict[str, Any]]) -> List[str]:
    """Detecta saltos/inversões na numeração de folhas dos autos (possíveis folhas ausentes)."""
    seq: List[Tuple[int, int]] = [(p["page"], int(p["autos"])) for p in pages if p.get("autos")]
    notes: List[str] = []
    for (pa, fa), (pb, fb) in zip(seq, seq[1:]):
        expected = fa + (pb - pa)
        if fb == expected:
            continue
        if fb > expected:
            notes.append(f"Entre a página física {pa:05d} (fls. {fa}) e {pb:05d} (fls. {fb}) há salto de "
                         f"{fb - expected} folha(s) na numeração: possíveis folhas ausentes no arquivo.")
        elif fb <= fa:
            notes.append(f"Numeração de folhas não crescente: página física {pa:05d} (fls. {fa}) → "
                         f"{pb:05d} (fls. {fb}). Possível juntada fora de ordem ou erro de leitura.")
    return notes
