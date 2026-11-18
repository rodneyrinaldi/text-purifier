"""Status do processamento e estimativa de tempo restante.

O `Tracker` é o estado compartilhado entre o pipeline (que o atualiza) e a API (que
consulta `snapshot()` a cada segundo). A estimativa combina:

* custo médio por unidade aprendido em execuções anteriores (média móvel, por processo);
* custo medido na execução atual (prevalece assim que há medições);
* para a IA, a velocidade real de geração (tokens/s) da página em andamento;
* antes de saber quantas páginas irão para a IA, a proporção observada nas páginas já lidas.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import config

STAGES = [
    ("prepare", "Abrir e validar o PDF"),
    ("ai_connect", "Conectar à IA local"),
    ("read", "Ler páginas (texto nativo + OCR)"),
    ("vision", "IA: transcrever páginas comprometidas"),
    ("text", "IA: revisar erros de OCR"),
    ("post", "Remover cabeçalhos/rodapés repetidos"),
    ("anon", "Anonimizar dados pessoais"),
    ("build", "Montar documento e relatório"),
]
_LABELS = dict(STAGES)

# Custo inicial por unidade (segundos de relógio) antes de qualquer medição.
# read: por página, já considerando os workers em paralelo.
_RATES: Dict[str, float] = {"read": 6.0 / max(1, config.WORKERS) + 0.5, "vision": 90.0, "text": 20.0,
                            "anon": 0.3, "prepare": 2.0, "post": 0.5, "build": 1.0, "ai_connect": 1.0}
_RATES_LOCK = threading.Lock()
_ALPHA = 0.35  # peso da medição nova na média móvel entre execuções


def _learn(key: str, seconds: float) -> None:
    if seconds <= 0:
        return
    with _RATES_LOCK:
        _RATES[key] = (1 - _ALPHA) * _RATES.get(key, seconds) + _ALPHA * seconds


def _rate(key: str) -> float:
    with _RATES_LOCK:
        return _RATES.get(key, 1.0)


class Tracker:
    def __init__(self, options: Optional[Dict[str, Any]] = None,
                 on_change: Optional[Callable[["Tracker"], None]] = None) -> None:
        options = options or {}
        self._lock = threading.RLock()
        self.on_change = on_change
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.state = "queued"           # queued | running | done | error
        self.detail = "Aguardando a vez (outro processamento em andamento)"
        self.enabled = {"prepare", "read", "build"}
        if options.get("use_vision") or options.get("use_llm"):
            self.enabled.add("ai_connect")
        if options.get("use_vision"):
            self.enabled.add("vision")
        if options.get("use_llm"):
            self.enabled.add("text")
        if options.get("remove_repeated"):
            self.enabled.add("post")
        if options.get("anon"):
            self.enabled.add("anon")
        self.status = {k: ("pending" if k in self.enabled else "off") for k, _ in STAGES}
        self.notes: Dict[str, str] = {}
        self.t_start: Dict[str, float] = {}
        self.t_end: Dict[str, float] = {}
        self.total: Dict[str, Optional[int]] = {k: None for k, _ in STAGES}
        self.done: Dict[str, int] = {k: 0 for k, _ in STAGES}
        self.unit_times: Dict[str, List[float]] = {"vision": [], "text": []}
        self.unit: Optional[Dict[str, Any]] = None   # unidade de IA em andamento
        self.pages_total: Optional[int] = None
        self.page_first = 1
        self.page_health: Dict[int, int] = {}
        self.read_flags = {"vision": 0, "ocr": 0}
        self.vision_cap = config.LLM_VISION_MAX_PAGES or None

    # ---- atualização (pipeline) --------------------------------------------- #
    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change(self)
            except Exception:
                pass

    def begin(self) -> None:
        with self._lock:
            self.started = time.time()
            self.state = "running"
        self._changed()

    def stage(self, key: str, detail: Optional[str] = None, total: Optional[int] = None) -> None:
        """Inicia uma etapa (encerra a anterior em andamento)."""
        now = time.time()
        with self._lock:
            for k, st in self.status.items():
                if st == "running" and k != key:
                    self._finish_locked(k, now)
            self.status[key] = "running"
            self.enabled.add(key)
            self.t_start.setdefault(key, now)
            if total is not None:
                self.total[key] = total
            self.detail = detail or _LABELS[key]
            self.unit = None
        self._changed()

    def _finish_locked(self, key: str, now: float) -> None:
        if self.status.get(key) != "running":
            return
        self.status[key] = "done"
        self.t_end[key] = now
        if key in ("prepare", "post", "build", "ai_connect"):
            _learn(key, now - self.t_start.get(key, now))
        elif key in ("read", "anon") and self.done[key]:
            _learn(key, (now - self.t_start.get(key, now)) / self.done[key])

    def finish_stage(self, key: str, note: Optional[str] = None) -> None:
        with self._lock:
            self._finish_locked(key, time.time())
            if note:
                self.notes[key] = note
        self._changed()

    def skip(self, key: str, note: str) -> None:
        with self._lock:
            self.status[key] = "skipped"
            self.notes[key] = note
            self.enabled.discard(key)
        self._changed()

    def set_pages(self, n: int, first: int = 1) -> None:
        with self._lock:
            self.pages_total = n
            self.page_first = first
            self.total["read"] = n
            if "anon" in self.enabled:
                self.total["anon"] = n

    def page_read(self, page: int, health: int, vision_candidate: bool, is_ocr: bool) -> None:
        with self._lock:
            self.page_health[page] = health
            self.done["read"] = len(self.page_health)
            self.read_flags["vision"] += bool(vision_candidate)
            self.read_flags["ocr"] += bool(is_ocr)
            self.detail = f"Lendo páginas — {self.done['read']}/{self.pages_total or '?'}"
        self._changed()

    def update_health(self, page: int, health: int) -> None:
        with self._lock:
            self.page_health[page] = health

    def step(self, key: str, done: int, detail: Optional[str] = None) -> None:
        with self._lock:
            self.done[key] = done
            if detail:
                self.detail = detail
        self._changed()

    def unit_begin(self, key: str, page: int, expected_tokens: int, detail: str) -> None:
        with self._lock:
            self.unit = {"key": key, "page": page, "start": time.time(), "expected": max(30, expected_tokens),
                         "tokens": 0, "first_token": None, "phase": "analisando"}
            self.detail = detail
        self._changed()

    def llm_activity(self, tokens: int) -> None:
        """Chamado a cada fragmento recebido do Ollama (streaming)."""
        with self._lock:
            u = self.unit
            if not u:
                return
            if tokens and u["first_token"] is None:
                u["first_token"] = time.time()
            u["tokens"] = tokens
            if tokens > u["expected"]:  # texto maior que o previsto: ajusta a meta
                u["expected"] = int(tokens * 1.15) + 5
            u["phase"] = "escrevendo" if tokens else "analisando"

    def llm_retry(self, reason: str) -> None:
        with self._lock:
            if self.unit:
                self.unit.update(tokens=0, first_token=None, phase=f"nova tentativa ({reason})")

    def unit_end(self, key: str) -> None:
        with self._lock:
            u = self.unit
            if u and u["key"] == key:
                secs = time.time() - u["start"]
                self.unit_times[key].append(secs)
                _learn(key, secs)
            self.unit = None
            self.done[key] += 1
        self._changed()

    def complete(self) -> None:
        now = time.time()
        with self._lock:
            for k, st in self.status.items():
                if st == "running":
                    self._finish_locked(k, now)
            self.state, self.finished, self.unit = "done", now, None
            self.detail = "Concluído"
        self._changed()

    def fail(self, message: str) -> None:
        with self._lock:
            self.state, self.finished, self.unit = "error", time.time(), None
            self.detail = message
            for k, st in self.status.items():
                if st == "running":
                    self.status[k] = "error"
        self._changed()

    # ---- estimativa --------------------------------------------------------- #
    def _unit_rate(self, key: str) -> float:
        own = self.unit_times.get(key) or []
        if own:
            # execução atual pesa mais que o histórico
            mean = sum(own) / len(own)
            return (mean * len(own) + _rate(key)) / (len(own) + 1)
        return _rate(key)

    def _expected_units(self, key: str) -> Optional[float]:
        """Quantas páginas devem ir para a IA, enquanto a leitura não terminou."""
        if self.total[key] is not None:
            return float(self.total[key])
        read, n = self.done["read"], self.pages_total
        if not n:
            return None
        if read == 0:
            return None
        frac = self.read_flags["vision" if key == "vision" else "ocr"] / read
        est = frac * n
        if key == "vision" and self.vision_cap:
            est = min(est, self.vision_cap)
        return est

    def _current_unit_remaining(self, now: float) -> float:
        u = self.unit
        if not u:
            return 0.0
        rate = self._unit_rate(u["key"])
        elapsed = now - u["start"]
        if u["first_token"] and u["tokens"] >= 10:
            tps = u["tokens"] / max(0.5, now - u["first_token"])
            left = max(0, u["expected"] - u["tokens"]) / max(0.1, tps)
            return max(2.0, left)
        return max(rate * 0.15, rate - elapsed)

    def _eta(self, now: float) -> Dict[str, Any]:
        partial = False
        remaining = 0.0
        for key, _ in STAGES:
            st = self.status[key]
            if key not in self.enabled or st in ("done", "skipped", "off", "error"):
                continue
            if key == "read":
                n = self.pages_total
                if n is None:
                    remaining += 0.0
                    partial = True
                    continue
                done = self.done["read"]
                if done and "read" in self.t_start:
                    per = (now - self.t_start["read"]) / done
                    w = min(1.0, done / 5)
                    per = w * per + (1 - w) * _rate("read")
                else:
                    per = _rate("read")
                remaining += max(0, n - done) * per
            elif key in ("vision", "text"):
                units = self._expected_units(key)
                if units is None:
                    partial = True
                    continue
                if self.total[key] is None:
                    partial = True
                left = max(0.0, units - self.done[key])
                if self.unit and self.unit["key"] == key:
                    remaining += self._current_unit_remaining(now)
                    left = max(0.0, left - 1)
                remaining += left * self._unit_rate(key)
            elif key == "anon":
                n = self.total["anon"] or self.pages_total or 0
                remaining += max(0, n - self.done["anon"]) * _rate("anon")
            else:
                if st == "running":
                    remaining += max(0.0, _rate(key) - (now - self.t_start.get(key, now)))
                else:
                    remaining += _rate(key)
        return {"seconds": int(round(remaining)), "partial": partial}

    # ---- leitura (API) ------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            elapsed = (self.finished or now) - (self.started or now) if self.started else 0.0
            eta = self._eta(now) if self.state == "running" else {"seconds": 0 if self.state == "done" else None,
                                                                  "partial": self.state == "queued"}
            stages = []
            for key, label in STAGES:
                st = self.status[key]
                if st == "off":
                    continue
                t0, t1 = self.t_start.get(key), self.t_end.get(key)
                total = self.total[key]
                if total is None and key in ("vision", "text") and st == "pending":
                    exp = self._expected_units(key)
                    total_txt = f"~{round(exp)}" if exp is not None else None
                else:
                    total_txt = None
                stages.append({
                    "key": key, "label": label, "status": st, "done": self.done[key], "total": total,
                    "total_estimate": total_txt, "note": self.notes.get(key),
                    "elapsed": int(round((t1 or now) - t0)) if t0 else None,
                })
            unit = None
            if self.unit:
                u = self.unit
                tps = (u["tokens"] / max(0.5, now - u["first_token"])) if u["first_token"] else None
                unit = {"key": u["key"], "page": u["page"], "phase": u["phase"], "tokens": u["tokens"],
                        "expected_tokens": u["expected"], "tokens_per_s": round(tps, 1) if tps else None,
                        "elapsed": int(now - u["start"]),
                        "remaining": int(round(self._current_unit_remaining(now)))}
            return {
                "state": self.state, "detail": self.detail,
                "elapsed": int(round(elapsed)),
                "queued_for": int(round((self.started or now) - self.created)),
                "eta_seconds": eta["seconds"], "eta_partial": eta["partial"],
                "eta_finish": (now + eta["seconds"]) if eta["seconds"] is not None and self.state == "running" else None,
                "pages_total": self.pages_total, "page_first": self.page_first, "pages_read": self.done["read"],
                "health": [{"page": p, "health": h} for p, h in sorted(self.page_health.items())],
                "stages": stages, "unit": unit,
            }

    # ---- compatibilidade: texto curto para CLI/log --------------------------- #
    def line(self) -> str:
        s = self.snapshot()
        eta = s["eta_seconds"]
        eta_txt = f" | restante ~{fmt_secs(eta)}{'+' if s['eta_partial'] else ''}" if eta is not None else ""
        return f"{s['detail']} | decorrido {fmt_secs(s['elapsed'])}{eta_txt}"


def fmt_secs(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
