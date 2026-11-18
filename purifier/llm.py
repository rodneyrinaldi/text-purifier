"""Cliente da IA local (Ollama): descoberta de modelos, transcrição por visão e correção de OCR."""

import base64
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from . import config
from .quality import agreement, repetition_ratio, text_metrics

logger = logging.getLogger("text_purifier.llm")

VISION_HINTS = ("vl", "vision", "llava", "minicpm-v", "moondream", "bakllava", "gemma3", "granite3.2-vision",
                "mistral-small3.1", "llama4")
NON_TEXT_HINTS = ("embed", "bge", "nomic", "minilm", "rerank", "whisper")
CODE_HINTS = ("coder", "code", "starcoder", "codestral", "sqlcoder")
OOM_HINTS = ("out of memory", "allocation failed", "cuda error", "oom", "insufficient memory")

# capacidades por modelo (preenchido por discover); usado p/ desligar "thinking"
MODEL_CAPS: Dict[str, List[str]] = {}

VISION_PROMPT = (
    "Você é um transcritor forense. Transcreva INTEGRAL e LITERALMENTE todo o texto visível nesta "
    "imagem de uma página de processo judicial brasileiro, inclusive texto manuscrito, carimbos, "
    "anotações à margem, cabeçalhos e rodapés, na ordem natural de leitura. Regras obrigatórias:\n"
    "- Não resuma, não interprete, não corrija ortografia, não complete lacunas, não invente.\n"
    "- Preserve exatamente números, datas, valores, nomes e numeração de folhas.\n"
    "- Trechos ilegíveis: escreva [ilegível]. Assinaturas: escreva [assinatura]. Carimbos: [carimbo: texto lido].\n"
    "- Se a página não tiver texto, responda apenas: [SEM TEXTO]\n"
    "- Responda somente com a transcrição, sem comentários."
)

CORRECTION_PROMPT = (
    "Corrija somente erros evidentes de OCR no texto abaixo (letras trocadas, palavras partidas, "
    "espaçamento). Não resuma, não omita, não acrescente, não comente. Preserve exatamente números, "
    "datas, valores e marcadores entre colchetes. Devolva apenas o texto corrigido.\n\nTEXTO:\n"
)

_REFUSAL = re.compile(r"(?i)^(desculpe|sinto muito|não (posso|consigo)|i('m| am) (sorry|unable)|"
                      r"como (um|uma) (modelo|ia)|as an ai|here is|aqui está)")


def base_url(url: str = "") -> str:
    url = (url or config.LLM_URL).strip().rstrip("/")
    url = re.sub(r"/api(/generate|/chat|/tags)?$", "", url)
    if url and not url.startswith("http"):
        url = "http://" + url
    return url


def _size_b(name: str, details: Dict[str, Any]) -> Optional[float]:
    for source in (details.get("parameter_size") or "", name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", source)
        if m:
            return float(m.group(1))
    return None


def discover(timeout: float = 8.0) -> Dict[str, Any]:
    """Consulta o servidor Ollama e escolhe os modelos de texto e de visão."""
    status: Dict[str, Any] = {"url": base_url(), "available": False, "models": [], "text_model": None,
                              "vision_model": None, "version": None, "error": None, "notes": []}
    if not status["url"]:
        status["error"] = "LLM_URL não configurada"
        return status
    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                status["version"] = client.get(f"{status['url']}/api/version").json().get("version")
            except Exception:
                pass
            tags = client.get(f"{status['url']}/api/tags")
            tags.raise_for_status()
            models = []
            for m in tags.json().get("models", []):
                name = m.get("name") or m.get("model")
                details = m.get("details") or {}
                caps: List[str] = []
                try:
                    show = client.post(f"{status['url']}/api/show", json={"model": name})
                    if show.status_code == 200:
                        caps = show.json().get("capabilities") or []
                except Exception:
                    pass
                MODEL_CAPS[name] = caps
                low = name.lower()
                vision = "vision" in caps or any(h in low for h in VISION_HINTS)
                models.append({
                    "name": name, "size_b": _size_b(name, details), "vision": vision,
                    "embedding": "embedding" in caps or any(h in low for h in NON_TEXT_HINTS),
                    "family": details.get("family"), "quant": details.get("quantization_level"),
                    "code": any(h in low for h in CODE_HINTS), "thinking": "thinking" in caps,
                })
            status["models"] = models
            status["available"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status

    names = {m["name"]: m for m in status["models"]}

    def pick(requested: str, pool: List[Dict[str, Any]], preferred_sizes: List[float]) -> Optional[str]:
        if requested:
            if requested in names:
                return requested
            match = [n for n in names if n.split(":")[0] == requested or n.startswith(requested)]
            if match:
                return match[0]
            status["notes"].append(f"Modelo '{requested}' não encontrado no servidor; escolha automática aplicada.")
        for size in preferred_sizes:
            for m in pool:
                if m["size_b"] and abs(m["size_b"] - size) <= 1.5:
                    return m["name"]
        return pool[0]["name"] if pool else None

    text_pool = [m for m in status["models"] if not m["embedding"] and not m["vision"]]
    # modelos de código ficam por último; depois prefere qwen (bom em português) e o maior
    text_pool.sort(key=lambda m: (m["code"], "qwen" not in m["name"].lower(), -(m["size_b"] or 0)))
    vision_pool = [m for m in status["models"] if m["vision"] and not m["embedding"]]
    vision_pool.sort(key=lambda m: ("qwen" not in m["name"].lower(), m["size_b"] or 99))

    status["text_model"] = pick(config.LLM_TEXT_MODEL, text_pool or vision_pool, [14, 8, 7])
    status["vision_model"] = pick(config.LLM_VISION_MODEL, vision_pool, [7, 8, 11, 12, 14]) if vision_pool or config.LLM_VISION_MODEL else None
    vm = names.get(status["vision_model"] or "")
    if vm and (vm.get("size_b") or 0) > 16:
        status["notes"].append(f"Modelo de visão '{vm['name']}' é grande ({vm['size_b']:.0f}B): transcrição lenta "
                               "se não couber na GPU. Alternativa mais leve: 'ollama pull qwen2.5vl:7b'.")
    if not vision_pool and not config.LLM_VISION_MODEL:
        status["notes"].append("Nenhum modelo com visão no servidor: páginas manuscritas/imagem dependem só do OCR. "
                               "Sugestão: 'ollama pull qwen2.5vl:7b'.")
    return status


class LlmError(Exception):
    pass


def _generate(model: str, prompt: str, images: Optional[List[bytes]] = None, num_predict: int = 4096,
              num_ctx: Optional[int] = None, timeout: Optional[int] = None,
              on_tokens: Optional[Callable[[int], None]] = None) -> str:
    """Chamada ao Ollama em streaming: permite acompanhar a geração (tokens) em tempo real.

    `timeout` é o limite total da chamada (o servidor pode passar minutos analisando a imagem
    antes do primeiro token)."""
    payload: Dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": True, "keep_alive": "15m",
        "options": {"temperature": 0, "top_p": 0.9, "num_ctx": num_ctx or config.LLM_NUM_CTX,
                    "num_predict": num_predict, "repeat_penalty": 1.05},
    }
    if "thinking" in MODEL_CAPS.get(model, []):
        payload["think"] = False  # transcrição literal: raciocínio só custa tempo
    if images:
        payload["images"] = [base64.b64encode(img).decode("ascii") for img in images]
    limit = timeout or config.LLM_TIMEOUT
    deadline = time.monotonic() + limit
    parts: List[str] = []
    tokens = 0
    with httpx.Client(timeout=httpx.Timeout(limit, connect=10.0)) as client:
        with client.stream("POST", f"{base_url()}/api/generate", json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", "replace")
                try:
                    msg = json.loads(body).get("error") or body
                except Exception:
                    msg = body or f"HTTP {response.status_code}"
                raise LlmError(str(msg)[:300])
            if on_tokens:
                on_tokens(0)
            for line in response.iter_lines():
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout(f"tempo limite de {limit}s excedido")
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if data.get("error"):
                    raise LlmError(str(data["error"])[:300])
                chunk = data.get("response") or ""
                if chunk:
                    parts.append(chunk)
                    tokens += 1
                    if on_tokens:
                        on_tokens(tokens)
                if data.get("done"):
                    break
    return "".join(parts).strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", text).strip()


def digits_key(token: str) -> str:
    return re.sub(r"\D", "", token)


def number_tokens(text: str) -> List[str]:
    """Números relevantes (>= 3 dígitos), normalizados só para dígitos (CEP, valores, datas, processos)."""
    return [d for d in (digits_key(t) for t in re.findall(r"\d[\d.,/\-]*\d", text or "")) if len(d) >= 3]


def transcribe_page(model: str, render: Callable[[int], bytes], ocr_text: str, ocr_conf: Optional[float],
                    ocr_numbers: Optional[List[str]] = None,
                    on_tokens: Optional[Callable[[int], None]] = None,
                    on_retry: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Transcrição da página por modelo de visão, com verificações contra alucinação.

    `render(max_side)` devolve o JPEG da página; em falta de memória na GPU a imagem e o
    contexto são reduzidos e a chamada é repetida.
    """
    out: Dict[str, Any] = {"text": "", "accepted": False, "reason": "", "agreement": None, "health": 0,
                           "missing_numbers": [], "attempts": []}
    side = config.LLM_VISION_MAX_SIDE
    plans = [(side, config.LLM_VISION_NUM_CTX), (min(side, 1024), 4096), (768, 3072)]
    text = None
    for attempt, (max_side, ctx) in enumerate(dict.fromkeys(plans)):
        if attempt and on_retry:
            on_retry(f"memória da GPU insuficiente; imagem {max_side}px")
        try:
            text = _strip_fences(_generate(model, VISION_PROMPT, [render(max_side)], num_predict=3000,
                                           num_ctx=ctx, timeout=config.LLM_VISION_TIMEOUT,
                                           on_tokens=on_tokens))
            out["attempts"].append(f"{max_side}px/ctx {ctx}: ok")
            break
        except LlmError as exc:
            out["attempts"].append(f"{max_side}px/ctx {ctx}: {exc}")
            if not any(h in str(exc).lower() for h in OOM_HINTS):
                out["reason"] = f"erro do modelo: {exc}"
                return out
        except Exception as exc:
            out["reason"] = f"falha na chamada: {type(exc).__name__}: {exc}"
            return out
    if text is None:
        out["reason"] = "memória insuficiente na GPU mesmo com imagem reduzida"
        return out

    if not text or text.strip() == "[SEM TEXTO]":
        out["reason"] = "modelo não encontrou texto"
        return out
    if _REFUSAL.search(text[:80]):
        out["reason"] = "resposta do modelo não é transcrição (recusa/comentário)"
        return out
    if repetition_ratio(text) > 0.4:
        out["reason"] = "resposta descartada: repetição anômala (laço do modelo)"
        return out

    words, quality = text_metrics(text)
    agree = agreement(text, ocr_text) if ocr_text.strip() else None
    out["agreement"] = agree
    ocr_words, _ = text_metrics(ocr_text)
    # OCR confiável e IA divergente => suspeita de alucinação
    if agree is not None and agree < 0.25 and (ocr_conf or 0) >= 70 and ocr_words >= 30:
        out["reason"] = f"transcrição diverge do OCR confiável (concordância {agree:.0%}); descartada"
        return out
    if words < 3:
        out["reason"] = "transcrição sem palavras suficientes"
        return out

    base = 0.55 + (0.30 * min(1.0, agree / 0.8) if agree is not None else 0.10)
    health = 100 * min(0.85, base * (0.6 + 0.4 * min(1.0, quality / 0.9)))
    # números lidos com confiança pelo OCR que sumiram na transcrição (ex.: CEP/valor cortado)
    got = set(number_tokens(text))
    missing = [n for n in dict.fromkeys(ocr_numbers or []) if n not in got and not any(n in g for g in got)]
    if missing:
        out["missing_numbers"] = missing
        health -= min(20, 5 * len(missing))
    out["health"] = int(round(max(0, health)))
    out.update(text=text, accepted=True, reason="aceita")
    return out


def _numbers(text: str) -> List[str]:
    return sorted(re.findall(r"\d+", text))


def correct_text(model: str, text: str, on_tokens: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
    """Correção de erros de OCR com salvaguardas (tamanho, números e marcadores preservados)."""
    out: Dict[str, Any] = {"text": text, "applied": False, "reason": ""}
    if not text.strip():
        return out
    try:
        fixed = _strip_fences(_generate(model, CORRECTION_PROMPT + text, num_predict=max(1024, len(text)),
                                        on_tokens=on_tokens))
    except Exception as exc:
        out["reason"] = f"falha na chamada: {type(exc).__name__}"
        return out
    ratio = len(fixed) / max(1, len(text.strip()))
    if not 0.85 <= ratio <= 1.15:
        out["reason"] = f"descartada: alterou o tamanho do texto ({ratio:.0%})"
    elif _numbers(fixed) != _numbers(text):
        out["reason"] = "descartada: alterou números/datas/valores"
    elif sorted(re.findall(r"\[[^\]]{1,40}\]", fixed)) != sorted(re.findall(r"\[[^\]]{1,40}\]", text)):
        out["reason"] = "descartada: alterou marcadores"
    elif _REFUSAL.search(fixed[:80]):
        out["reason"] = "descartada: resposta não é o texto corrigido"
    else:
        out.update(text=fixed, applied=True, reason="aplicada")
    return out
