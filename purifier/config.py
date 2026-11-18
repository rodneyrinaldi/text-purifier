"""Configuração via variáveis de ambiente (.env)."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # evita oversubscription do Tesseract em paralelo


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = env_str(name, "")
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "sim", "on", "s", "y"}


def available_cpus() -> int:
    """CPUs efetivas, respeitando limite de cgroup (docker --cpus)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        return os.cpu_count() or 2


# ---- Extração / OCR -------------------------------------------------------- #
ALWAYS_OCR = env_bool("ALWAYS_OCR", True)            # OCR em todas as páginas (validação cruzada)
OCR_DPI = env_int("OCR_DPI", 300)
OCR_MAX_DPI = env_int("OCR_MAX_DPI", 600)
OCR_MAX_PIXELS = env_int("OCR_MAX_PIXELS", 60_000_000)
OCR_LANG = env_str("OCR_LANG", "por+eng")
OCR_TIMEOUT = env_int("OCR_TIMEOUT", 240)
OCR_GOOD_CONF = env_float("OCR_GOOD_CONF", 80.0)    # confiança média que dispensa novas tentativas
MIN_WORDS = env_int("MIN_WORDS", 6)
MIN_QUALITY = env_float("MIN_QUALITY", 0.55)
BLANK_INK_RATIO = env_float("BLANK_INK_RATIO", 0.002)
WORKERS = max(1, env_int("WORKERS", available_cpus()))
MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 500)

# ---- Saúde da leitura ------------------------------------------------------ #
HEALTH_WARN = env_int("HEALTH_WARN", 70)          # abaixo: leitura comprometida (revisar)
HEALTH_CRITICAL = env_int("HEALTH_CRITICAL", 50)  # abaixo: leitura crítica

# ---- Pós-processamento ----------------------------------------------------- #
REMOVE_REPEATED_DEFAULT = env_bool("REMOVE_REPEATED_HEADERS", True)

# ---- IA local (Ollama) ----------------------------------------------------- #
LLM_URL = env_str("LLM_URL", "http://192.168.15.6:11434")
LLM_TEXT_MODEL = env_str("LLM_TEXT_MODEL", env_str("LLM_MODEL", ""))  # vazio = escolha automática
LLM_VISION_MODEL = env_str("LLM_VISION_MODEL", "")                     # vazio = escolha automática
LLM_TIMEOUT = env_int("LLM_TIMEOUT", 300)                   # correção de texto (por página)
LLM_VISION_TIMEOUT = env_int("LLM_VISION_TIMEOUT", 1200)    # visão (por página; modelos grandes são lentos)
LLM_VISION_NUM_CTX = env_int("LLM_VISION_NUM_CTX", 6144)
LLM_VISION_THRESHOLD = env_int("LLM_VISION_THRESHOLD", 75)  # saúde abaixo disso -> transcrição por visão
LLM_VISION_MAX_PAGES = env_int("LLM_VISION_MAX_PAGES", 0)   # 0 = sem limite
LLM_VISION_MAX_SIDE = env_int("LLM_VISION_MAX_SIDE", 1400)
LLM_NUM_CTX = env_int("LLM_NUM_CTX", 8192)

# ---- Anonimização ---------------------------------------------------------- #
SPACY_MODEL = env_str("SPACY_MODEL", "pt_core_news_sm")


def resolve_tesseract_cmd() -> str:
    """Localiza o executável do Tesseract (variável, PATH ou instalação padrão no Windows)."""
    explicit = env_str("TESSERACT_CMD", "")
    if explicit:
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ):
        if Path(candidate).exists():
            return candidate
    return "tesseract"


TESSERACT_CMD = resolve_tesseract_cmd()
