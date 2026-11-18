"""OCR com Tesseract: múltiplas tentativas (orientação, binarização, deskew) e confiança por palavra."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from . import config
from .quality import text_metrics

logger = logging.getLogger("text_purifier.ocr")

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
cv2.setNumThreads(1)


def tesseract_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {"available": False, "version": None, "langs": [], "missing_langs": [],
                              "cmd": config.TESSERACT_CMD}
    try:
        status["version"] = str(pytesseract.get_tesseract_version())
        status["langs"] = sorted(pytesseract.get_languages(config=""))
        status["available"] = True
        wanted = [l for l in config.OCR_LANG.split("+") if l]
        status["missing_langs"] = [l for l in wanted if l not in status["langs"]]
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def effective_lang(status: Dict[str, Any]) -> str:
    langs = [l for l in config.OCR_LANG.split("+") if l and l in status.get("langs", [])]
    if not status.get("available"):
        return config.OCR_LANG
    return "+".join(langs) or "eng"


def ocr_image(arr: np.ndarray, psm: int, lang: str, page_number: int) -> Dict[str, Any]:
    """Executa Tesseract e devolve texto estruturado em linhas, confiança média e caixas das palavras."""
    result: Dict[str, Any] = {"text": "", "conf": 0.0, "words": 0, "quality": 0.0, "boxes": [],
                              "lines": [], "psm": psm, "error": None}
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(arr),
            lang=lang,
            config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
            timeout=config.OCR_TIMEOUT,
        )
    except Exception as exc:
        result["error"] = f"Tesseract (psm {psm}): {type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("Tesseract falhou na página %s (psm %s): %s", page_number, psm, exc)
        return result

    lines: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    order: List[Tuple[int, int, int]] = []
    conf_sum = weight = 0.0
    for i, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in lines:
            lines[key] = {"words": [], "confs": [], "bbox": [x, y, x + w, y + h]}
            order.append(key)
        item = lines[key]
        item["words"].append(word)
        item["confs"].append(conf)
        bb = item["bbox"]
        item["bbox"] = [min(bb[0], x), min(bb[1], y), max(bb[2], x + w), max(bb[3], y + h)]
        result["boxes"].append((x, y, x + w, y + h, conf))
        wl = max(1, len(word))
        conf_sum += conf * wl
        weight += wl

    out_lines: List[str] = []
    prev_par = None
    for key in order:
        item = lines[key]
        par = key[:2]
        if prev_par is not None and par != prev_par:
            out_lines.append("")
        prev_par = par
        text = " ".join(item["words"])
        out_lines.append(text)
        result["lines"].append({
            "text": text,
            "conf": sum(item["confs"]) / len(item["confs"]),
            "bbox": item["bbox"],
        })

    result["text"] = "\n".join(out_lines)
    result["conf"] = conf_sum / weight if weight else 0.0
    result["words"], result["quality"] = text_metrics(result["text"])
    return result


def ocr_score(res: Dict[str, Any]) -> float:
    return res["words"] * res["quality"] * (max(res["conf"], 1.0) / 100.0)


def osd_rotation(arr: np.ndarray) -> Optional[int]:
    """Rotação (0/90/180/270) sugerida pelo OSD do Tesseract; None se indeterminado."""
    try:
        osd = pytesseract.image_to_osd(
            Image.fromarray(arr), config="--psm 0 -c min_characters_to_try=5", timeout=60
        )
        match = re.search(r"Rotate:\s*(\d+)", osd)
        conf = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
        if match and (not conf or float(conf.group(1)) >= 1.0):
            return int(match.group(1)) % 360
    except Exception:
        pass
    return None


def rotate(arr: np.ndarray, degrees_cw: int) -> np.ndarray:
    if degrees_cw == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if degrees_cw == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if degrees_cw == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return arr


def binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE: realça escrita fraca (lápis, carbono, cópia desbotada)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(binary: np.ndarray) -> np.ndarray:
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


def _good(res: Dict[str, Any]) -> bool:
    return (res["words"] >= config.MIN_WORDS and res["quality"] >= config.MIN_QUALITY
            and res["conf"] >= config.OCR_GOOD_CONF)


def best_ocr(gray: np.ndarray, lang: str, page_number: int) -> Dict[str, Any]:
    """Executa uma sequência de tentativas até obter leitura confiável; devolve a melhor."""
    attempts: List[Dict[str, Any]] = []

    def attempt(arr: np.ndarray, psm: int, label: str, angle: int) -> Dict[str, Any]:
        res = ocr_image(arr, psm, lang, page_number)
        res["variant"] = label
        res["angle"] = angle
        res["shape"] = arr.shape[:2]
        attempts.append(res)
        return res

    # 1) imagem em tons de cinza: resolve a maioria dos documentos limpos
    if _good(attempt(gray, 3, "cinza", 0)):
        return _finish(attempts)

    # 2) orientação (OSD) + binarização + deskew
    rotation = osd_rotation(gray)
    order = [rotation] if rotation else [0]
    clean = deskew(binarize(gray))
    for angle in order:
        if _good(attempt(rotate(clean, angle), 3, "binarizada+deskew", angle)):
            return _finish(attempts)
        if _good(attempt(rotate(clean, angle), 6, "binarizada+bloco", angle)):
            return _finish(attempts)

    # 3) realce de contraste (escrita fraca/manuscrita)
    if _good(attempt(rotate(enhance_contrast(gray), order[0]), 3, "contraste", order[0])):
        return _finish(attempts)

    # 4) poucas palavras: testa as demais orientações
    best = max(attempts, key=ocr_score)
    if best["words"] < 25:
        for angle in (0, 90, 270, 180):
            if angle in order:
                continue
            if _good(attempt(rotate(clean, angle), 3, "binarizada+rotação", angle)):
                break
    return _finish(attempts)


def _finish(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = max(attempts, key=ocr_score)
    best["attempts"] = len(attempts)
    best["errors"] = [a["error"] for a in attempts if a.get("error")]
    return best
