"""API HTTP do text-purifier.

Endpoints:
  GET  /                  interface web
  GET  /health            verificação simples
  GET  /status            diagnóstico (Tesseract, IA local, bibliotecas)
  POST /jobs              inicia processamento assíncrono (retorna job_id)
  GET  /jobs/{id}         progresso (etapas, páginas lidas, IA em tempo real, estimativa de término)
  GET  /jobs/{id}/result  Markdown final
  POST /process           processamento síncrono (compatibilidade)
"""

import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from purifier import APP_NAME, VERSION, config, llm
from purifier.pipeline import InputError, library_versions, run, shutdown_executor, tesseract_info
from purifier.progress import Tracker
from purifier.report import build_output_filename

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("text_purifier")

app = FastAPI(title=APP_NAME, version=VERSION)
INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 6 * 3600


def _save_upload(file: UploadFile) -> str:
    filename = file.filename or "arquivo.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Arquivo inválido. Envie um PDF.")
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        size, head = 0, b""
        while chunk := file.file.read(1 << 20):
            if not head:
                head = chunk[:1024]
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail=f"Arquivo excede {config.MAX_UPLOAD_MB} MB.")
            tmp.write(chunk)
        tmp.close()
        if b"%PDF" not in head:
            raise HTTPException(status_code=415, detail="Arquivo inválido: não é um PDF.")
        return tmp.name
    except BaseException:
        tmp.close()
        _remove(tmp.name)
        raise


def _remove(path: Optional[str]) -> None:
    if path:
        try:
            os.remove(path)
        except Exception:
            pass


def _options(anon: bool, llm_flag: bool, llm_vision: bool, page_start: Optional[int], page_end: Optional[int],
             password: Optional[str], operador: Optional[str], remove_repeated: Optional[bool]) -> Dict[str, Any]:
    return {
        "anon": anon, "use_llm": llm_flag, "use_vision": llm_vision,
        "page_start": page_start, "page_end": page_end,
        "password": (password or "").strip(), "operator": (operador or "").strip()[:120],
        "remove_repeated": config.REMOVE_REPEATED_DEFAULT if remove_repeated is None else remove_repeated,
    }


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _purge_jobs() -> None:
    now = time.time()
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if v.get("finished_at") and now - v["finished_at"] > JOB_TTL]:
            JOBS.pop(jid, None)


@app.on_event("shutdown")
def _shutdown() -> None:
    shutdown_executor()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    with open(INDEX_FILE, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/health", response_class=PlainTextResponse)
def health() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/status")
def status(check_llm: bool = True) -> JSONResponse:
    info: Dict[str, Any] = {
        "app": APP_NAME, "version": VERSION, "workers": config.WORKERS,
        "always_ocr": config.ALWAYS_OCR, "tesseract": tesseract_info(refresh=True),
        "libraries": library_versions(),
    }
    if check_llm:
        info["llm"] = llm.discover(timeout=5)
    return JSONResponse(info)


@app.post("/jobs")
def create_job(
    request: Request,
    file: UploadFile = File(...),
    anon: bool = Form(False),
    llm: bool = Form(False),
    llm_vision: bool = Form(False),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    password: Optional[str] = Form(None),
    operador: Optional[str] = Form(None),
    remove_repeated: Optional[bool] = Form(None),
) -> JSONResponse:
    _purge_jobs()
    tmp_path = _save_upload(file)
    filename = file.filename or "arquivo.pdf"
    job_id = uuid.uuid4().hex[:12]
    options = _options(anon, llm, llm_vision, page_start, page_end, password, operador, remove_repeated)
    tracker = Tracker(options)
    job = {"id": job_id, "status": "queued", "tracker": tracker,
           "filename": filename, "created_at": time.time(), "finished_at": None,
           "result": None, "output_name": None, "summary": None, "error": None}
    with JOBS_LOCK:
        JOBS[job_id] = job

    def work() -> None:
        try:
            markdown, output_name, summary = run(tmp_path, filename, options, tracker, _client_ip(request), job_id)
            job.update(status="done", result=markdown, output_name=output_name, summary=summary)
        except InputError as exc:
            job.update(status="error", error=str(exc))
        except Exception as exc:
            logger.exception("Falha no processamento do job %s", job_id)
            job.update(status="error", error=f"Falha inesperada: {type(exc).__name__}: {exc}")
        finally:
            job["finished_at"] = time.time()
            _remove(tmp_path)

    threading.Thread(target=work, name=f"job-{job_id}", daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processamento não encontrado.")
    data = {k: job[k] for k in ("id", "filename", "output_name", "summary", "error")}
    progress = job["tracker"].snapshot()
    status = job["status"]
    if status not in ("done", "error"):
        status = "running" if progress["state"] == "running" else "queued"
    data.update(status=status, progress=progress, phase=progress["detail"],
                done=progress["pages_read"], total=progress["pages_total"] or 0)
    return JSONResponse(data)


@app.get("/jobs/{job_id}/result", response_class=PlainTextResponse)
def job_result(job_id: str) -> PlainTextResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processamento não encontrado.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=job.get("error") or "Processamento ainda não concluído.")
    return PlainTextResponse(job["result"], media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{job["output_name"]}"'})


@app.post("/process", response_class=PlainTextResponse)
def process_pdf(
    request: Request,
    file: UploadFile = File(...),
    anon: bool = Form(False),
    llm: bool = Form(False),
    llm_vision: bool = Form(False),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    password: Optional[str] = Form(None),
    operador: Optional[str] = Form(None),
    remove_repeated: Optional[bool] = Form(None),
) -> PlainTextResponse:
    filename = file.filename or "arquivo.pdf"
    tmp_path = _save_upload(file)
    options = _options(anon, llm, llm_vision, page_start, page_end, password, operador, remove_repeated)
    try:
        markdown, output_name, _ = run(tmp_path, filename, options, client_ip=_client_ip(request))
        return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{output_name}"'})
    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Falha no processamento do PDF")
        raise HTTPException(status_code=500, detail=f"Falha inesperada: {type(exc).__name__}: {exc}")
    finally:
        _remove(tmp_path)
