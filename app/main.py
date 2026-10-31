"""Drawing2Quote service, called by the POS backend (.NET):
  GET  /health                 is it running
  POST /extract   (file)       validate + read the drawing -> floors / areas / point counts (waits for the result)
  POST /jobs      (file)       the same in the background -> {jobId}; refused files answer 422 straight away
  GET  /jobs/{id}              {status: queued|running|done|failed, page, pages, message, result | detail}
  POST /suggest   (json)       {extraction, catalog, plan, choices?, sizeId?} -> suggested solution
Run:  uvicorn app.main:app --host 127.0.0.1 --port 8765
Set D2Q_KEY to require the same key from the POS (X-Service-Key header)."""
from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from .extract import RejectedFile, SUPPORTED_MESSAGE, extract, ocr
from .extract.validator import MAX_BYTES, check
from .rules.suggest import suggest

app = FastAPI(title="Drawing2Quote", version="1.0")


def service_key(x_service_key: str | None = Header(default=None)) -> None:
    key = os.environ.get("D2Q_KEY")
    if key and x_service_key != key:
        raise HTTPException(status_code=401, detail="Wrong service key.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "supported": SUPPORTED_MESSAGE}


# a plain def (not async): FastAPI runs it on a worker thread, so a long drawing doesn't freeze the service
@app.post("/extract", dependencies=[Depends(service_key)])
def extract_drawing(file: UploadFile = File(...)) -> dict:
    data = file.file.read(MAX_BYTES + 1)
    try:
        return extract(file.filename or "", data)
    except RejectedFile as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── background reading: a printed PDF is read with OCR (about a minute per sheet), far longer than a request may take
JOB_KEEP_SECONDS = 2 * 3600
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)     # one drawing at a time: the OCR engine and its progress hook are shared


def _run_job(job_id: str, file_name: str, data: bytes) -> None:
    job = _jobs[job_id]
    job.update(status="running", message="Reading the drawing")

    def progress(page_no: int, page_count: int) -> None:
        job.update(page=page_no, pages=page_count,
                   message=f"Reading page {page_no} of {page_count} - it was printed to PDF, so its words are read "
                           "from the page image (about a minute per sheet; a drawing already read is quick)")

    ocr.progress = progress
    try:
        job.update(status="done", result=extract(file_name, data), message="Done")
    except RejectedFile as exc:
        job.update(status="failed", detail=str(exc))
    except Exception as exc:     # never leave a job "running" forever
        job.update(status="failed", detail=f"The drawing could not be read: {exc}")
    finally:
        ocr.progress = None
        job["finished"] = time.time()


@app.post("/jobs", dependencies=[Depends(service_key)])
def start_job(file: UploadFile = File(...)) -> dict:
    data = file.file.read(MAX_BYTES + 1)
    name = file.filename or ""
    try:
        check(name, data)          # quick checks first: a refused file is answered now, not after the queue
    except RejectedFile as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    now = time.time()
    with _jobs_lock:
        for k in [k for k, j in _jobs.items() if j.get("finished") and now - j["finished"] > JOB_KEEP_SECONDS]:
            del _jobs[k]
        ahead = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "page": 0, "pages": 0, "fileName": name,
                         "message": f"Waiting for {ahead} other drawing(s) to be read" if ahead else "Starting"}
    _worker.submit(_run_job, job_id, name, data)
    return {"jobId": job_id}


@app.get("/jobs/{job_id}", dependencies=[Depends(service_key)])
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="This reading is no longer available - upload the drawing again.")
    return {k: v for k, v in job.items() if k != "finished"}


class SuggestRequest(BaseModel):
    extraction: dict[str, Any]
    catalog: dict[str, Any]
    plan: str
    choices: dict[str, Any] | None = None
    sizeId: int | None = None


@app.post("/suggest", dependencies=[Depends(service_key)])
def suggest_solution(req: SuggestRequest) -> dict:
    try:
        return suggest(req.extraction, req.catalog, req.plan, req.choices, req.sizeId)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
