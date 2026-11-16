"""Drawing2Quote service, called by the POS backend (.NET):
  GET  /health                 is it running
  POST /extract   (file)       validate + read the drawing -> floors / areas / point counts (waits for the result)
  POST /jobs      (file)       the same in the background -> {jobId}; refused files answer 422 straight away
  GET  /jobs/{id}              {status: queued|running|done|failed, page, pages, message, result | detail}
  POST /suggest   (json)       {extraction, catalog, plan, choices?, sizeId?, installation?} -> suggested solution,
                               with its product breakdown per area ("breakdown")
  POST /suggest/xlsx (json)    the same -> the product breakdown as an Excel sheet
Run:  start-service.bat (development) / start-service-production.bat (production) - python -m app.serve
Settings: config/service.yaml (D2Q_ENV = development | production): where it listens, the key the POS must send
(X-Service-Key; D2Q_KEY overrides), the browser origins allowed (the POS frontends)."""
from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .extract import RejectedFile, SUPPORTED_MESSAGE, extract, ocr
from .extract.validator import MAX_BYTES, check
from .rules.breakdown import to_xlsx
from .rules.suggest import rule_list, suggest
from .settings import SETTINGS

app = FastAPI(title="Drawing2Quote", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=SETTINGS["allowed_origins"], allow_methods=["*"], allow_headers=["*"])


def service_key(x_service_key: str | None = Header(default=None)) -> None:
    key = SETTINGS["key"]
    if key and x_service_key != key:
        raise HTTPException(status_code=401, detail="Wrong service key.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "supported": SUPPORTED_MESSAGE, "reader": "python"}


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
    otherBrand: dict[str, Any] | None = None     # {company, areas[]}: another company's automation -> Matter gateways
    installation: float | None = None            # installation charges (in the product breakdown's total)
    client: str | None = None                    # name on the breakdown sheet (else the drawing's client / file name)


@app.get("/rules", dependencies=[Depends(service_key)])
def rules() -> list[dict]:
    """The drawing rules (config/solution_rules.yaml) a company chooses its products for: name, gangs, points"""
    return rule_list()


@app.post("/suggest", dependencies=[Depends(service_key)])
def suggest_solution(req: SuggestRequest) -> dict:
    try:
        result = suggest(req.extraction, req.catalog, req.plan, req.choices, req.sizeId, other_brand=req.otherBrand,
                         installation=req.installation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if req.client and result.get("breakdown"):
        result["breakdown"]["client"] = req.client
    return result


@app.post("/suggest/xlsx", dependencies=[Depends(service_key)])
def suggest_xlsx(req: SuggestRequest) -> Response:
    """The suggested solution's product breakdown per area as an Excel sheet (a row per product: unit price, qty,
    total; a column per area under its floor; installation charges and the total at the bottom)"""
    result = suggest_solution(req)
    if result.get("needsConfirmation"):
        raise HTTPException(status_code=409, detail="Confirm the products ticked for both Standard and Premium first.")
    name = re.sub(r"[^A-Za-z0-9 ._-]+", "", result["breakdown"]["client"] or "solution").strip() or "solution"
    return Response(to_xlsx(result["breakdown"]),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name} - {result["plan"]} solution.xlsx"'})
