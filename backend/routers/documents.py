import re
import json
import time
import asyncio
from pathlib import Path
from typing import AsyncGenerator
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from backend.dependencies import job_manager, state
from backend.runtime import RAGANYTHING_AVAILABLE
from backend.config import (
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    TEXT_ONLY_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    IS_SERVERLESS,
)

router = APIRouter(tags=["Documents & Queue"])

# Lite mode (serverless): no MinerU parser available — only plain-text files
# can be inserted directly into LightRAG. Full parsing requires Docker mode.
LITE_MODE = IS_SERVERLESS or not RAGANYTHING_AVAILABLE


@router.get("/uploads")
def list_uploads():
    files = []
    for f in Path(UPLOAD_DIR).iterdir():
        if f.is_file():
            files.append(
                {
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return sorted(files, key=lambda x: x["modified"], reverse=True)


@router.delete("/uploads/{filename}")
def delete_upload(filename: str):
    safe = Path(UPLOAD_DIR) / Path(filename).name
    if not safe.exists():
        raise HTTPException(status_code=404, detail="File not found")
    safe.unlink()
    return {"deleted": filename}


@router.post("/queue/pause")
def pause_queue():
    job_manager.queue_paused = True
    return {"paused": True}


@router.post("/queue/resume")
def resume_queue():
    job_manager.queue_paused = False
    return {"paused": False}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if state.rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialised yet")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")

    if LITE_MODE and suffix not in TEXT_ONLY_EXTENSIONS:
        raise HTTPException(
            status_code=503,
            detail=(
                "Serverless mode has no MinerU parser: only plain-text files "
                "(.txt, .md, .html, .json, .csv) can be ingested here. "
                "Deploy with Docker to parse PDF/DOCX/PPTX documents."
            ),
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=400, detail=f"File too large (max {limit_mb} MB)"
        )

    dest = Path(UPLOAD_DIR) / safe_filename
    with dest.open("wb") as fh:
        fh.write(content)

    if LITE_MODE:
        # Direct insertion — chunking + entity extraction happen inline.
        # Must complete within the serverless function timeout.
        job = job_manager.new_job(safe_filename)
        job.status = "parsing"
        job.push("status", f"Inserting {safe_filename} into the knowledge graph …")
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
            await state.rag.ainsert(text, file_paths=[safe_filename])
            job.status = "done"
            job.finished_at = time.time()
            job.push("done", f"✓ Inserted {safe_filename} into the knowledge graph")
            job_manager.save_completed(safe_filename, 0, 0, 0, job.finished_at)
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.push("error", f"✗ {exc}")
        return {
            "job_id": job.id,
            "filename": safe_filename,
            "queue_position": 0,
        }

    job = job_manager.new_job(safe_filename)
    job.push("queued", f"File received: {safe_filename}")
    job_manager.processing_queue.append((job, str(dest)))

    return {
        "job_id": job.id,
        "filename": safe_filename,
        "queue_position": len(job_manager.processing_queue),
    }


@router.get("/progress/{job_id}")
async def progress_stream(job_id: str, from_index: int = 0):
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate() -> AsyncGenerator[str, None]:
        job = job_manager.jobs[job_id]
        sent = from_index
        idle = 0
        while True:
            events = list(job.events)
            while sent < len(events):
                yield f"data: {json.dumps({'index': sent, **events[sent]})}\n\n"
                sent += 1
                idle = 0

            if job.status in ("done", "error"):
                events = list(job.events)
                while sent < len(events):
                    yield f"data: {json.dumps({'index': sent, **events[sent]})}\n\n"
                    sent += 1
                break

            idle += 1
            if idle % 50 == 0:
                yield ": keepalive\n\n"

            await asyncio.sleep(0.3)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
