"""Vercel Python serverless entrypoint.

Exposes the FastAPI ASGI app as the `app` variable, which the @vercel/python
runtime wraps automatically. On Vercel (VERCEL=1) the application boots in
"lite" mode:

  - Dashboard, 3D graph explorer and RAG query work via cloud services
    (Neo4j Aura + any OpenAI-compatible API).
  - Text ingestion (.txt/.md/.html/.json/.csv) is inserted directly into
    LightRAG — no MinerU parser is available on serverless.
  - PDF/DOCX/PPTX parsing requires the Docker deployment instead.
"""

from app import app  # noqa: F401  (`app` is detected by the Vercel runtime)
