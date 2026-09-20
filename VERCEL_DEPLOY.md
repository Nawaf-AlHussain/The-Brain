# Deploying The Brain on Vercel

The Brain now supports **two deployment modes** from a single codebase:

| | **Full mode** (Docker / self-hosted) | **Lite mode** (Vercel serverless) |
|---|---|---|
| Dashboard, Live Log, Stats | ✅ | ✅ |
| 3D Knowledge Graph explorer | ✅ | ✅ |
| RAG query (all modes) | ✅ | ✅ |
| Ingest `.txt` `.md` `.html` `.json` `.csv` | ✅ | ✅ (inserted directly) |
| Ingest PDF / DOCX / PPTX (MinerU OCR) | ✅ | ❌ — needs the Docker image |
| Local reranker model (`bge-reranker-v2-m3`) | ✅ | ❌ — use an external reranker API |
| Background document queue | ✅ | ❌ — ingestion runs inline |
| Vector index persistence | ✅ (named volume) | ⚠️ `/tmp` is ephemeral → attach Qdrant Cloud |

Lite mode is auto-activated when `VERCEL=1` (set automatically by Vercel) or `DEPLOYMENT_MODE=serverless`.

---

## Why lite mode exists

Vercel serverless functions have hard constraints that make the full RAG-Anything stack impossible:

1. **250 MB function size limit** — `raganything[all]` pulls in MinerU + PyTorch (~2 GB+).
2. **No persistent filesystem** — only `/tmp` is writable, and it is wiped when an instance recycles.
3. **No background workers** — the document queue worker can't survive between requests.
4. **~4.5 MB request body limit** — large PDF uploads would be rejected by the platform anyway.

The repository therefore ships a slim `requirements.txt` (serverless) and a `requirements-full.txt` (Docker).

---

## Step 1 — Create the backing cloud services

### Neo4j Aura DB (optional — persistent knowledge graph)
If you skip this step the app runs in **demo mode**: an in-memory graph that works fully (dashboard, query, graph explorer, text ingestion) but is wiped whenever Vercel recycles the instance. Add Aura later by setting `NEO4J_*` env vars and redeploying — no code changes needed.
1. Go to <https://neo4j.com/cloud/aura/> and create a **free** instance (Neo4j 5).
2. Download/save the credentials. You get a URI like `neo4j+s://xxxxxxxx.databases.neo4j.io`.

### Qdrant Cloud (recommended — persistent vector storage)
Without it, vector indexes are lost every time Vercel recycles your function instance, and queries return empty results on a cold start.
1. Go to <https://qdrant.tech/cloud/> and create a **free** cluster.
2. Note the cluster URL and API key.

### MinerU (optional — PDF parsing outside Vercel)
PDF/DOCX parsing cannot run on Vercel. For heavy documents, run the Docker stack locally or on a VPS (see the main README) — it connects to the **same** Neo4j/Qdrant instances, so your Vercel deployment can query everything it ingested.

---

## Step 2 — Deploy to Vercel

### Option A: Git integration (recommended)
1. Push this repository to GitHub (it already contains `vercel.json` and `api/index.py`).
2. In <https://vercel.com/new>, import the repository.
3. Framework preset: **Other** (auto-detected Python works fine).
4. Add the environment variables from the table below.
5. Deploy.

### Option B: CLI
```bash
npm i -g vercel
vercel login
vercel link
vercel env add NEO4J_URI          # repeat for every variable
vercel env add OPENAI_API_KEY
vercel --prod
```

---

## Step 3 — Environment variables (Vercel → Settings → Environment Variables)

### Required

| Variable | Example | Purpose |
|---|---|---|
| `LLM_ENGINE` | `openai` | Engine router — use `openai` for any OpenAI-compatible cloud API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint of your provider |
| `OPENAI_API_KEY` | `sk-…` | Your provider API key |
| `LLM_MODEL` | `gpt-4o-mini` | Entity extraction + query answering |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Vectorisation model |
| `EMBEDDING_DIM` | `1536` | **Must match the embedding model's output dimension exactly** |
| `NEO4J_URI` | `neo4j+s://xxxx.databases.neo4j.io` | Aura DB URI |
| `NEO4J_USERNAME` | `neo4j` | Aura DB user |
| `NEO4J_PASSWORD` | `…` | Aura DB password |

### Recommended

| Variable | Example | Purpose |
|---|---|---|
| `VISION_MODEL` | `gpt-4o-mini` | Multimodal processing (used for multimodal *queries*; ingestion-time image parsing needs Docker mode) |
| `QDRANT_URL` | `https://xxxx.cloud.qdrant.io:6333` | Persistent vectors |
| `QDRANT_API_KEY` | `…` | Qdrant auth |
| `RERANKER_BASE_URL` | `https://api.siliconflow.com/v1` | External reranker (Cohere-style `/rerank`) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model name |
| `RERANKER_API_KEY` | `…` | Reranker key (falls back to `OPENAI_API_KEY`) |

### Optional tuning

| Variable | Default | Notes |
|---|---|---|
| `MAX_EMBED_TOKENS` | `8192` | Embedding model context window |
| `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE` | `600` / `100` | Token chunking |
| `LLM_TIMEOUT` | `7200` locally | On Vercel keep ≤ 55 (function cap is 60 s on Hobby) |
| `LLM_MAX_ASYNC` / `EMBEDDING_MAX_ASYNC` | `1` | Concurrency against the provider |
| `MAX_UPLOAD_BYTES` | 4 MB serverless / 500 MB Docker | Platform body limit still applies |

> **Tip (EMBEDDING_DIM):** `text-embedding-3-small → 1536`, `text-embedding-3-large → 3072`. If the dimension doesn't match the model, insertion fails with a vector-size error.

---

## Verifying the deployment

1. `GET https://<your-app>.vercel.app/health` → `{"status": "ok", ...}`
2. Open `https://<your-app>.vercel.app/` → dashboard should load and show node/relation counts from Aura.
3. Upload a small `.md` file in the **Documents** tab → status turns **done** (direct insertion).
4. Query it in the **Query** tab (mode `hybrid` or `mix`).
5. Check the **Graph** tab for the extracted entity graph.

---

## Known limitations on Vercel

- **Cold-start vector search**: without Qdrant, the first request after instance recycling has empty local vector caches (graph/keyword retrieval still works against Neo4j).
- **No PDF ingestion**: upload returns HTTP 503 with an explanation for non-text files.
- **60 s function cap (Hobby plan)**: long LLM generations may time out; Pro plan allows longer durations via `maxDuration`.
- **SSE logs** work, but idle keepalives are subject to Vercel's streaming timeouts.
- **No queue**: pause/resume buttons have no effect; ingestion is synchronous.

For the complete experience (OCR-heavy PDFs, background queue, local reranker), deploy the Docker image on any VPS/Railway/Render/Fly.io and point it at the same Neo4j + Qdrant instances.
