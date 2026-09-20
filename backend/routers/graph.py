import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.dependencies import neo4j_manager, state

# Create the router instance
router = APIRouter(tags=["Graph & Query"])


class QueryRequest(BaseModel):
    question: str
    mode: str = "mix"
    only_need_context: bool = False
    return_nodes: bool = False


class QueryLogCapture(logging.Handler):
    """Captures the 'Query nodes:' log line emitted by LightRAG during a query."""

    _QUERY_NODES_RE = re.compile(r"Query nodes?:\s*(.+?)(?:\s*\(top_k|$)")

    def __init__(self):
        super().__init__()
        self.entity_names: list[str] = []

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        m = self._QUERY_NODES_RE.search(msg)
        if m:
            raw = m.group(1)
            self.entity_names = [e.strip() for e in raw.split(",") if e.strip()]


async def _graph_from_lightrag(search: str = "", limit: int = 300) -> dict:
    """Build the graph payload from LightRAG's own graph storage.

    Used in demo mode (no Neo4j configured) where the graph lives in
    LightRAG's NetworkX storage instead of Neo4j.
    """
    rag = state.rag
    storage = getattr(rag, "chunk_entity_relation_graph", None)
    if storage is None:
        return {"nodes": [], "links": []}

    if search.strip():
        kg = await storage.get_knowledge_graph(
            node_label=search.strip(), max_depth=2, max_nodes=limit
        )
    else:
        kg = await storage.get_knowledge_graph(
            node_label="*", max_depth=2, max_nodes=limit
        )

    nodes: list[dict] = []
    links: list[dict] = []
    for n in kg.nodes:
        props = n.properties or {}
        nodes.append(
            {
                "id": n.id,
                "type": (props.get("entity_type") or "unknown").lower(),
                "desc": (props.get("description") or "")[:200],
                "degree": 0,
            }
        )
    for e in kg.edges:
        props = e.properties or {}
        links.append(
            {
                "source": e.source,
                "target": e.target,
                "label": (props.get("description") or props.get("type") or "")[:80],
                "weight": float(props.get("weight") or 1.0),
            }
        )

    degree_map: dict[str, int] = {}
    for link in links:
        degree_map[link["source"]] = degree_map.get(link["source"], 0) + 1
        degree_map[link["target"]] = degree_map.get(link["target"], 0) + 1
    for node in nodes:
        node["degree"] = degree_map.get(node["id"], 0)

    return {"nodes": nodes, "links": links}


@router.get("/graph")
async def get_graph(limit: int = 300, search: str = ""):
    try:
        if state.graph_backend == "neo4j":
            return await neo4j_manager.get_graph(limit, search)
        return await _graph_from_lightrag(search, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query(req: QueryRequest):
    if state.rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")

    capture = QueryLogCapture() if req.return_nodes else None
    if capture:
        for name in ["lightrag", "raganything"]:
            logging.getLogger(name).addHandler(capture)

    try:
        # RAGAnything.aquery accepts vlm_enhanced; plain LightRAG does not.
        if type(state.rag).__name__ == "RAGAnything":
            answer = await state.rag.aquery(
                req.question, mode=req.mode, vlm_enhanced=False
            )
        else:
            answer = await state.rag.aquery(req.question, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if capture:
            for name in ["lightrag", "raganything"]:
                logging.getLogger(name).removeHandler(capture)

    if req.only_need_context:
        if not answer:
            return {"context": "No relevant information found.", "mode": req.mode}
        return {"context": str(answer), "mode": req.mode}

    result: dict = {
        "answer": str(answer) if answer else "No results found.",
        "mode": req.mode,
    }

    if req.return_nodes and capture:
        result["highlighted_nodes"] = await neo4j_manager.resolve_node_ids(
            capture.entity_names
        )

    return result
