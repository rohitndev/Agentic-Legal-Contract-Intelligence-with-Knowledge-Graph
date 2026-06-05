"""FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.config import settings

from .routes import router

app = FastAPI(
    title="Agentic Legal Contract Intelligence with Knowledge Graph",
    description=(
        "Ingest contracts, build a legal knowledge graph, classify clauses into "
        "15 risk categories, retrieve CUAD precedents via RAG, quantify monetary "
        "exposure, and run a multi-turn negotiation agent."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "service": "Agentic Legal Contract Intelligence with Knowledge Graph",
        "version": __version__,
        "docs": "/docs",
        "capabilities": settings.summary(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
