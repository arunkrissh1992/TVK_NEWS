from __future__ import annotations

from fastapi import FastAPI

from tnmi import __version__

app = FastAPI(title="Tamil Nadu Media Intelligence API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__}
