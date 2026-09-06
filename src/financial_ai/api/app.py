from fastapi import FastAPI

from financial_ai.api.routes import (
    router,
)


app = FastAPI(
    title="Financial AI API",
    description=(
        "Agentic financial analysis API "
        "combining predictive ML, "
        "news sentiment and SEC RAG."
    ),
    version="0.1.0",
)


app.include_router(
    router,
)


@app.get("/")
def root():

    return {
        "service": "Financial AI API",
        "status": "running",
        "docs": "/docs",
    }