import time

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from financial_ai.monitoring.prometheus_metrics import (
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)

from fastapi import Depends

from financial_ai.security.auth import (
    require_api_key,
)

from financial_ai.security.guardrails import (
    validate_financial_request,
)

app = FastAPI(
    title="Financial AI API",
    version="1.0.0",
)


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def prometheus_monitoring(request: Request, call_next):
    start_time = time.perf_counter()
    endpoint = request.url.path

    try:
        response = await call_next(request)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()

        return response

    except Exception:
        REQUEST_ERRORS.labels(
            endpoint=endpoint
        ).inc()

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=500,
        ).inc()

        raise

    finally:
        latency = time.perf_counter() - start_time

        REQUEST_LATENCY.labels(
            endpoint=endpoint
        ).observe(latency)
        
        
@app.post(
    "/analyze",
    dependencies=[
        Depends(require_api_key)
    ],
)
def analyze(request: AnalyzeRequest):

    query, ticker = (
        validate_financial_request(
            request.query,
            request.ticker,
        )
    )

    result = analyze_financial_query(
        query=query,
        ticker=ticker,
    )

    return result