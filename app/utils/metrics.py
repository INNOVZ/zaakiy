import os
from time import perf_counter
from typing import Callable

from fastapi import Request, Response
from prometheus_client import multiprocess  # noqa: F401
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus metrics objects
HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status"],
)

HTTP_INFLIGHT = Gauge(
    "http_inflight_requests",
    "In-flight HTTP requests",
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    labelnames=["method", "path"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        method = request.method
        # Prefer the route template to avoid high-cardinality metrics (e.g. /api/users/{id})
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)

        HTTP_INFLIGHT.inc()
        start = perf_counter()
        status_code = "500"
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            duration = perf_counter() - start
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)
            HTTP_REQUEST_TOTAL.labels(
                method=method, path=path, status=status_code
            ).inc()
            HTTP_INFLIGHT.dec()


def generate_metrics_response() -> Response:
    # Support multiprocess mode if PROMETHEUS_MULTIPROC_DIR is set (e.g., uvicorn workers>1)
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    else:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
