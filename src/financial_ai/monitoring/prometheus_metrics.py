from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "financial_ai_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_ERRORS = Counter(
    "financial_ai_request_errors_total",
    "Total API request errors",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "financial_ai_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

DIRECTION_PREDICTIONS = Counter(
    "financial_ai_direction_predictions_total",
    "Direction model predictions",
    ["direction"],
)

DIRECTION_CONFIDENCE = Histogram(
    "financial_ai_direction_confidence",
    "Direction prediction confidence",
)

VOLATILITY_PREDICTION = Histogram(
    "financial_ai_volatility_prediction",
    "Predicted volatility",
)

MISSING_FEATURE_RATE = Gauge(
    "financial_ai_missing_feature_rate",
    "Share of missing ML feature values",
)

DIRECTION_MODEL_VERSION = Gauge(
    "financial_ai_direction_model_version",
    "Direction MLflow model version",
)

VOLATILITY_MODEL_VERSION = Gauge(
    "financial_ai_volatility_model_version",
    "Volatility MLflow model version",
)


def observe_prediction(
    direction=None,
    confidence=None,
    volatility=None,
    direction_version=None,
    volatility_version=None,
):
    if direction is not None:
        DIRECTION_PREDICTIONS.labels(
            direction=str(direction)
        ).inc()

    if confidence is not None:
        DIRECTION_CONFIDENCE.observe(float(confidence))

    if volatility is not None:
        VOLATILITY_PREDICTION.observe(float(volatility))

    if direction_version is not None:
        DIRECTION_MODEL_VERSION.set(float(direction_version))

    if volatility_version is not None:
        VOLATILITY_MODEL_VERSION.set(float(volatility_version))


def set_missing_feature_rate(rate: float):
    MISSING_FEATURE_RATE.set(float(rate))