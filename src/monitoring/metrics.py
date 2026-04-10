from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "ml_api_request_total",
    "Total number of inference requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "ml_api_request_latency_seconds",
    "Latency of inference requests",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

DRIFT_SCORE_GAUGE = Gauge(
    "ml_drift_score",
    "Latest computed drift score",
)

PREDICTION_CLASS_GAUGE = Gauge(
    "ml_prediction_class_count",
    "Distribution of predicted classes",
    ["class_label"],
)
