from __future__ import annotations

import logging

from src.pipeline.tasks import task_ingest_data, task_preprocess_data, task_train_baseline
from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def bootstrap() -> dict:
    ingestion = task_ingest_data()
    preprocessing = task_preprocess_data()
    training = task_train_baseline()

    return {
        "ingestion": ingestion,
        "preprocessing": preprocessing,
        "training": training,
    }


if __name__ == "__main__":
    configure_logging()
    logger.info("Bootstrap result: %s", bootstrap())
