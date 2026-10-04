import logging

import mlflow
from mlflow import MlflowClient

from src.config import MLFLOW_TRACKING_URI


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def inspect_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_name = "diabetes-regressor"

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise RuntimeError(
            f"No registered versions found for '{model_name}'."
        )

    for version in versions:
        LOGGER.info(
            "Model name: %s",
            version.name,
        )
        LOGGER.info(
            "Model version: %s",
            version.version,
        )
        LOGGER.info(
            "Run ID: %s",
            version.run_id,
        )
        LOGGER.info(
            "Source: %s",
            version.source,
        )


if __name__ == "__main__":
    configure_logging()
    inspect_model()