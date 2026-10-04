import logging
import os
import shutil
import sys
from pathlib import Path

import mlflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not configured."
        )

    return value


def main() -> None:
    if len(sys.argv) != 3:
        raise RuntimeError(
            "Usage: python mlflow_storage_initializer.py "
            "<model_uri> <destination_path>"
        )

    model_uri = sys.argv[1]
    destination = Path(sys.argv[2])

    tracking_uri = get_required_env("MLFLOW_TRACKING_URI")
    username = get_required_env("MLFLOW_TRACKING_USERNAME")
    password = get_required_env("MLFLOW_TRACKING_PASSWORD")

    LOGGER.info("Configuring MLflow tracking.")
    LOGGER.info("Tracking URI: %s", tracking_uri)
    LOGGER.info("Model URI: %s", model_uri)
    LOGGER.info("Destination: %s", destination)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        LOGGER.info("Downloading MLflow model artifact.")

        temporary_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
            )
        )

        LOGGER.info(
            "Model downloaded to temporary location: %s",
            temporary_path,
        )

        for item in temporary_path.iterdir():
            target = destination / item.name

            if item.is_dir():
                shutil.copytree(
                    item,
                    target,
                    dirs_exist_ok=True,
                )
            else:
                shutil.copy2(
                    item,
                    target,
                )

        LOGGER.info(
            "MLflow model successfully copied to: %s",
            destination,
        )

    except Exception:
        LOGGER.exception(
            "Failed to download MLflow model."
        )
        raise

    finally:
        if temporary_path and temporary_path.exists():
            shutil.rmtree(
                temporary_path,
                ignore_errors=True,
            )

            LOGGER.info(
                "Temporary model directory cleaned up."
            )


if __name__ == "__main__":
    main()