import os
from dotenv import load_dotenv

load_dotenv()

def get_required_env(name: str)->str:
    """Return a required environment variable or raise an error."""
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required env variable '{name}' is not configured."
        )
    return value

MLFLOW_TRACKING_URI = get_required_env("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = get_required_env("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = get_required_env("MLFLOW_TRACKING_PASSWORD")