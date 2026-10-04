import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
import yaml
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.config import MLFLOW_TRACKING_URI

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"

def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def load_config()->dict:
    """Load training configuration from YAML."""
    LOGGER.info("Loading training configuration from %s", CONFIG_PATH)

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as exc:
        LOGGER.error("Training configuration not found: %s", CONFIG_PATH)
        raise RuntimeError("Training configuration file is missing.") from exc
    except yaml.YAMLError as exc:
        LOGGER.error("Invalid YAML configuration: %s", exc)
        raise RuntimeError("Training configuration is invalid.") from exc

def load_dataset(config: dict):
    """Load and split the sklearn diabetes dataset."""
    LOGGER.info("Loading sklearn diabetes dataset.")

    dataset = load_diabetes()

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
    )

    LOGGER.info(
        "Dataset loaded. Training samples=%d, test samples=%d",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test

def create_model(config: dict) -> RandomForestRegressor:
    """Create the configured sklearn model."""
    training_config = config["training"]

    model_type = training_config["model_type"]

    if model_type != "random_forest":
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Currently supported: random_forest."
        )

    return RandomForestRegressor(
        n_estimators=training_config["n_estimators"],
        max_depth=training_config["max_depth"],
        random_state=training_config["random_state"],
        n_jobs=-1,
    )

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the trained model."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    LOGGER.info(
        "Model evaluation completed. "
        "MAE=%.4f, RMSE=%.4f, R2=%.4f",
        mae,
        rmse,
        r2,
    )

    return metrics

def train()->None:
    """Execute the complete training pipeline."""
    config = load_config()

    experiment_name = config["experiment"]["name"]
    model_name = config["model"]["name"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)

    LOGGER.info("MLflow tracking URI configured.")
    LOGGER.info("MLflow experiment: %s", experiment_name)
    LOGGER.info("Model name: %s", model_name)

    X_train, X_test, y_train, y_test = load_dataset(config)

    model = create_model(config)

    with mlflow.start_run(run_name=model_name) as run:
        LOGGER.info("Started MLflow run: %s", run.info.run_id)

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        training_config = config["training"]

        mlflow.log_params(
            {
                "model_type": training_config["model_type"],
                "n_estimators": training_config["n_estimators"],
                "max_depth": training_config["max_depth"],
                "random_state": training_config["random_state"],
                "test_size": config["data"]["test_size"],
                "data_random_state": config["data"]["random_state"],
            }
        )

        mlflow.log_metrics(metrics=metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        LOGGER.info(
            "Model logged and registered successfully. "
            "model=%s",
            model_name,
        )

        LOGGER.info(
            "Training completed successfully. run_id=%s",
            run.info.run_id,
        )
    
if __name__ == "__main__":
    configure_logging()
    train()