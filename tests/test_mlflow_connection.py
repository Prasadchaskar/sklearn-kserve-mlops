import mlflow
from src.config import MLFLOW_TRACKING_URI

def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_name = "connection-test"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="mlflow-connection-test",
    ):
        mlflow.log_param("test", "connection")
        mlflow.log_metric("status", 1.0)

    print("DagsHub MLflow connection successful.")


if __name__ == "__main__":
    main()