import mlflow
from datetime import datetime

def set_experiment(experiment_name, gcs_bucket):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=gcs_bucket,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

def get_run_name(run_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{run_name}_{timestamp}"