from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.steps import BaseParameters, Output
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.constants import MLFLOW 
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from steps.ingestion import load_data
from steps.training import train_model
from steps.preprocessing import handle_data
from steps.evaluation import evaluate_model
from steps.deployment import deployment_trigger


docker_settings = DockerSettings(required_integrations = [MLFLOW])

@pipeline(enable_cache = False, settings = {"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    
    df = load_data()

    X_train, X_test, y_train, y_test = handle_data(df)

    model = train_model(X_train, y_train)

    rmse, r2, mae = evaluate_model(model, X_test, y_test)

    deploy_decision = deployment_trigger(rmse)
    
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deploy_decision,
        workers = workers,
        timeout = timeout, 
    )

