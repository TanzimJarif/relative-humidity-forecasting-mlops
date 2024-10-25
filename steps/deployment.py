from zenml import step

from .config import DeploymentTriggerConfig

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy
