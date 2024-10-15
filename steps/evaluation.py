import mlflow

import logging

import pandas as pd

from zenml import step

from zenml.client import Client

from typing import Annotated, Tuple

from .config import ModelParameterConfig

from sklearn.neural_network import MLPRegressor

from src.model_testing import RMSEEvaluation, R2Evaluation, MAEEvaluation


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model: MLPRegressor, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    config: ModelParameterConfig
) -> Tuple[
    Annotated[float, "rmse"],
    Annotated[float, "r2"],
    Annotated[float, "mae"]
]:
    """
    This function predicts the target values using the test data, then calculates and returns
    the evaluation metrics: RMSE, R², and MAE. It also logs the evaluation process and any errors.
    
    Args:
        model (MLPRegressor): The trained MLP model to be evaluated.
        X_test (pd.DataFrame): The feature data used for testing.
        y_test (pd.Series): The actual target values corresponding to the test data.
        config (ModelParameterConfig): Configuration object containing model details.
    
    Returns:
        Tuple[float, float, float]: A tuple containing the evaluation metrics:
            - rmse (float): Root Mean Squared Error.
            - r2 (float): R-squared (coefficient of determination).
            - mae (float): Mean Absolute Error.
    """

    try:
        logging.info(f"Evaluating {config.name} model...")

        # generating predictions using the test data
        y_pred = model.predict(X_test)

        # calculating and logging RMSE (Root Mean Squared Error)
        rmse = RMSEEvaluation().calculate_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        # calculating and logging R² (coefficient of determination)
        r2 = R2Evaluation().calculate_score(y_test, y_pred)
        mlflow.log_metric("r2", r2)

        # calculating and logging MAE (Mean Absolute Error)
        mae = MAEEvaluation().calculate_score(y_test, y_pred)
        mlflow.log_metric("mae", mae)

        return rmse, r2, mae 

    except Exception as e:
        # logging any errors that occur during model evaluation
        logging.error(f"Error while testing {config.name} model: {e}")
        raise e
