import logging

import pandas as pd

from zenml import step

from typing import Annotated, Tuple

from .config import ModelParameterConfig

from sklearn.neural_network import MLPRegressor

from src.model_testing import RMSEEvaluation, R2Evaluation, MAEEvaluation


@step
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

        # Generate predictions using the test data
        y_pred = model.predict(X_test)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = RMSEEvaluation().calculate_score(y_test, y_pred)

        # Calculate R² (coefficient of determination)
        r2 = R2Evaluation().calculate_score(y_test, y_pred)

        # Calculate MAE (Mean Absolute Error)
        mae = MAEEvaluation().calculate_score(y_test, y_pred)
        
        return rmse, r2, mae 

    except Exception as e:
        # Log any errors that occur during model evaluation
        logging.error(f"Error while testing {config.name} model: {e}")
        raise e
