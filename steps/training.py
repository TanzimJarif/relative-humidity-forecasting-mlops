import logging

import pandas as pd

from zenml import step

from src.model_training import MLPmodel

from .config import ModelParameterConfig

from sklearn.neural_network import MLPRegressor


@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    config: ModelParameterConfig
) -> MLPRegressor:
    """
    Trains a Multi-Layer Perceptron (MLP) model using the provided training data and configuration.
    
    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The target values for training.
        config (ModelParameterConfig): Configuration object containing model parameters.
    
    Returns:
        MLPRegressor: The trained MLP regression model.
    """
    try:
        # Log the start of the model training process
        logging.info(f"Training {config.name} model...")

        # Initialize the MLPmodel class and train the model with the specified configuration
        mlp = MLPmodel()
        trained_model = mlp.train(
            X_train, y_train, 
            config.hidden_layers, config.activation, config.optimizer,
            config.alpha, config.max_iteration, config.random_seed
        )

        return trained_model

    except Exception as e:
        # Log any errors encountered during model training
        logging.error(f"Error while training {config.name} model: {e}")
        raise e
