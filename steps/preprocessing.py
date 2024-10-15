import logging

import pandas as pd

from zenml import step

from typing import Annotated, Tuple

from src.data_cleaning import CleanData

from .config import PreprocesserConfig

@step
def handle_data(
    df: pd.DataFrame, 
    config: PreprocesserConfig
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans the data and splits it into train and test sets.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be preprocessed.
        config (PreprocesserConfig): Configuration object containing preprocessing settings.
    
    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): The training feature set.
            - X_test (pd.DataFrame): The testing feature set.
            - y_train (pd.Series): The training target values.
            - y_test (pd.Series): The testing target values.
    """
    try:
        logging.info(f"Cleaning the data...")

        # Initialize the CleanData object with the input DataFrame
        preprocesser = CleanData(df)

        # Remove rows with NaN values from the dataset
        preprocesser.drop_nan()

        # Select features and the target variable based on the configuration
        preprocesser.feature_selection(config.cols_to_drop, config.target)

        # Split the cleaned data into training and testing sets
        X_train, X_test, y_train, y_test = preprocesser.split_data(config.train_test_ratio, config.random_seed)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        # Log any errors that occur during the preprocessing step
        logging.error(f"Error while cleaning data: {e}")
        raise e
