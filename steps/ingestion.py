import logging

import pandas as pd

from zenml import step

from .config import DataConfig

from src.data_loading import LoadData


@step
def load_data(config: DataConfig) -> pd.DataFrame:
    """
    Loads data from the specified file path into a pandas DataFrame.
    
    Args:
        config (DataConfig): Configuration object containing data-related settings, including the file path.
    
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        logging.info(f"Loading data from {config.data_path}...")
        # Load the data using the LoadData class and return it as a DataFrame
        return LoadData(config.data_path).get_data()

    except Exception as e:
        # Log any errors that occur during data ingestion
        logging.error(f"Error while loading data from {config.data_path}: {e}")
        raise e 
