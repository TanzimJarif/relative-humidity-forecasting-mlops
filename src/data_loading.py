import pandas as pd

class LoadData:
    """
    A class for loading data from a specified file path.

    Attributes:
        data_path (str): The path to the data file.
    """
    def __init__(self, data_path: str):
        """
        Initializes the LoadData object with the given file path.
        
        Args:
            data_path (str): The path to the data file.
        """
        self.data_path = data_path
        
    
    def get_data(self) -> pd.DataFrame:
        """
        This method reads the data from the specified data path.
        
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        return pd.read_csv(self.data_path)
