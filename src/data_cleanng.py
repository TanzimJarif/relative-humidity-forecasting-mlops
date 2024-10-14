import pandas as pd

from typing import Union

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

class CleanData:
    """
    A class for cleaning preprocessing and spliting the data.
    
    Attributes:
        df (pd.DataFrame): A DataFrame containing the dataset.
        X (list): A list that will store the feature variables after processing.
        y (list): A list that will store the target variable after processing.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the CleanData object with the given DataFrame.
        
        Args:
            df (pd.DataFrame): The dataset to be cleaned and processed.
        """
        self.df = df
        self.X = []
        self.y = []


    def drop_nan(self, verbose: boolean = false) -> None:
        """
        Drops rows with any NaN values in the DataFrame.
        """
        # Get the number of rows before dropping NaN
        initial_row_count = self.df.shape[0]

        # Drop rows with any NaN values
        self.df = self.df.dropna()

        # Get the number of rows after dropping NaN
        final_row_count = self.df.shape[0]

        # Calculate the number of rows dropped
        rows_dropped = initial_row_count - final_row_count
        
        if verbose:
            print(f"Number of rows dropped: {rows_dropped}")


    def feature_selection(self, cols_to_drop: list, target: str) -> None:
        """
        This method filters out the specified columns to drop from the feature set and assigns 
        the target column for prediction. It flattens the target for single-step forecasting.
        
        Args:
            cols_to_drop (list): A list of columns to be excluded from the features.
            target (str): The name of the column to be used as the target variable.
        """
        # Extract features and target
        features = [col for col in self.df.columns if col not in cols_to_drop]
        target = target

        # Flatten the target for a single step ahead forecast
        self.X = self.df[features]
        self.y = self.df[target]


    def standerdize_data(self) -> None:
        """
        This method applies standardization to both the feature set `X` 
        and the target variable `y` to ensure that all values are on a similar scale.
        """
        # Standardize the features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.y = scaler.fit_transform(self.y)


    def split_data(self, ratio: float, seed: int = 42) -> Union[pd.DataFrame, pd.Series]:
        """
        This method splits the preprocessed feature and target sets into training and test subsets, 
        using the specified ratio for test size.
        
        Args:
            ratio (float): The proportion of the dataset to include in the test split.
            seed (int): The random state seed to ensure reproducibility of the split. Defaults to 42.
        
        Returns:
            tuple: A tuple containing four elements:
                - X_train (pd.DataFrame): The training set features.
                - X_test (pd.DataFrame): The test set features.
                - y_train (pd.Series): The training set target variable.
                - y_test (pd.Series): The test set target variable.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = ratio, random_state = seed)
        return X_train, X_test, y_train, y_test