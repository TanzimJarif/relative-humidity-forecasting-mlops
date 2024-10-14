import numpy as np

import pandas as pd

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract base class for different evaluation metrics.
    """

    @abstractmethod
    def calculate_score(self):
        """
        Abstract method for calculating a score. 
        """
        score = 0
        return score


class RMSEEvaluation(Evaluation):
    """
    Class for calculating the Root Mean Squared Error (RMSE) evaluation metric.
    """

    def calculate_score(self, y_test: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculates the RMSE score between the actual and predicted values.
        
        Args:
            y_test (pd.Series): The actual target values from the test set.
            y_pred (pd.Series): The predicted values from the model.
        
        Returns:
            float: The RMSE score.
        """
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        return score


class R2Evaluation(Evaluation):
    """
    Class for calculating the R-squared (R²) evaluation metric.
    """

    def calculate_score(self, y_test: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculates the R² score between the actual and predicted values.
        
        Args:
            y_test (pd.Series): The actual target values from the test set.
            y_pred (pd.Series): The predicted values from the model.
        
        Returns:
            float: The R² score.
        """
        score = r2_score(y_test, y_pred)
        return score


class MAEEvaluation(Evaluation):
    """
    Class for calculating the Mean Absolute Error (MAE) evaluation metric.
    """

    def calculate_score(self, y_test: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculates the MAE score between the actual and predicted values.
        
        Args:
            y_test (pd.Series): The actual target values from the test set.
            y_pred (pd.Series): The predicted values from the model.
        
        Returns:
            float: The MAE score.
        """
        score = mean_absolute_error(y_test, y_pred) 
        return score
