import pandas as pd

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPRegressor

class Model(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def train(self):
        """
        Abstract method for training a model.
        """
        pass



class MLPmodel(Model):
    """
    A class for training a Multi-Layer Perceptron (MLP) model.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, hidden_layers: tuple, activation: str, 
              optimizer: str, alpha: float, max_iteration: int, random_seed: int) -> MLPRegressor:
        """
        Trains a Multi-Layer Perceptron (MLP) regression model with the given parameters.
        
        Args:
            X_train (pd.DataFrame): The feature matrix for training the model.
            y_train (pd.Series): The target variable for training.
            hidden_layers (tuple): The number of neurons in each hidden layer.
            activation (str): The activation function to use (e.g., 'relu', 'logistic').
            optimizer (str): The solver for weight optimization (e.g., 'adam', 'lbfgs').
            alpha (float): The regularization term (L2 penalty).
            max_iteration (int): The maximum number of iterations for training.
            random_seed (int): A seed to ensure reproducibility of the results.
        
        Returns:
            MLPRegressor: The trained MLP regression model.
        """
        # Initialize and train the MLP model with the provided hyperparameters
        mlp = MLPRegressor(hidden_layer_sizes = hidden_layers, activation = activation, solver = optimizer, 
                           alpha = alpha, max_iter = max_iteration, random_state = random_seed)

        # Train the model using the training data
        mlp.fit(X_train, y_train)

        return mlp
