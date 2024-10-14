from zenml.steps import BaseParameters

class ModelParameterConfig(BaseParameters):
    """
    Model Parameters Configuration
    """
    name: str = "Multilayer Perception"
    hidden_layers: tuple = (32, 64, 32)
    activation: str = 'relu'
    optimizer: str = 'adam'
    alpha: float = 0.02
    max_iteration: int = 500
    random_seed: int = 42


class PreprocesserConfig(BaseParameters):
    """
    Preprocesser Configuration
    """
    train_test_ratio: float = 0.2
    cols_to_drop: list = ['year', 'groundfrost_12','rainfall_12', 'psl_12', 'pv_12', 'sfcWind_12', 'sun_12', 'tas_12', 'snowLying_12', 'hurs_12']
    target: str = 'hurs_12'
    random_seed: int = 42


class DataConfig(BaseParameters):
    """
    Dataset Configuration
    """
    data_path: str = "D:\Projects\Forecasting Relative Humidity\data\Met dataset - 2015-to-2022_12months.csv"
