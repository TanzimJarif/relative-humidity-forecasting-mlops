# relative-humidity-forecasting-mlops

### Table of Contents
* Project Overview
* Pipeline Overview
* Code Structure
* Usage

## Project Overview
This project aims to develop a Multilayer Perceptron (MLP) model for forecasting relative humidity percentages for any geographical region up to one month into the future. Using historical weather data, the model predicts humidity levels to assist in agriculture, meteorology, and climate research applications.


## Pipeline Overview
The project follows a structured pipeline to manage data loading, preprocessing, model training, and evaluation. The main components of the pipeline are as follows:

* Data Ingestion: Loads the dataset into a DataFrame.
* Data Preprocessing: Cleans the data and prepares it for model training.
* Model Training: Trains the MLP model using the processed data.
* Model Evaluation: Evaluates the model's performance on a test set.


## Step-by-Step Description

### Data Loading:

File: ingestion.py

Function: ***load_data(config: DataConfig)***

Description: This function reads the dataset from a specified file path provided in the configuration. It utilizes the LoadData class to load the data into a pandas DataFrame and handles any errors during the loading process.

### Data Preprocessing:

File: preprocessing.py

Function: ***handle_data(df: pd.DataFrame, config: PreprocesserConfig)***

Description: This function cleans the data by removing rows with missing values and selecting relevant features and the target variable. It then splits the data into training and testing sets based on the specified ratio and random seed.

### Model Training:

File: training.py

Function: ***train_model(X_train: pd.DataFrame, y_train: pd.Series, config: ModelParameterConfig)***

Description: This function initializes and trains the MLP model using the provided training data and configuration parameters. It logs the training process and returns the trained model.

### Model Evaluation:

File: evaluation.py

Function: ***evaluate_model(model: MLPRegressor, X_test: pd.DataFrame, y_test: pd.Series, config: ModelParameterConfig)***

Description: This function evaluates the trained model's performance on the test set using metrics such as RMSE, R-squared, and MAE. It logs the evaluation process and handles any exceptions.

### Pipeline Execution:

File: basic_pipeline.py

Function: ***forecast_pipeline()***

Description: This function orchestrates the entire pipeline by sequentially calling the data loading, preprocessing, training, and evaluation functions, providing a streamlined workflow.

### Main Execution:

File: run_pipeline.py

Description: This is the main entry point for running the entire pipeline. When executed, it triggers the forecast pipeline and outputs the model evaluation results.

### Additional:

File: config.py 

Description: The file contains parameter configurations used throughout the project. It defines three classes for model parameters, preprocessing settings, and dataset configurations:
* ModelParameterConfig: Contains settings for the MLP model
* PreprocesserConfig: Specifies parameters for data preprocessing.
* DataConfig: Contains configuration for data loading.


## Usage
To run this project, follow these steps:

### Clone the repository:

```bash
git clone https://github.com/TanzimJarif/relative-humidity-forecasting.git
```

```bash
cd relative-humidity-forecasting-mlops
```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset is available at the specified path in the configuration file.

### Execute the pipeline:

Before running the pipeline, initiate ZenML using the following command:

```bash
zenml init
```

To start the ZenML server, run:

```bash
zenml up --blocking
```

Now the pipeline is ready to be executed.

```bash
python run_pipeline.py
```
