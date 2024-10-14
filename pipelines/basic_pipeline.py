from zenml import pipeline
from steps.ingestion import load_data
from steps.preprocessing import handle_data
from steps.training import train_model
from steps.evaluation import evaluate_model

@pipeline
def forecast_pipeline():
    """
    This function represents a data forecasting pipeline that predicts the relative humidity of a given place one month into the future.

    The pipeline follows these steps:
    1. Ingest the data.
    2. Preprocess and split the data into training and test sets.
    3. Train a MLP model on the training data.
    4. Evaluate the model on the test data using RMSE, R², and MAE.
    """
    
    # Step 1: Load the data from a data source 
    df = load_data()

    # Step 2: Preprocess the data, handle missing values, and split it into training and test sets
    X_train, X_test, y_train, y_test = handle_data(df)

    # Step 3: Train the model using the training data
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the trained model on the test data and calculate evaluation metrics
    rmse, r2, mae = evaluate_model(model, X_test, y_test)