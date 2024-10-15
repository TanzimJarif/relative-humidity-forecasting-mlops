from zenml.client import Client

from pipelines.basic_pipeline import forecast_pipeline


if __name__ == "__main__":

    #printing the tracker
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    #run the pipeline
    forecast_pipeline()