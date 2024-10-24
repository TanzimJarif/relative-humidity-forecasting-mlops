from pipelines.deployment_pipeline import deployment_pipeline, inference_pipeline

import click

@click.command()

@click(
    "--config",
    "-c",
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help= "Optionally you can choose to only run the deployment pipeline to train and deploy a model ('deploy')." 
    "To only run a prediction against the deployed model ('predict')."
    "by default both will be run ('deploy_and_predict')",

)

@click.option(
    "--min-rmse",
    default=0.02,
    help = "Minimum error required to deploy model",
) 

def run_deployment(config:str, min_rmse = float):
    if deploy:
        deployment_pipeline(min_accuracy)
    if predict:
        inference_pipeline()
git 