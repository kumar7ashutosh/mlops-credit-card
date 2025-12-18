import json
import mlflow
import logging
import os
import sys
from src.logger import logging
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/mlops-credit-card.mlflow')
# dagshub.init(repo_owner='kumarashutoshbtech2023', repo_name='mlops-credit-card', mlflow=True)

def load_model_info(file_path:str)->dict:
    with open(file_path,'r') as file:
        model_info=json.load(file)
    logging.info('model info loaded from %s',model_info)
    return model_info

def register_model_and_transformer(model_name:str,model_info:dict,transformer_name:str,transformer_path:str):
    client=mlflow.tracking.MlflowClient()
    model_uri=f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    logging.info(model_uri)
    model_version=mlflow.register_model(model_uri,model_name)
    client.transition_model_version_stage(name=model_name,version=model_version.version,stage='Staging')
    logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    mlflow.log_artifact(transformer_path,artifact_path='preprocessing')
    transformer_uri=f"runs:/{model_info['run_id']}/preprocessing/{os.path.basename(transformer_path)}"
    transformer_version=mlflow.register_model(transformer_uri,transformer_name)
    client.transition_model_version_stage(name=transformer_name,version=transformer_version.version,stage='Staging')
    logging.debug(f'PowerTransformer {transformer_name} version {transformer_version.version} registered and transitioned to Staging.')

def main():
    model_info=load_model_info('reports/experiment_info.json')
    register_model_and_transformer(model_name='my_model',model_info=model_info,transformer_name='PowerTransformer',transformer_path='models/power_transformer.pkl')

if __name__=='__main__':
    main()
    