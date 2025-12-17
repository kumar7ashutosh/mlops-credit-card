import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
from src.logger import logging

# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise RuntimeError("Missing DagsHub token")

# dagshub.auth.add_app_token(dagshub_token)

mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/mlops-credit-card.mlflow')
dagshub.init(repo_owner='kumarashutoshbtech2023', repo_name='mlops-credit-card', mlflow=True)
def load_model(file_path:str):
    with open(file_path,'rb') as file:
        model=pickle.load(file)
    logging.info('Model loaded from %s', file_path)
    return model

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logging.info('data loaded from %s',file_path)
    return df

def save_metrics(metrics:dict,file_path:str):
    with open(file_path,'w') as file:
        json.dump(metrics,file,indent=4)
    logging.info('file saved at %s',file_path)
    
def save_model_info(run_id:str,model_path:str,file_path:str):
    model_info={'run_id':run_id,
                'model_path':model_path}
    with open(file_path,'w') as file:
        json.dump(model_info,file,indent=4)
    logging.info('file saved at %s',file_path)
    
def evaluate_model(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)[:,1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
    }
    logging.info('Model evaluation metrics calculated')
    return metrics_dict

def main():
    mlflow.set_experiment('dvc pipeline experiment')
    with mlflow.start_run() as run:
        clf=load_model('models/model.pkl')
        test_data=load_data('data/processed/test_final.csv')
        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        metrics=evaluate_model(clf,x_test,y_test)
        save_metrics(metrics,'reports/metrics.json')
        for metric_name,metric_value in metrics.items():
            mlflow.log_metric(metric_name,metric_value)
        if hasattr(clf,'get_params'):
            params=clf.get_params()
            for param_name,param_value in params.items():
                mlflow.log_param(param_name,param_value)
        mlflow.sklearn.log_model(clf,'model')
        save_model_info(run.info.run_id,'model','reports/experiment_info.json')

if __name__=='__main__':
    main()