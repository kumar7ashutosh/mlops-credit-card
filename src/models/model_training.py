import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import os
import sys
from src.logger import logging

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logging.info('data loaded from %s',file_path)
    return df

def load_params(file_path:str)->dict:
    with open(file_path,'r') as f:
        file=yaml.safe_load(f)
    logging.info('params loaded from %s',file_path)
    return file

def train_model(x_train:np.ndarray,y_train:np.ndarray)->LogisticRegression:
    params=load_params('params.yaml')
    model_params = params['model']  # Get model parameters
    C = model_params['C']
    solver = model_params['solver']
    penalty = model_params['penalty']
    class_weight=model_params['class_weight']
    clf = LogisticRegression(C=C, solver=solver, penalty=penalty,class_weight=class_weight)
    clf.fit(x_train, y_train)
    logging.info('Model training completed')
    return clf

def save_model(model:LogisticRegression,file_path:str):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
        
def main():
    train_data=load_data('data/processed/train_final.csv')
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    clf=train_model(x_train,y_train)
    save_model(clf,'models/model.pkl')
    
if __name__=='__main__':
    main()