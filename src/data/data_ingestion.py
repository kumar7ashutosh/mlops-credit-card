import numpy as np, pandas as pd,os
from src.logger import logging
import yaml

def load_params(file_path:str)->dict:
    with open(file_path,'r') as file:
        params=yaml.safe_load(file)
    logging.info('params loaded from %s',file_path)
    return params

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logging.info('data loaded from %s',file_path)
    return df
def save_data(df:pd.DataFrame,file_path:str):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    df.to_csv(file_path,index=False)
    logging.info('Data saved to %s', file_path)
    
def main():
    df=load_data(file_path='https://media.githubusercontent.com/media/kumar7ashutosh/mlops-credit-card/refs/heads/main/notebooks/creditcard.csv')
    save_data(df,'data/raw/credit.csv')
    
if __name__=='__main__':
    main()