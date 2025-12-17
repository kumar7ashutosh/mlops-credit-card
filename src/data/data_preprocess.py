import numpy as np, pandas as pd,os
from src.logger import logging
import yaml
from sklearn.model_selection import train_test_split

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logging.info('data loaded from %s',file_path)
    return df

def load_params(file_path:str)->dict:
    with open(file_path,'r') as f:
        file=yaml.safe_load(f)
    logging.info('params loaded from %s',file_path)
    return file

def preprocess_data(df:pd.DataFrame,test_size)->tuple:
    if 'Class' in df.columns:
        train_data,test_data=train_test_split(df,test_size=test_size,stratify=df['Class'],random_state=42)
    logging.info('data splitted to train and test')
    return train_data,test_data
def save_data(df:pd.DataFrame,file_path:str):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    df.to_csv(file_path,index=False)
    logging.info('%s saved at %s',df,file_path)
    
def main():
    df=load_data('data/raw/credit.csv')
    params=load_params('params.yaml')
    test_size=params['data_preprocessing']['test_size']
    logging.info('test size loaded from %s',params)
    train_data,test_data=preprocess_data(df=df,test_size=test_size)
    save_data(train_data,'data/interim/train.csv')
    save_data(test_data,'data/interim/test.csv')
    
if __name__=='__main__':
    main()