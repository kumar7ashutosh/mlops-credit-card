import numpy as np, pandas as pd,os
from src.logger import logging
import yaml,pickle
from sklearn.preprocessing import PowerTransformer

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logging.info('data loaded from %s',file_path)
    return df

def load_params(file_path:str)->dict:
    with open(file_path,'r') as f:
        file=yaml.safe_load(f)
    logging.info('params loaded from %s',file_path)
    return file

def sample_and_transform(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    train_cols = x_train.columns
    test_cols = x_test.columns

    pt = PowerTransformer(method='yeo-johnson')

    x_train_trans = pt.fit_transform(x_train)
    x_test_trans = pt.transform(x_test)

    train_df = pd.DataFrame(x_train_trans, columns=train_cols)
    test_df = pd.DataFrame(x_test_trans, columns=test_cols)

    train_df['Class'] = y_train.values
    test_df['Class'] = y_test.values

    with open('models/power_transformer.pkl', 'wb') as f:
        pickle.dump(pt, f)

    logging.info('Feature transformation finished')
    logging.info(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')

    return train_df, test_df


def save_data(df:pd.DataFrame,file_path:str):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    df.to_csv(file_path,index=False)
    logging.info('%s saved at %s',df,file_path)
    
def main():
    train_data=load_data('data/interim/train.csv')
    test_data=load_data('data/interim/test.csv')
    train_df,test_df=sample_and_transform(train_data=train_data,test_data=test_data)
    save_data(train_df,'data/processed/train_final.csv')
    save_data(test_df,'data/processed/test_final.csv')
    
if __name__=='__main__':
    main()