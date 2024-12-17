from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from constants import DATASET_PATH

def load_train_test_datasets(dataset:str, filter_attacks_train:bool=False):
    if dataset == 'CICIDS2017':
        df_train = pd.read_csv(f'{DATASET_PATH}/{dataset}/train.csv', index_col='index')
        ytrain = df_train[' Label']
        if filter_attacks_train:
            df_train = df_train[df_train[' Label']=='BENIGN']
        df_train.drop(columns=[' Label'], inplace=True)
        
        df_test = pd.read_csv(f'{DATASET_PATH}/{dataset}/test.csv', index_col='index')
        ytest = df_test[' Label']
        df_test.drop(columns=[' Label'], inplace=True)
        
        scaler = MinMaxScaler()
        scaler.fit(df_train)
        xtrain = scaler.transform(df_train)
        xtest = scaler.transform(df_test)
        return xtrain, xtest, ytrain, ytest, scaler
    else:
        df_train = pd.read_csv(f'{DATASET_PATH}/{dataset}/train.csv', index_col='index')
        ytrain = df_train['attack']
        if filter_attacks_train:
            df_train = df_train[df_train['attack']=='normal']
        df_train.drop(columns=['attack'], inplace=True)

        df_test = pd.read_csv(f'{DATASET_PATH}/{dataset}/test.csv', index_col='index')
        ytest = df_test['attack']
        df_test.drop(columns=['attack'], inplace=True)

        scaler = MinMaxScaler()
        scaler.fit(df_train)
        xtrain = scaler.transform(df_train)
        xtest = scaler.transform(df_test)
        return xtrain, xtest, ytrain, ytest, scaler
def test_files_cic():
    files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
             'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
             'Friday-WorkingHours-Morning.pcap_ISCX.csv',
             'Tuesday-WorkingHours.pcap_ISCX.csv',
             'Wednesday-workingHours.pcap_ISCX.csv',
             'Test.csv']
    return files
    
def cic_file_metrics(X_pca, labels, scalar, pca):
    path = './Data/CICIDS2017/TestDownsampled100/'
    files = test_files_cic()
    clust_normal, clust_anomaly = np.mean(X_pca[labels == 0], axis=0), np.mean(X_pca[labels !=0], axis=0)
    for file in files:
        df_test = pd.read_csv(f'{path}{file}', index_col='index')
        ytest = np.where(df_test[' Label']=='BENIGN', 0, 1)
        




