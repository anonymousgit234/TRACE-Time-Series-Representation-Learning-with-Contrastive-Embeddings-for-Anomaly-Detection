from joblib import Parallel, delayed
import pandas as pd
import os
import sys
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import optuna
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import psutil

current_dir = os.getcwd()
utils_path = os.path.abspath(os.path.join(current_dir, '..'))
# Add the python scripts folder to the system path
sys.path.append(utils_path)

# Now you can import the functions
from utils import *
from plots import *
from load_dataset import *
from constants import *
from cl_functions import *
from cl_utils import *
from ae_utils import *
from aug_utils import *


###############################################################

LABELS_FILEPATH = "/mnt/work/digitwin/subset_data_for_experiment/SMD/ServerMachineDataset/test_label/"
TRAINSET_FILEPATH = "/mnt/work/digitwin/subset_data_for_experiment/SMD/ServerMachineDataset/train"
TESTSET_FILEPATH = "/mnt/work/digitwin/subset_data_for_experiment/SMD/ServerMachineDataset/test"

params = {
    'n_features': 38,
    'encoding_dim': 16,
    'layer1': 50,
    'layer2': 40,
    'epochs': 200,
    'batch_size': 128
}


# Define the objective function for Optuna
def objective(trial):
  eps = trial.suggest_float('eps', 0.1, 1.0)  # Adjust the range as needed
  min_samples = trial.suggest_int('min_samples', features - range_limit, 2 * features + range_limit)
  print(eps, min_samples)
  model = DBSCAN(eps=eps, min_samples=min_samples)
  labels = model.fit_predict(X)
    
  # Ensure there is more than one cluster
  if len(np.unique(labels)) > 1:
    score = silhouette_score(X, labels)
  else:
    score = -1  # If only one cluster is found, return a low score    
  return score

model_name = 'TCNAE'
dataset_name = 'SMD'
mixup = False
Ytr = None
augmentation_type = 'recon_error'


files = os.listdir(TRAINSET_FILEPATH)
print(len(files))


for file in files:

  inverter = file
  train_data = pd.read_csv(TRAINSET_FILEPATH+'/'+file, sep=",", header=None)
  test_data = pd.read_csv(TESTSET_FILEPATH+'/'+file, sep=",", header=None)
  label_data = pd.read_csv(LABELS_FILEPATH+'/'+file, sep=",", header=None, names=['ytrue'])

  # Getting temporal dataframes 
  Xtr = temporalize(train_data.values)
  Xte = temporalize(test_data.values)

  # There is no Ytr in SMD dataset
  Yte = label_data[timesteps:]
  print(label_data.shape, Yte.shape)
  print(file, train_data.shape, Xtr.shape, test_data.shape, Xte.shape, label_data.shape, Yte.shape)
  print('Epochs:', params['epochs'], 'Encoding Dim', params['encoding_dim'])

  encoder, siamese_network, latent_train, latent_test = build_ae_siamese_network_v3(params, Xtr, Xte, Yte, dataset_name, inverter, model_name, Ytr=None, mixup=mixup, augmentation_type=augmentation_type)

  latent_train = encoder.predict(Xtr)
  latent_test = encoder.predict(Xte)

  # Scaling the latent embeddings 
  latent_scaler = MinMaxScaler()
  latent_scaler.fit(latent_train)
  latent_train = latent_scaler.transform(latent_train)
  latent_test = latent_scaler.transform(latent_test)
  print('latent_train.shape, latent_test.shape', latent_train.shape, latent_test.shape)

  CL_PLOTS_PATH = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Plots/CL/'

  kmeans_clustering_v3(model_name, dataset_name, latent_train, latent_test, Yte, inverter, n_clusters=2, Ytr=None, mixup=mixup, augmentation_type=augmentation_type)
  augmentation = model_name
  features = latent_train.shape[1]
  getEps_K_dist(latent_train, features, f'{CL_PLOTS_PATH}CL_{dataset_name}_{augmentation}_Kdist', 100)

  start_time = time.time()

  range_limit = 10
  X = latent_train

  # Run the optimization
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=35)
  

  # Output the best parameters
  best_params = study.best_params
  print("Best parameters:", best_params)
  print("Best silhouette score:", study.best_value)
  end_time = time.time()

  print(f"Optuna Runtime: {end_time - start_time} seconds")

  eps = best_params['eps']
  min_samples = best_params['min_samples']

  dbscan_clustering_v3(model_name, dataset_name, latent_train, latent_test, Yte, eps, min_samples, inverter, Ytr=None, mixup=mixup, augmentation_type=augmentation_type)

  
