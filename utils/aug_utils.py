# Created by  : Vineela Pulagura
# Description : The file is used for generatin the different augmentation techniques along with positive and negative pair generation
import pandas as pd
import numpy as np

import pickle
from ae_utils import trainAEModel
from utils import flatten 
from constants import AUG_PAIRS_PATH, DATASET_PATH
timesteps = 8

# def generate_positive_pairs(X, X_enc):
#     ''' Generate positive pairs with the original dataset and the augmented dataset
#     CN changed first for loop iteration removing (1, X.shape[0]] in this as it was not giving augmentations for first index sample
#     '''
#     positive_pairs = []

#     #for i in range(1, X.shape[0]):
#     for i in range(X.shape[0]):
#         positive_pairs.append((X[i], X_enc[i - 1]))

#         if i > 1:
#             positive_pairs.append((X[i], X_enc[i - 2]))

#         positive_pairs.append((X[i], X_enc[i]))

#         if i < X.shape[0] - 1:
#             positive_pairs.append((X[i], X_enc[i + 1]))

#         if i < X.shape[0] - 2:
#             positive_pairs.append((X[i], X_enc[i + 2]))
#     return positive_pairs

def generate_positive_pairs(X, X_enc):
    ''' Generate positive pairs with the original dataset and the augmented dataset
    ensuring each element gets 5 pairs. For the first elements, rolling values 
    from the end of the second dataframe are used.
    CN wrote this function commenting previous one as it was not giving correct augmentations for first and last instances
    '''
    positive_pairs = []

    for i in range(X.shape[0]):
        pairs = []

        # For the first element, use the last two elements from X_enc
        if i == 0:
            pairs.append((X[i], X_enc[-1]))
            pairs.append((X[i], X_enc[-2]))
        else:
            pairs.append((X[i], X_enc[i - 1]))

            if i > 1:
                pairs.append((X[i], X_enc[i - 2]))
            else:
                pairs.append((X[i], X_enc[-1]))

        pairs.append((X[i], X_enc[i]))

        if i < X.shape[0] - 1:
            pairs.append((X[i], X_enc[i + 1]))
        else:
            pairs.append((X[i], X_enc[0]))

        if i < X.shape[0] - 2:
            pairs.append((X[i], X_enc[i + 2]))
        else:
            pairs.append((X[i], X_enc[(i + 2) % X.shape[0]]))

        positive_pairs.extend(pairs[:5])  # Ensure only 5 pairs per element

    return positive_pairs

def generate_negative_pairs(X, X_neg, neg_pairs_list):
    ''' generate negative pairs with the original dataset and the augmented dataset with the given random indices
    '''
    negative_pairs = []
    for i in range(X.shape[0]):
        for j in neg_pairs_list[i]:
            negative_pairs.append((X[i], X_neg[j]))
    return negative_pairs

def random_masking_array(X, num_values=5):
    ''' Mask the data randomly with the chosen number of random values
    CN changed this code to add X.copy() as without using .copy() as prev version it didn't work
    '''
    X_tmp = X.copy()
    random_values = np.random.choice(X.shape[0], num_values)
    print(random_values)
    for val in random_values:
        X_tmp[val] = 0
    return X_tmp

def jitter_time_series(time_series, strength=0.1):
    ''' Add noise to the features with a normal distribution
    '''
    noise = np.random.normal(0, strength, size=time_series.shape)
    jittered_time_series = time_series + noise
    return jittered_time_series
# def permutation(x, max_segments=5, seg_mode="equal"):
#     ''' Permute the series for the input data
#     '''
#     orig_steps = np.arange(x.shape[1])
#     num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
#     ret = np.zeros_like(x)
#     for i, pat in enumerate(x):
#         if num_segs[i] > 1:
#             if seg_mode == "random":
#                 split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
#                 split_points.sort()
#                 splits = np.split(orig_steps, split_points)
#             else:
#                 splits = np.array_split(orig_steps, num_segs[i])

#             # Pad each split to the same length
#             # max_len = max(len(split) for split in splits)
#             # padded_splits = [np.pad(split, (0, max_len - len(split)), mode='constant') for split in splits]
            
#             warp = np.concatenate(np.random.permutation(splits)).ravel()
#             #warp = np.concatenate(np.random.permutation(padded_splits)).ravel()
#             ret[i] = pat[warp]
#         else:
#             ret[i] = pat
#     return ret


def permutation(x, max_segments=5, seg_mode="equal"):
    ''' Permute the series for the input data '''
    ''' CN authored this function with some changes to VPs code as prev version didnt work for SMA data
    '''
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:  # Default to "equal" segmentation
                splits = np.array_split(orig_steps, num_segs[i])

            # Pad each split to the same length
            max_len = max(len(split) for split in splits)
            padded_splits = [np.pad(split, (0, max_len - len(split)), mode='constant') for split in splits]

            # Permute the padded splits
            permuted_splits = np.random.permutation(padded_splits)
            # Concatenate the permuted splits and trim to the original length
            warp = np.concatenate(permuted_splits)[:x.shape[1]]
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    
    return ret

def scaling_2d(x, sigma=1.0):
    '''Scale the data with the given value
    '''
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[1]))
    augmented_data = np.multiply(x, factor)
    return augmented_data

def neg_index_list(arr, current_index, num_values=5):
    '''Generate negative indices list with the length of the array
    '''
    indices_to_exclude = [current_index, current_index + 1, current_index - 1, current_index + 2, current_index - 2]
    available_indices = np.setdiff1d(np.arange(len(arr)), indices_to_exclude)
    selected_indices = np.random.choice(available_indices, num_values, replace=False)
    selected_values = arr[selected_indices]
    return selected_values

def generate_negative_pair_indices(dataset):
    ''' Generate negative pairs indices and save the indices
    '''
    df = pd.read_csv(f'{DATASET_PATH}{dataset}/train.csv', index_col='index')
    num_rows = df.shape[0]-(timesteps+1)
    arr = np.array(list(range(num_rows)))
    neg_pairs_list = [neg_index_list(arr, i) for i in range(num_rows)]
    with open(f'{AUG_PAIRS_PATH}{dataset}/indices_list.pkl', 'wb') as f:
        pickle.dump(neg_pairs_list, f)

def save_pairs(positive_pairs, negative_pairs, dataset,augmentation_name):
    ''' Input the postive and negative pairs and save them to the pickle files
    '''
    with open(f'{AUG_PAIRS_PATH}{dataset}/PP_{augmentation_name}.pkl', 'wb') as f:
        pickle.dump(positive_pairs, f)
    with open(f'{AUG_PAIRS_PATH}{dataset}/NP_{augmentation_name}.pkl', 'wb') as f:
        pickle.dump(negative_pairs, f)


def generatePairsAEModel(xtrain, dataset, model_name, inverter, hyp_path, model_save_path, save_model=False):
    ''' Generate positive and negative pairs of autoencoder based augmentation
    '''
    with open(f'{AUG_PAIRS_PATH}{dataset}/indices_list.pkl', 'rb') as f:
        neg_pairs_list = pickle.load(f)
    model = trainAEModel(xtrain, dataset, model_name, inverter, hyp_path ,model_save_path,save_model=save_model)
    X = xtrain if model_name=='MLPAE' else flatten(xtrain)
    X_enc = model.predict(xtrain) if model_name=='MLPAE' else flatten(model.predict(xtrain))
    positive_pairs = generate_positive_pairs(X, X_enc)
    negative_pairs = generate_negative_pairs(X, X, neg_pairs_list)
    save_pairs(positive_pairs, negative_pairs, dataset,augmentation_name=model_name)
    return model


def generatePairsSJP(X, dataset, augmentation_name):
    ''' Generate positive and negative pairs of scaling jittering and permutation based augmentation
    '''
    jitter_permuted_data = jitter_time_series(permutation(X))
    negative_pairs = [(i, j) for i, j in zip(X, jitter_permuted_data)]
    scaled_data = scaling_2d(X)
    positive_pairs = [(i, j) for i, j in zip(X, scaled_data)]
    save_pairs(positive_pairs, negative_pairs, dataset,augmentation_name=augmentation_name)

def generatePairsRM(X, dataset, augmentation_name):
    ''' Generate positive and negative pairs of random masking based augmentation
    '''
    with open(f'{AUG_PAIRS_PATH}{dataset}/indices_list.pkl', 'rb') as f:
        neg_pairs_list = pickle.load(f)
    X = X[timesteps+1:] 
    X_masking = [random_masking_array(x) for x in X]
    positive_pairs = generate_positive_pairs(X, X_masking)
    negative_pairs = generate_negative_pairs(X, X_masking, neg_pairs_list)
    save_pairs(positive_pairs, negative_pairs, dataset,augmentation_name=augmentation_name)