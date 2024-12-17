import pandas as pd
import pickle
from cl_functions import compute_accuracy, create_clModel
from plots import plot_accuracy_loss
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from constants import AUG_PAIRS_PATH, CL_AUG_MODEL_SAVE_PATH, CL_AUG_HYP_PATH
from itertools import product

def get_cl_hyperparams(dataset, augmentation):
    ''' Return hyperparameters of the contrastive learning using different augmentations saved in a CSV file'''
    df = pd.read_csv(f'{CL_AUG_HYP_PATH}CL_{dataset}_{augmentation}.csv')
    #df = df_hyp[df_hyp['augmentation']==augmentation]
    return (
        df['epochs'].values[0],
        df['batch_size'].values[0],
        df['layer1'].values[0],
        df['layer2'].values[0],
        df['layer3'].values[0],
        df['drop_layer1'].values[0],
        df['drop_layer2'].values[0],
        df['lr'].values[0],
        df['regularizer'].values[0]
    )
def train_cl_model(dataset, augmentation, save_model:bool=False):
    ''' Train the contrastive learning model using positive and negative pairs obtained from the augmentation technique.
        if save_model is true saves the model else loads the saved model
    '''
    with open(f'{AUG_PAIRS_PATH}{dataset}/PP_{augmentation}.pkl', 'rb') as f:
        positive_pairs = pickle.load(f)
    with open(f'{AUG_PAIRS_PATH}{dataset}/NP_{augmentation}.pkl', 'rb') as f:
        negative_pairs = pickle.load(f)
    print(np.array(positive_pairs).shape)
    np.random.seed(42)
    tf.random.set_seed(42)
    pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
    labels = np.concatenate((np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))), axis=0)
    shuffled_indices = np.random.permutation(pairs.shape[0])
    pairs = pairs[shuffled_indices]
    labels = labels[shuffled_indices]
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.1, shuffle=False)
    X_train_0, X_train_1, X_test_0, X_test_1 = X_train[:, 0], X_train[:, 1], X_test[:, 0], X_test[:, 1]
    xtrain, xtest = [X_train_0, X_train_1], [X_test_0, X_test_1]
    epochs, batch_size, layer1, layer2, layer3, drop_layer1, drop_layer2, lr, regularizer = get_cl_hyperparams(dataset, augmentation)
    model ,base_network= create_clModel(features=positive_pairs[0][0].shape[0], layer1=layer1, layer2=layer2, layer3=layer3, regularizer=regularizer,lr = lr, drop_layer1=drop_layer1, drop_layer2=drop_layer2)
    if save_model:
        history = model.fit(xtrain, y_train,
            batch_size=batch_size,
            epochs=epochs, validation_split=0.3,
            verbose=2).history
        model.save(f'{CL_AUG_MODEL_SAVE_PATH}CL_{dataset}_{augmentation}.h5')
    else:
        model.load_weights(f'{CL_AUG_MODEL_SAVE_PATH}CL_{dataset}_{augmentation}.h5')
    ytr_pred = model.predict(xtrain)
    tr_acc = compute_accuracy(y_train, ytr_pred)
    yte_pred = model.predict(xtest)
    te_acc = compute_accuracy(y_test, yte_pred)
    print(f'Train Accuracy: {tr_acc}')
    print(f'Test Accuracy: {te_acc}')
    return model, base_network, tr_acc, te_acc

def trainCLModelHyp(dataset, augmentation, comb):
    ''' Train the contrastive learning model using positive and negative pairs obtained from the augmentation technique
        with the given parameter grid combination.
    '''
    with open(f'{AUG_PAIRS_PATH}{dataset}/PP_{augmentation}.pkl', 'rb') as f:
        positive_pairs = pickle.load(f)
    with open(f'{AUG_PAIRS_PATH}{dataset}/NP_{augmentation}.pkl', 'rb') as f:
        negative_pairs = pickle.load(f)
    print(np.array(positive_pairs).shape)
    np.random.seed(42)
    tf.random.set_seed(42)
    pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
    labels = np.concatenate((np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))), axis=0)
    shuffled_indices = np.random.permutation(pairs.shape[0])
    pairs = pairs[shuffled_indices]
    labels = labels[shuffled_indices]
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.1, shuffle=False)
    X_train_0, X_train_1, X_test_0, X_test_1 = X_train[:, 0], X_train[:, 1], X_test[:, 0], X_test[:, 1]
    xtrain, xtest = [X_train_0, X_train_1], [X_test_0, X_test_1]
    epochs, batch_size, layer1, layer2, layer3, drop_layer1, drop_layer2, lr, regularizer = comb
    model ,base_network= create_clModel(features=positive_pairs[0][0].shape[0], layer1=layer1, layer2=layer2, layer3=layer3, regularizer=regularizer,lr = lr, drop_layer1=drop_layer1, drop_layer2=drop_layer2)
    history = model.fit(xtrain, y_train,
        batch_size=batch_size,
        epochs=epochs, validation_split=0.3,
        verbose=False).history
    y_pred = model.predict(xtrain)
    tr_acc = compute_accuracy(y_train, y_pred)
    y_pred = model.predict(xtest)
    te_acc = compute_accuracy(y_test, y_pred)
    print(f'Train Accuracy: {tr_acc}')
    print(f'Test Accuracy: {te_acc}')
    return model, base_network, tr_acc, te_acc

def tuneCLModelHyp(dataset, augmentation, param_grid):
    ''' Train the CL model with the parameter grid and the augmentation type and save the parameter combination
        that has highest test accuracy.
    '''

    all_combinations = list(product(*param_grid.values()))
    com = 0
    prev_acc = 0
    for combination in all_combinations:
        epochs, batch_size, layer1, layer2, layer3, drop_layer1, drop_layer2, lr, regularizer = combination
        model, base_network, tr_acc, te_acc = trainCLModelHyp(dataset, augmentation, combination)
        df = pd.DataFrame({'file':[f'CL_{dataset}_{augmentation}'],'epochs': epochs, 'batch_size': batch_size, 'layer1': layer1, 'layer2': layer2,
                                                     'layer3': layer3, 'drop_layer1':drop_layer1,'drop_layer2':drop_layer2,'lr':lr,'regularizer':regularizer,
                                                     'accuracy_tr': tr_acc, 'accuracy_te': te_acc})
        if prev_acc<te_acc:
            prev_acc = te_acc
            model.save(f'{CL_AUG_MODEL_SAVE_PATH}CL_{dataset}_{augmentation}.h5')
            df.to_csv(f'{CL_AUG_HYP_PATH}CL_{dataset}_{augmentation}.csv')

