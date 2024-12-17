import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, make_scorer, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from plots import plot_silhouttescores, plot_kdist_graph
import ast
from joblib import Parallel, delayed


def getEps_K_dist(X, features, path, bias=0):
    ''' calculate and generate k-distance plot
    '''
    k = features
    nn_model = NearestNeighbors(n_neighbors=2*k)
    nn_model.fit(X)
    distances, _ = nn_model.kneighbors(X)
    k_distances = np.sort(distances[:, k])
    sorted_distances = np.sort(np.ravel(k_distances))[::-1]
    plot_kdist_graph(sorted_distances[bias:], path)

# def getEps_MinPts_SilScore(X, features, range_limit, eps_range, path):
#     ''' Get the best eps and minPts with best silhouette score
#     '''
#     range_start = features-range_limit
#     min_samples_range = range(3 if range_start<2 else range_start, 2*features+range_limit)
#     scores = {}
#     # Grid search
#     for eps in eps_range:
#         for min_samples in min_samples_range:
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             labels = dbscan.fit_predict(X)

#             # Check if more than one cluster is found
#             if len(np.unique(labels))>1:
#                 silhouette = silhouette_score(X, labels)
#                 scores[(eps, min_samples)] = silhouette
    
#     max_value = max(scores.values())
#     max_keys = [key for key, value in scores.items() if value == max_value]
#     best_eps, best_min_samples = max_keys[0]
#     plot_silhouttescores(scores=scores, path=path)
#     return best_eps, best_min_samples, scores

def evaluate_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        return (eps, min_samples), silhouette
    return (eps, min_samples), -1

def getEps_MinPts_SilScore(X, features, range_limit, eps_range, path, n_jobs=-1):
    ''' Get the best eps and minPts with best silhouette score
    '''
    range_start = features - range_limit
    min_samples_range = range(3 if range_start < 2 else range_start, 2 * features + range_limit)
    
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_dbscan)(X, eps, min_samples) for eps in eps_range for min_samples in min_samples_range)
    
    scores = {key: value for key, value in results if value != -1}
    
    max_value = max(scores.values())
    max_keys = [key for key, value in scores.items() if value == max_value]
    best_eps, best_min_samples = max_keys[0]
    plot_silhouette_scores(scores, path)
    return best_eps, best_min_samples, scores


def cosine_distance_to_center(X, center):
    ''' calculate the distance from test points to the center of a given cluster center
    '''
    dot_products = np.dot(X, center)
    norms_X = np.linalg.norm(X, axis=1)
    norm_center = np.linalg.norm(center)
    cosine_similarity = dot_products / (norms_X * norm_center)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def mahalanobis_distance_to_center(X, center):
    ''' calculate mahalanobis distance from test points to the center of a given cluster center
    '''
    covariance_matrix = np.cov(X, rowvar=False) 
    diff = X - center
    mahalanobis_distance = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(covariance_matrix)) * diff, axis=1))
    return mahalanobis_distance


# def getMetrics(ytrue:np.array, ypred:np.array,yscore: np.array, dataset, inverter, model):
#     ''' calculate metrics for the binary classification
#     '''
#     f1 = f1_score(y_true=ytrue, y_pred=ypred)
#     acc = accuracy_score(y_true=ytrue, y_pred=ypred)
#     prec = precision_score(y_true=ytrue, y_pred=ypred)
#     rec = recall_score(y_true=ytrue, y_pred=ypred)
#     roc_auc = roc_auc_score(y_true=ytrue, y_score=ypred)

#     # Calculate precision-recall curve
#     precision, recall, _ = precision_recall_curve(y_true=ytrue, probas_pred=yscore)
#     pr_auc = auc(recall, precision)
    
#     cm = confusion_matrix(ytrue, ypred)
#     TN = cm[0, 0]
#     FP = cm[0, 1]
#     FPR = FP / (FP + TN)
#     TP = cm[1, 1]
#     FN = cm[1, 0]
#     TPR = TP / (TP + FN)
#     print('Binary Scores:')
#     print('F1-score: ', f1)
#     print('Accuracy: ', acc)
#     print('Precision: ', prec)
#     print('Recall: ', rec)
#     print('FPR:', FPR)
#     print('TPR:', TPR)
#     print('ROC-AUC: ', roc_auc)
#     print('PR-AUC: ', pr_auc)
#     df = pd.DataFrame({'dataset':[dataset],'file':inverter,'model':model, 'f1':f1, 'acc':acc, 'prec':prec, 'rec':rec, 'roc_auc':roc_auc, 'pr_auc':pr_auc,'FPR':FPR, 'TPR':TPR})
#     return df


def getMetrics(ytrue: np.array, ypred: np.array, yscore: np.array, dataset, inverter, model, save_arrays=False, weighted=False):
    ''' Calculate metrics for the binary classification with and without optimizing for the best F1 score '''
    
    average = 'weighted' if weighted else 'binary'
    
    # Initialize metrics with NA
    metrics = {
        'f1': np.nan, 'acc': np.nan, 'prec': np.nan, 'rec': np.nan,
        'roc_auc': np.nan, 'pr_auc': np.nan, 'FPR': np.nan, 'TPR': np.nan,
        'best_f1': np.nan, 'best_acc': np.nan, 'best_prec': np.nan, 'best_rec': np.nan,
        'best_FPR': np.nan, 'best_TPR': np.nan, 'best_threshold': np.nan
    }

    if len(np.unique(ytrue)) > 1:  # Proceed only if there are both classes in ytrue
        # Calculate metrics with or without weighted averaging
        metrics['f1'] = f1_score(y_true=ytrue, y_pred=ypred, average=average)
        metrics['acc'] = accuracy_score(y_true=ytrue, y_pred=ypred)
        metrics['prec'] = precision_score(y_true=ytrue, y_pred=ypred, average=average)
        metrics['rec'] = recall_score(y_true=ytrue, y_pred=ypred, average=average)
        metrics['roc_auc'] = roc_auc_score(y_true=ytrue, y_score=yscore)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true=ytrue, probas_pred=yscore)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Compute confusion matrix and related metrics
        cm = confusion_matrix(ytrue, ypred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        metrics['FPR'] = FP / (FP + TN) if (FP + TN) > 0 else 0
        TP = cm[1, 1]
        FN = cm[1, 0]
        metrics['TPR'] = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate metrics with optimizing for the best F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_index = f1_scores.argmax()
        metrics['best_f1'] = f1_scores[best_f1_index]
        metrics['best_threshold'] = thresholds[best_f1_index] if best_f1_index < len(thresholds) else 1.0
        metrics['best_prec'] = precision[best_f1_index]
        metrics['best_rec'] = recall[best_f1_index]

        # Generate predictions based on the best threshold
        best_ypred = (yscore >= metrics['best_threshold']).astype(int)

        # Calculate confusion matrix and other metrics for best F1
        metrics['best_acc'] = accuracy_score(y_true=ytrue, y_pred=best_ypred)
        best_cm = confusion_matrix(ytrue, best_ypred)
        best_TN = best_cm[0, 0]
        best_FP = best_cm[0, 1]
        metrics['best_FPR'] = best_FP / (best_FP + best_TN) if (best_FP + best_TN) > 0 else 0
        best_TP = best_cm[1, 1]
        best_FN = best_cm[1, 0]
        metrics['best_TPR'] = best_TP / (best_TP + best_FN) if (best_TP + best_FN) > 0 else 0
    else:
        # Calculate accuracy when only one class is present
        metrics['acc'] = accuracy_score(y_true=ytrue, y_pred=ypred)
    
    # print('Binary Scores:' if not weighted else 'Weighted Scores:')
    # for key, value in metrics.items():
    #     print(f'{key}: {value}')
    
    metrics_df = pd.DataFrame({
        'dataset': [dataset],
        'file': [inverter],
        'model': [model],
        'f1': [metrics['f1']],
        'acc': [metrics['acc']],
        'prec': [metrics['prec']],
        'rec': [metrics['rec']],
        'roc_auc': [metrics['roc_auc']],
        'pr_auc': [metrics['pr_auc']],
        'FPR': [metrics['FPR']],
        'TPR': [metrics['TPR']],
        'best_f1': [metrics['best_f1']],
        'best_acc': [metrics['best_acc']],
        'best_prec': [metrics['best_prec']],
        'best_rec': [metrics['best_rec']],
        'best_FPR': [metrics['best_FPR']],
        'best_TPR': [metrics['best_TPR']],
        'best_threshold': [metrics['best_threshold']]
    })

    if save_arrays:
        arrays_df = pd.DataFrame({
            'ytrue': [ytrue],
            'ypred': [ypred],
            'yscore': [yscore]
        })
        return metrics_df, arrays_df
    else:
        return metrics_df


# def getMetrics(ytrue: np.array, ypred: np.array, yscore: np.array, dataset, inverter, model, save_arrays=False, weighted=False):
#     ''' Calculate metrics for the binary classification with and without optimizing for the best F1 score '''
    
#     average = 'weighted' if weighted else 'binary'
    
#     # Calculate metrics with or without weighted averaging
#     f1 = f1_score(y_true=ytrue, y_pred=ypred, average=average)
#     acc = accuracy_score(y_true=ytrue, y_pred=ypred)
#     prec = precision_score(y_true=ytrue, y_pred=ypred, average=average)
#     rec = recall_score(y_true=ytrue, y_pred=ypred, average=average)
#     roc_auc = roc_auc_score(y_true=ytrue, y_score=yscore)
    
#     # Calculate precision-recall curve
#     precision, recall, thresholds = precision_recall_curve(y_true=ytrue, probas_pred=yscore)
#     pr_auc = auc(recall, precision)
    
#     # Compute confusion matrix and related metrics
#     cm = confusion_matrix(ytrue, ypred)
#     TN = cm[0, 0]
#     FP = cm[0, 1]
#     FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#     TP = cm[1, 1]
#     FN = cm[1, 0]
#     TPR = TP / (TP + FN) if (TP + FN) > 0 else 0

#     # Calculate metrics with optimizing for the best F1 score
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#     best_f1_index = f1_scores.argmax()
#     best_f1 = f1_scores[best_f1_index]
#     best_threshold = thresholds[best_f1_index] if best_f1_index < len(thresholds) else 1.0
#     best_precision = precision[best_f1_index]
#     best_recall = recall[best_f1_index]

#     # Generate predictions based on the best threshold
#     best_ypred = (yscore >= best_threshold).astype(int)

#     # Calculate confusion matrix and other metrics for best F1
#     best_acc = accuracy_score(y_true=ytrue, y_pred=best_ypred)
#     best_cm = confusion_matrix(ytrue, best_ypred)
#     best_TN = best_cm[0, 0]
#     best_FP = best_cm[0, 1]
#     best_FPR = best_FP / (best_FP + best_TN) if (best_FP + best_TN) > 0 else 0
#     best_TP = best_cm[1, 1]
#     best_FN = best_cm[1, 0]
#     best_TPR = best_TP / (best_TP + best_FN) if (best_TP + best_FN) > 0 else 0

#     print('Binary Scores:' if not weighted else 'Weighted Scores:')
#     print('F1-score: ', f1)
#     print('Accuracy: ', acc)
#     print('Precision: ', prec)
#     print('Recall: ', rec)
#     print('FPR:', FPR)
#     print('TPR:', TPR)
#     print('ROC-AUC: ', roc_auc)
#     print('PR-AUC: ', pr_auc)
#     print('Best F1-score: ', best_f1)
#     print('Best Accuracy: ', best_acc)
#     print('Best Precision: ', best_precision)
#     print('Best Recall: ', best_recall)
#     print('Best FPR:', best_FPR)
#     print('Best TPR:', best_TPR)
#     print('Best Threshold: ', best_threshold)
    
#     metrics_df = pd.DataFrame({
#         'dataset': [dataset],
#         'file': [inverter],
#         'model': [model],
#         'f1': [f1],
#         'acc': [acc],
#         'prec': [prec],
#         'rec': [rec],
#         'roc_auc': [roc_auc],
#         'pr_auc': [pr_auc],
#         'FPR': [FPR],
#         'TPR': [TPR],
#         'best_f1': [best_f1],
#         'best_acc': [best_acc],
#         'best_prec': [best_precision],
#         'best_rec': [best_recall],
#         'best_FPR': [best_FPR],


        
#         'best_TPR': [best_TPR],
#         'best_threshold': [best_threshold]
#     })

#     if save_arrays:
#         arrays_df = pd.DataFrame({
#             'ytrue': [ytrue],
#             'ypred': [ypred],
#             'yscore': [yscore]
#         })
#         return metrics_df, arrays_df
#     else:
#         return metrics_df



# def getWeightedMetrics(ytrue:np.array, ypred:np.array,yscore: np.array, dataset, inverter, model):
#     ''' calculate weighted scores for the class imbalance
#     '''
#     f1 = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
#     acc = accuracy_score(y_true=ytrue, y_pred=ypred)
#     prec = precision_score(y_true=ytrue, y_pred=ypred, average='weighted')
#     rec = recall_score(y_true=ytrue, y_pred=ypred, average='weighted')
#     roc_auc = roc_auc_score(y_true=ytrue, y_score=ypred, average='weighted')
    
#     # Calculate precision-recall curve
#     precision, recall, _ = precision_recall_curve(y_true=ytrue, probas_pred=yscore)
#     pr_auc = auc(recall, precision)
    
#     cm = confusion_matrix(ytrue, ypred)
#     TN = cm[0, 0]
#     FP = cm[0, 1]
#     FPR = FP / (FP + TN)
#     TP = cm[1, 1]
#     FN = cm[1, 0]
#     TPR = TP / (TP + FN)
#     print('Weighted Scores:')
#     print('F1-score: ', f1)
#     print('Accuracy: ', acc)
#     print('Precision: ', prec)
#     print('Recall: ', rec)
#     print('FPR:', FPR)
#     print('TPR:', TPR)
#     print('ROC-AUC: ', roc_auc)
#     print('PR-AUC: ', pr_auc)
#     df = pd.DataFrame({'dataset':[dataset],'file':inverter,'model':model, 'f1':f1, 'acc':acc, 'prec':prec, 'rec':rec, 'roc_auc':roc_auc, 'pr_auc':pr_auc,'FPR':FPR, 'TPR':TPR})
#     return df
    
def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)
    
# def temporalize(X, timesteps=8):
#     output_X = []
#     for i in range(len(X) - timesteps - 1):
#         t = []
#         for j in range(1, timesteps + 1):
#             t.append(X[[(i + j + 1)], :])
#         output_X.append(t)
#     return np.squeeze(np.array(output_X))


def temporalize(X, timesteps=8):
    output_X = []
    for i in range(len(X) - timesteps):
        t = []
        for j in range(timesteps):
            t.append(X[i + j, :])
        output_X.append(t)
    return np.array(output_X)


def mask_sequences(sequences, mask_fraction=0.3):
    masked_sequences = sequences.copy()
    num_timesteps = sequences.shape[1]
    num_features = sequences.shape[2]

    for seq in masked_sequences:
        num_masked = int(num_timesteps * mask_fraction)
        
       #print('num_masked', num_masked)
        mask_indices = np.random.choice(num_timesteps, num_masked, replace=False)
        #print(len(mask_indices))
        
        for idx in mask_indices:
            seq[idx, :] = 0  # Masking by setting to zeros
            #print(f"Masking timestep {idx} in sequence")
        
    return masked_sequences

# def getHyperParams(file, df_hyp, model):
#     param_dict = ast.literal_eval(df_hyp.loc[(df_hyp['file'] == file) & (df_hyp['model'] == model)]['Hyperparameters'].values[0])
#     return param_dict['batch_size'], param_dict['encoding_dim'], param_dict['epochs'],param_dict['layer1'], param_dict['layer2'] 
    
def getHyperParams(file, df_hyp, inverter, model):
    print(df_hyp)
    if pd.isna(df_hyp['model'][0]):
        df_split = df_hyp['file'].str.extract(r'([^,]+),([^,]+),([^,]+),(.+)') # extracting columns from single column csv
        df_split.columns = ['file', 'inverter', 'model', 'Hyperparameters']
        print(df_split)
        param_dict = ast.literal_eval(df_split.loc[(df_split['inverter'] == str('inv') + str('_')+str(inverter)) & (df_split['model'] == model)]['Hyperparameters'].values[0])
        param_dict = ast.literal_eval(param_dict)
        return param_dict['batch_size'], param_dict['encoding_dim'], param_dict['epochs'],param_dict['layer1'], param_dict['layer2']
    else:
        param_dict = ast.literal_eval(df_hyp.loc[(df_hyp['file'] == str('inv') + str('_')+str(inverter)) & (df_hyp['model'] == model)]['Hyperparameters'].values[0])
        return param_dict['batch_size'], param_dict['encoding_dim'], param_dict['epochs'],param_dict['layer1'], param_dict['layer2']
        


def CustomScoring(y_true, y_pred):
    return mean_squared_error(y_true=flatten(y_true),y_pred = flatten(y_pred))
scorer = make_scorer(CustomScoring, greater_is_better=False)
