import pandas as pd
pd.set_option("display.max_columns", None)
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def plot_clusters_TSNE(X, ytrue, ypred, path):
    ''' Generate plots with TSNE taking input data with true and predicted labels
    '''

    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)

    fig, ax = plt.subplots(ncols=2, figsize=(20,10), facecolor='white')
    legend_labels = {'Normal': 'green', 'Anomaly': 'red'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=legend_labels[label], markersize=10, label=label) for label in legend_labels]
    colors1 = ['red' if x==1 else 'green' for x in ytrue]
    ax[0].scatter(X_embedded[:,0], X_embedded[:,1], c=colors1)
    ax[0].set_title('True Labels', fontsize=14)
    ax[0].legend(handles=handles, loc='upper right', fontsize=14)
    
    colors2 = ['red' if x==1 else 'green' for x in ypred]
    ax[1].scatter(X_embedded[:,0], X_embedded[:,1], c=colors2)
    ax[1].set_title('Predicted Labels', fontsize=14)
    ax[1].legend(handles=handles, loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{path}.pdf', dpi=100)
    plt.savefig(f'{path}.png', dpi=100)

def plot_kdist_graph(sorted_distances, path):
    ''' Generate the k-distance graph fom the generated k distances    
    '''
    plt.figure(figsize=(10, 7), facecolor='white')
    plt.plot(sorted_distances, label='k-distances')
    plt.xlabel('Samples', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.title('k-Distance Graph', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(f'{path}.pdf', dpi=100)
    plt.savefig(f'{path}.png', dpi=100)
    
def plot_silhouttescores(scores, path):
    ''' generate a 3D plot with silhouette scores 
    '''
    eps_values = [key[0] for key in scores.keys()]
    min_samples_values = [key[1] for key in scores.keys()]
    silhouette_scores = scores.values()

    max_value = max(scores.values())
    max_keys = [key for key, value in scores.items() if value == max_value]
    best_eps, best_min_samples = max_keys[0]

    best_silhouette_score = max(scores.values())
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(eps_values, silhouette_scores, min_samples_values, c='b', marker='o')
    ax.scatter(best_eps, best_silhouette_score, best_min_samples, c='r', marker='o', s=200, label=f'Best Point at eps = {best_eps:.4f}, min_samples = {best_min_samples}, silhouette_score = {best_silhouette_score:.4f}')
    ax.set_xlabel('Epsilon', fontsize=14)
    ax.set_zlabel('Min Samples', fontsize=14)
    ax.set_ylabel('Silhouette Scores', fontsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{path}.pdf', dpi=100)
    plt.savefig(f'{path}.png', dpi=100)

def plot_accuracy_loss(model_fit, path):
        ''' generate accuracy loss curve with the given history of the model
        '''
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), facecolor='white')
        ax1.plot(model_fit['accuracy'], label="Train")
        ax1.plot(model_fit['val_accuracy'], label="Validation")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy')
        ax1.legend()

        ax2.plot(model_fit['loss'], label="Train")
        ax2.plot(model_fit['val_loss'], label="Validation")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f'{path}.pdf', dpi=100)
        plt.savefig(f'{path}.png', dpi=100)
        plt.show()
def plot_threshold(scores, ytrue, ypred, decision_point, path):   
    ''' generate decision plot
    '''
    plt_df = pd.DataFrame({'test_mses':scores, 'ytrue':ytrue, 'ypred':ypred})
    fig, ax = plt.subplots(ncols=2,figsize=(20,10), facecolor='white')
    ax[0].scatter(plt_df.index, plt_df['test_mses'], color = 'green', alpha = 1, label = 'Anomaly Scores')
    ax[0].scatter(plt_df[plt_df['ytrue']==1].index, plt_df[plt_df['ytrue']==1]['test_mses'], color = 'red', label='True Anomalies', alpha=1)
    ax[0].set_xlabel('Samples', fontsize=14)
    ax[0].set_ylabel('Anomaly Scores', fontsize=14)
    ax[0].legend(fontsize=14, loc='upper right')

    ax[1].scatter(plt_df.index, plt_df['test_mses'], color = 'green', alpha = 1, label = 'Anomaly Scores')
    ax[1].scatter(plt_df[plt_df['ypred']==1].index, plt_df[plt_df['ypred']==1]['test_mses'],   color = 'red', label='Predicted Anomalies', alpha=1)
    ax[1].axhline(y=decision_point, color='black', linestyle='--', label=f'Threshold at {decision_point:.6f}')
    ax[1].set_xlabel('Samples', fontsize=14)
    ax[1].set_ylabel('Anomaly Scores', fontsize=14)
    ax[1].legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{path}.png', dpi=100)
    plt.savefig(f'{path}.pdf', dpi=100)

def plot_threshold_withrange(test_mses, ytrue, ypred, threshold, yrange, path):   
    plt_df = pd.DataFrame({'test_mses':test_mses, 'ytrue':ytrue, 'ypred':ypred})
    plt_df = plt_df[plt_df['test_mses']<yrange]
    fig, ax = plt.subplots(ncols=2,figsize=(20,10), facecolor='white')
    ax[0].scatter(plt_df.index, plt_df['test_mses'], color = 'green', alpha = 1, label = 'Reconstruction Errors')
    ax[0].scatter(plt_df[plt_df['ytrue']==1].index, plt_df[plt_df['ytrue']==1]['test_mses'], color = 'red', label='True Anomalies', alpha=1)
    ax[0].set_xlabel('Samples', fontsize=14)
    ax[0].set_ylabel('Reconstruction Errors', fontsize=14)
    ax[0].legend(fontsize=14, loc='upper right')

    ax[1].scatter(plt_df.index, plt_df['test_mses'], color = 'green', alpha = 1, label = 'Reconstruction Errors')
    ax[1].scatter(plt_df[plt_df['ypred']==1].index, plt_df[plt_df['ypred']==1]['test_mses'],   color = 'red', label='Predicted Anomalies', alpha=1)
    ax[1].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold at {threshold:.4f}')
    ax[1].set_xlabel('Samples', fontsize=14)
    ax[1].set_ylabel('Reconstruction Errors', fontsize=14)
    ax[1].legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{path}.png', dpi=100)
    plt.savefig(f'{path}.pdf', dpi=100)

def plot_loss_curve(model_fit, path):
        '''plot loss curves with the given history of the model
        '''
        plt.figure(figsize=(10, 7), facecolor='white')
        plt.plot(model_fit['loss'], label="Train")
        plt.plot(model_fit['val_loss'], label="Validation")
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Loss', fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{path}.png', dpi=100)
        plt.savefig(f'{path}.pdf', dpi=100)
        plt.show()
