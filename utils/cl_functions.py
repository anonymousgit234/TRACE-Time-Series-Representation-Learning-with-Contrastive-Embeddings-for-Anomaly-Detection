from keras.models import Model
import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
import tensorflow as tf
from collections import Counter
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, BatchNormalization
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from ae_utils import * 
from sklearn.cluster import KMeans


predictions_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/ModelPredictions/Predictions_with_negatives_on_randomseq/'
CL_PLOTS_PATH = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Plots/CL/'

def create_base_network(input_shape, layer1=32, layer2=16, layer3=8, drop_layer1=0.5, drop_layer2=0.5, regularizer=0.1):
    '''Encoder network of siames architecture.
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(layer1, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = Dropout(drop_layer1)(x)
    x = BatchNormalization()(x)
    x = Dense(layer2, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = Dropout(drop_layer2)(x)
    x = BatchNormalization()(x)
    x = Dense(layer3, activation='sigmoid', kernel_initializer=tf.keras.initializers.HeNormal(),kernel_regularizer=tf.keras.regularizers.l1_l2(regularizer))(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)



# def accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.'''
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), tf.float32))
    
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_similarity(vectors):
    '''Calculate cosine similarity between two vectors.'''
    x, y = vectors
    x = tf.linalg.l2_normalize(x, axis=-1)
    y = tf.linalg.l2_normalize(y, axis=-1)
    return tf.reduce_sum(x * y, axis=1, keepdims=True)

# def cosine_similarity(vectors):
#     '''calculate cosine similarity between two vectors
#     '''
#     x, y = vectors
#     x = K.l2_normalize(x, axis=-1)
#     y = K.l2_normalize(y, axis=-1)
#     return K.sum(x * y, axis=1, keepdims=True)
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss function.'''
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# def contrastive_loss(y_true, y_pred):
#     '''contrastive loss function
#     '''
#     margin = 1
#     square_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_clModel(features=68, layer1=32, layer2=16, layer3=8, drop_layer1=0.1, drop_layer2=0.1, regularizer=0.001, lr = 0.02):
    '''Contrastive learning network using siamese architecture with cosine similarity measures and contrastive loss
    '''
    input_shape = (features,)
    base_network = create_base_network(input_shape=input_shape,layer1=layer1, layer2=layer2, layer3=layer3, drop_layer1=drop_layer1, drop_layer2=drop_layer2, regularizer=regularizer)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(cosine_similarity,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    layer = Dense(1, activation='sigmoid')(distance)
    model = Model([input_a, input_b], layer)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=[accuracy])
    return model, base_network


def reshape_tensor(X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X

######### CL pipeline ##########
## augmentations for positive pairs
import numpy as np
def scale_sequence(seq, scale_factor=0.1):
    return seq * (1 + scale_factor * np.random.randn(*seq.shape))

def jitter_sequence(seq, noise_level=0.01):
    return seq + noise_level * np.random.randn(*seq.shape)

def permute_sequence_pos(seq):
    permuted_indices = np.random.permutation(len(seq))
    return seq[permuted_indices]

def augment_positive_sequences(reconstructions, n_augmentations=5, scaling=True, jittering=True, permutation=True):
    augmented_sequences = []

    for seq in reconstructions:
        augmented_seq_list = []
        for _ in range(n_augmentations):
            augmented_seq = seq.copy()
            if scaling:
                augmented_seq = scale_sequence(augmented_seq)
            if jittering:
                augmented_seq = jitter_sequence(augmented_seq)
            if permutation:
                augmented_seq = permute_sequence_pos(augmented_seq)
            augmented_seq_list.append(augmented_seq)
        augmented_sequences.append(augmented_seq_list)
    
    return np.array(augmented_sequences)

def augment_positive_sequences_new(reconstructions, n_augmentations=3, scaling=False, jittering=True, permutation=False):
    augmented_sequences = []

    # Iterate over each sequence in the reconstructions
    for seq in reconstructions:
        # Start with the original sequence
        augmented_seq_list = [seq]
        
        # Generate additional augmentations by jittering
        for _ in range(n_augmentations - 1):  # n_augmentations includes the original, so we do n_augmentations-1
            augmented_seq = seq.copy()
            
            if jittering:
                augmented_seq = jitter_sequence(augmented_seq)
            
            # Append each jittered sequence to the augmented_seq_list
            augmented_seq_list.append(augmented_seq)
        
        # Append the list of augmented sequences to the main list
        augmented_sequences.append(augmented_seq_list)
    print('only jittering for pos augs')
    # Convert the list of augmented sequences to a NumPy array
    return np.array(augmented_sequences)


# Time Warping
def time_warp(arr, n_augmentations=3, sigma=0.2):
    n_samples, sequence_length, n_features = arr.shape
    time_steps = np.arange(sequence_length)
    
    # Initialize an empty array to store the augmented series
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        random_warp = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, sequence_length))
        warping_factors = np.cumsum(random_warp, axis=1)
        
        for i in range(n_samples):
            interp_func = interp1d(warping_factors[i], arr[i], axis=0, fill_value="extrapolate")
            augmented_series[i, aug] = interp_func(time_steps)
    
    return augmented_series

# Time Shifting

def time_shift(arr, n_augmentations=3, max_shift=10):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        shifts = np.random.randint(-max_shift, max_shift, n_samples)
        
        for i in range(n_samples):
            shifted_series = np.roll(arr[i], shifts[i], axis=0)
            if shifts[i] > 0:
                shifted_series[:shifts[i], :] = 0
            elif shifts[i] < 0:
                shifted_series[shifts[i]:, :] = 0
            augmented_series[i, aug] = shifted_series
    
    return augmented_series


def time_mask(arr, n_augmentations=3, mask_ratio=0.4):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    mask_len = int(mask_ratio * sequence_length)
    
    for aug in range(n_augmentations):
        masked_arr = arr.copy()
        
        for i in range(n_samples):
            mask_start = np.random.randint(0, sequence_length - mask_len)
            masked_arr[i, mask_start:mask_start+mask_len, :] = 0
            augmented_series[i, aug] = masked_arr[i]
    
    return augmented_series

def jitter(arr, n_augmentations=3, sigma=0.01):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        noise = np.random.normal(loc=0, scale=sigma, size=arr.shape)
        jittered_arr = arr + noise
        augmented_series[:, aug, :, :] = jittered_arr
    
    return augmented_series

def scaling(arr, n_augmentations=3, sigma=0.1):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        scaling_factors = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, 1, n_features))
        scaled_arr = arr * scaling_factors
        augmented_series[:, aug, :, :] = scaled_arr
    
    return augmented_series

def permute_sequence(arr, n_augmentations=3):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        permuted_series = []
        for i in range(n_samples):
            permuted_indices = np.random.permutation(sequence_length)
            permuted_series.append(arr[i][permuted_indices])
        augmented_series[:, aug, :, :] = np.array(permuted_series)
    
    return augmented_series

def frequency_shift(arr, n_augmentations=3, shift_factor=2):
    n_samples, sequence_length, n_features = arr.shape
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        shifted_arr = []
        
        for i in range(n_samples):
            freq_shifted = np.fft.fft(arr[i], axis=0)
            freq_shifted = np.roll(freq_shifted, shift_factor, axis=0)
            shifted_series = np.fft.ifft(freq_shifted, axis=0).real
            shifted_arr.append(shifted_series)
        
        augmented_series[:, aug, :, :] = np.array(shifted_arr)
    
    return augmented_series


def window_slicing(arr, n_augmentations=3, slice_ratio=0.9):
    n_samples, sequence_length, n_features = arr.shape
    slice_len = int(slice_ratio * sequence_length)
    augmented_series = np.zeros((n_samples, n_augmentations, sequence_length, n_features))
    
    for aug in range(n_augmentations):
        sliced_arr = []
        
        start_idx = np.random.randint(0, sequence_length - slice_len, n_samples)
        for i in range(n_samples):
            sliced_series = arr[i, start_idx[i]:start_idx[i] + slice_len, :]
            padded_series = np.zeros_like(arr[i])
            padded_series[:slice_len, :] = sliced_series
            sliced_arr.append(padded_series)
        
        augmented_series[:, aug, :, :] = np.array(sliced_arr)
    
    return augmented_series


# def hard_negative_mining_with_error(data, reconstructions, top_k=3):
#     # Calculate reconstruction errors for each sequence
#     errors = np.mean(np.square(data - reconstructions), axis=(1, 2))  # Reconstruction error per sequence   
#     hard_negatives = []
#     # Loop through each sequence's reconstruction error
#     for i, error in enumerate(errors):
#         # Find indices of the top_k highest reconstruction errors
#         hard_neg_indices = np.argsort(-errors)[0:top_k]
#         hard_negatives.append(data[hard_neg_indices])
    
#     return np.array(hard_negatives)


def hard_negative_mining_with_error(data, reconstructions, top_k=3):
    hard_negatives = []

    for i, seq in enumerate(data):
        # Calculate the reconstruction errors for all sequences
        errors = np.mean(np.square(data - reconstructions), axis=(1, 2))
        
        # Sort indices based on errors and exclude the current sequence itself
        hard_neg_indices = np.argsort(-errors)[1:top_k+1]
        
        # Collect the hard negatives specific to the current sequence
        hard_negatives.append(data[hard_neg_indices])
    
    return np.array(hard_negatives)


def hard_negative_mining(data, reconstructions, top_k=3):#
    print('inside hard negative mining with raw dists function')
    
    hard_negatives = []

    for i, seq in enumerate(data):
        # Calculate pairwise distances to other sequences based on reconstruction error
        distances = pairwise_distances(seq.reshape(1, -1), data.reshape(data.shape[0], -1)).flatten()
        
        # Get indices of top_k farthest sequences excluding the sequence itself
        hard_neg_indices = np.argsort(-distances)[0:top_k] 
        ## considering hard negatives as the indices with top 5 highest reconbstruction errors

        #print(hard_neg_indices)
        # reversing the order of elements
        #hard_neg_indices = np.flip(hard_neg_indices)
        #print(hard_neg_indices)
        hard_negatives.append(data[hard_neg_indices])
    
    return np.array(hard_negatives)



def hard_negative_mining_with_random(data, num_negatives=3):
    """
    Select random negative samples from the dataset for each sequence.

    Parameters:
    - data: np.array, shape (num_sequences, sequence_length, feature_dim)
        The dataset containing sequences.
    - num_negatives: int
        The number of random negative samples to pick for each sequence.

    Returns:
    - random_negatives: np.array, shape (num_sequences, num_negatives, sequence_length, feature_dim)
        Array containing the random negative samples for each sequence.
    """
    num_sequences = data.shape[0]
    random_negatives = []

    for i in range(num_sequences):
        # Generate random indices for negative samples, excluding the current sequence
        neg_indices = np.random.choice(np.delete(np.arange(num_sequences), i), num_negatives, replace=False)
        random_negatives.append(data[neg_indices])

    return np.array(random_negatives)

def create_pairs_with_augmentations(data, augmented_reconstructions, hard_negatives):
    pairs = []
    labels = []

    for i in range(len(data)):
        # Augmented positive pairs (anchor, positive)
        for aug_rec in augmented_reconstructions[i]:
            pairs.append([data[i], aug_rec])
            labels.append(1)  # Positive pair

        # Hard negative pairs (anchor, negative)
        for hard_neg in hard_negatives[i]:
            pairs.append([data[i], hard_neg])
            labels.append(0)  # Negative pair

    pairs = np.array(pairs)
    labels = np.array(labels)

    return pairs, labels

def temporal_mixup(data, alpha=0.2):
    n_samples = data.shape[0]
    indices = np.random.permutation(n_samples)
    
    mixed_data = np.zeros_like(data)
    mixed_labels = np.zeros(n_samples)
    
    for i in range(n_samples):
        j = indices[i]
        
        lam = np.random.beta(alpha, alpha)
        
        mixed_data[i] = lam * data[i] + (1 - lam) * data[j]
        mixed_labels[i] = lam
    
    return mixed_data, mixed_labels




# def create_pairs_with_mixup_and_augmentations(data, augmented_reconstructions, hard_negatives, alpha=0.2):
#     pairs = []
#     labels = []

#     for i in range(len(data)):
#         # Augmented positive pair (anchor, positive)
#         for aug_recon in augmented_reconstructions[i]:
#             pairs.append([data[i], aug_recon])
#             labels.append(1)
        
#         # Hard negative pairs (anchor, negative)
#         for hard_neg in hard_negatives[i]:
#             pairs.append([data[i], hard_neg])
#             labels.append(0)
    
#     # Separate positive and negative pairs
#     positive_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
#     negative_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]
    
#     # Apply temporal mixup separately
#     pos_data = np.array([pair[1] for pair in positive_pairs])
#     mixed_pos_data, _ = temporal_mixup(pos_data, alpha=alpha)
    
#     neg_data = np.array([pair[1] for pair in negative_pairs])
#     mixed_neg_data, _ = temporal_mixup(neg_data, alpha=alpha)
    
#     # Collect mixed pairs
#     mixed_pairs = []
#     mixed_labels = []
    
#     for i, (anchor, _) in enumerate(positive_pairs):
#         mixed_pairs.append([anchor, mixed_pos_data[i]])
#         mixed_labels.append(1)
    
#     for i, (anchor, _) in enumerate(negative_pairs):
#         mixed_pairs.append([anchor, mixed_neg_data[i]])
#         mixed_labels.append(0)

#     print('mixed pos data shape', mixed_pos_data.shape)
#     print('mixed nega data shape', mixed_neg_data.shape)
#     print('mixed_pairs length:', len(mixed_pairs))
#     # Combine half of the original pairs with half of the mixed-up pairs
#     num_pairs = len(pairs)
#     combined_pairs = pairs[:num_pairs // 2] + mixed_pairs[:num_pairs // 2]
#     combined_labels = labels[:num_pairs // 2] + mixed_labels[:num_pairs // 2]
    
#     combined_pairs = np.array(combined_pairs)
#     combined_labels = np.array(combined_labels)

#     print('combined pairs shape:', combined_pairs.shape)
#     return combined_pairs, combined_labels





def create_pairs_with_mixup_and_augmentations(data, augmented_reconstructions, hard_negatives, alpha=0.2, mixup_ratio=0.5):
    original_pairs = []
    original_labels = []

    for i in range(len(data)):
        # Augmented positive pair (anchor, positive)
        for aug_recon in augmented_reconstructions[i]:
            original_pairs.append([data[i], aug_recon])
            original_labels.append(1)
        
        # Hard negative pairs (anchor, negative)
        for hard_neg in hard_negatives[i]:
            original_pairs.append([data[i], hard_neg])
            original_labels.append(0)
    
    # Separate positive and negative pairs
    positive_pairs = [pair for pair, label in zip(original_pairs, original_labels) if label == 1]
    negative_pairs = [pair for pair, label in zip(original_pairs, original_labels) if label == 0]
    
    # Apply temporal mixup separately
    pos_data = np.array([pair[1] for pair in positive_pairs])
    mixed_pos_data, _ = temporal_mixup(pos_data, alpha=alpha)
    
    neg_data = np.array([pair[1] for pair in negative_pairs])
    mixed_neg_data, _ = temporal_mixup(neg_data, alpha=alpha)
    
    # Collect mixed pairs
    mixed_pairs = []
    mixed_labels = []
    
    for i, (anchor, _) in enumerate(positive_pairs):
        mixed_pairs.append([anchor, mixed_pos_data[i]])
        mixed_labels.append(1)
    
    for i, (anchor, _) in enumerate(negative_pairs):
        mixed_pairs.append([anchor, mixed_neg_data[i]])
        mixed_labels.append(0)
    
    # Determine number of pairs based on mixup_ratio
    num_original_pairs = int(len(original_pairs) * (1 - mixup_ratio) / 2)
    num_mixed_pairs = int(len(mixed_pairs) * mixup_ratio / 2)

    print('num_original_pairs', num_original_pairs, 'num_mixed_pairs', num_mixed_pairs)
    combined_pairs = (positive_pairs[:num_original_pairs] + mixed_pairs[:num_mixed_pairs] +
                      negative_pairs[:num_original_pairs] + mixed_pairs[len(positive_pairs):len(positive_pairs)+num_mixed_pairs])
    
    combined_labels = [1] * (num_original_pairs + num_mixed_pairs) + [0] * (num_original_pairs + num_mixed_pairs)
    
    combined_pairs = np.array(combined_pairs)
    combined_labels = np.array(combined_labels)
    print('combined pairs shape:', combined_pairs.shape)
    return combined_pairs, combined_labels



# def create_pairs_with_mixup_and_augmentations(data, augmented_reconstructions, hard_negatives, alpha=0.2):
#     pairs = []
#     labels = []

#     for i in range(len(data)):
#         # Augmented positive pair (anchor, positive)
#         for aug_recon in augmented_reconstructions[i]:
#             pairs.append([data[i], aug_recon])
#             labels.append(1)
        
#         # Hard negative pairs (anchor, negative)
#         for hard_neg in hard_negatives[i]:
#             pairs.append([data[i], hard_neg])
#             labels.append(0)
        
#     positive_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
#     negative_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]
    
#     # Apply temporal mixup separately
#     pos_data = np.array([pair[1] for pair in positive_pairs])
#     mixed_pos_data, _ = temporal_mixup(pos_data, alpha=alpha)
    
#     for i, (anchor, _) in enumerate(positive_pairs):
#         pairs.append([anchor, mixed_pos_data[i]])
#         labels.append(1)
    
#     neg_data = np.array([pair[1] for pair in negative_pairs])
#     mixed_neg_data, _ = temporal_mixup(neg_data, alpha=alpha)
    
#     for i, (anchor, _) in enumerate(negative_pairs):
#         pairs.append([anchor, mixed_neg_data[i]])
#         labels.append(0)
    
#     pairs = np.array(pairs)
#     labels = np.array(labels)
    
#     return pairs, labels


import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def build_base_network(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(32, activation='relu')(x)
    return models.Model(input, x)

def build_siamese_network(input_shape):
    base_network = build_base_network(input_shape)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Compute the absolute difference between the two vectors
    distance = layers.Lambda(lambda x: K.abs(x[0] - x[1]))([processed_a, processed_b])
    
    # Fully connected layer to compute similarity
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    
    model = models.Model([input_a, input_b], outputs)
    return model



def build_ae_siamese_network(params, Xtr, Xte, Ytr, Yte, dataset_name, inverter, model_name, mixup = True):
    

    if model_name == 'CNNAE':
        cnnae_model = CNNAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        cnnae_model.compile(optimizer='adam', loss='mse')
        cnnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        train_re = cnnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = cnnae_model.evaluate(Xte, Xte, verbose=0)

        train_enc = cnnae_model.predict(Xtr)
        test_enc = cnnae_model.predict(Xte)

    elif model_name == 'CNNATTNAE':                
        cnnattnae_model = ConvAttnAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        cnnattnae_model.compile(optimizer='adam', loss='mse')
        cnnattnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        train_re = cnnattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = cnnattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = cnnattnae_model.predict(Xtr)
        test_enc = cnnattnae_model.predict(Xte)
        
    elif model_name == 'LSTMAE':
        lstmae_model = LSTMAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        lstmae_model.compile(optimizer='adam', loss='mse')
        lstmae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = lstmae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = lstmae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = lstmae_model.predict(Xtr)
        test_enc = lstmae_model.predict(Xte)
        
    elif model_name == 'LSTMATTNAE':
        lstmattnae_model = LstmAttnAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])                
        lstmattnae_model.compile(optimizer='adam', loss='mse')
        lstmattnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = lstmattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = lstmattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = lstmattnae_model.predict(Xtr)
        test_enc = lstmattnae_model.predict(Xte)
        
    elif model_name == 'BiLSTMAE':
        bilstmae_model = BidirectionalLSTMAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        bilstmae_model.compile(optimizer='adam', loss='mse')
        bilstmae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = bilstmae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = bilstmae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = bilstmae_model.predict(Xtr)
        test_enc = bilstmae_model.predict(Xte)

    elif model_name == 'BiLSTMATTNAE':
        bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        bilstmattnae_model.compile(optimizer='adam', loss='mse')
        bilstmattnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = bilstmattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = bilstmattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = bilstmattnae_model.predict(Xtr)
        test_enc = bilstmattnae_model.predict(Xte)
        
    elif model_name == 'TCNAE':
        tcnae_model = TCNAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        tcnae_model.compile(optimizer='adam', loss='mse')
        tcnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = tcnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = tcnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = tcnae_model.predict(Xtr)
        test_enc = tcnae_model.predict(Xte)
            
    elif model_name == 'TCNATTNAE':
        tcnattnae_model = TCNAttnAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        tcnattnae_model.compile(optimizer='adam', loss='mse')
        tcnattnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = tcnattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = tcnattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = tcnattnae_model.predict(Xtr)
        test_enc = tcnattnae_model.predict(Xte)

    elif model_name == 'MaskedBiLSTMATTNAE':
        masked_Xtr = mask_sequences(Xtr, mask_fraction=0.3)
        # Count the number of zero entries in the original and masked sequences
        num_zeros_original = np.sum(Xtr == 0)
        num_zeros_masked = np.sum(Xtr == 0)
        print(f"Number of zero entries in original sequences: {num_zeros_original}")
        print(f"Number of zero entries in masked sequences: {num_zeros_masked}")

        bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        bilstmattnae_model.compile(optimizer='adam', loss='mse')
        # Here passing input masked values and other input stays the same original data and it tries to reconstruct 
        bilstmattnae_model.fit(masked_Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        # the validation is done on original val data itself, no need of masking here 
        
        train_re = bilstmattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = bilstmattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = bilstmattnae_model.predict(Xtr)
        test_enc = bilstmattnae_model.predict(Xte)
        

    elif model_name == 'MultiHeadLSTMATTNAE':
        num_heads=4 # considering 4 heads 
        multiheadlstmattnae_model = LstmMultiHeadAttnAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'], num_heads = num_heads )
        multiheadlstmattnae_model.compile(optimizer='adam', loss='mse')
        multiheadlstmattnae_model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

        train_re = multiheadlstmattnae_model.evaluate(Xtr, Xtr, verbose=0)
        test_re = multiheadlstmattnae_model.evaluate(Xte, Xte, verbose=0)
        
        train_enc = multiheadlstmattnae_model.predict(Xtr)
        test_enc = multiheadlstmattnae_model.predict(Xte)
        

    elif model_name == 'TransformerAE':
        transformerae_model = TransformerAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], encoding_dim=params['encoding_dim'], num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
        transformerae_model.compile(optimizer='adam', loss='mse')
        transformerae_model.fit([Xtr, Xtr], Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

        train_re = transformerae_model.evaluate([Xtr, Xtr], Xtr, verbose=0)
        test_re = transformerae_model.evaluate([Xte, Xte], Xte, verbose=0)
        
        train_enc = transformerae_model.predict([Xtr, Xtr])
        test_enc = transformerae_model.predict([Xte, Xte])

    
    # Apply SJP augmentations
    n_augmentations = 3  # Number of positive augmentations per data point
    positive_augmentations = augment_positive_sequences(train_enc, n_augmentations)  
     #Generate hard negatives with pairwise distances in the top 5
    #negative_augmentations = hard_negative_mining(Xtr, train_enc)

    #Generate hard negatives with reconstruction errors in the farthest top 5
    negative_augmentations = hard_negative_mining_with_error(Xtr, train_enc)
    print('Positive Augmentations shape after SJP:', positive_augmentations.shape )
    print('Negative Augmentations shape with reconstruction errors:', negative_augmentations.shape)
    

    # Create pairs
    pairs, labels = create_pairs_with_augmentations(Xtr, positive_augmentations, negative_augmentations)
    print('Pairs and labels shapes', pairs.shape, labels.shape)

    if mixup == True:
        # Create pairs with mixup on the pairs made with augmentations
        pairs, labels = create_pairs_with_mixup_and_augmentations(Xtr, positive_augmentations, negative_augmentations, alpha=0.2)
        print('After Mixup, Pairs and labels shapes', pairs.shape, labels.shape)

    ## training siamese network on the augmented data
    
    input_shape = Xtr.shape[1:]  # Shape of a single input sequence
    siamese_network = build_siamese_network(input_shape)
    siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #siamese_network.summary()
    # Unpack the pairs into two separate arrays for training
    anchor_data = pairs[:, 0]
    comparison_data = pairs[:, 1]
    
    # Ensure the data has the correct shape
    anchor_data = np.array(anchor_data)
    comparison_data = np.array(comparison_data)
    
    # Split the data into training and testing sets
    anchor_train, anchor_test, comparison_train, comparison_test, labels_train, labels_test = train_test_split(anchor_data, comparison_data, labels, test_size=0.2, random_state=42)

    # Train the Siamese network
    history = siamese_network.fit([anchor_train, comparison_train], labels_train, 
                              epochs=20, batch_size=64, validation_split=0.2)
    # Evaluate the model
    evaluation = siamese_network.evaluate([anchor_train, comparison_train], labels_train)
    print(f"Siamese results - Train Loss: {evaluation[0]}, Train Accuracy: {evaluation[1]}")

    # Evaluate the model
    evaluation = siamese_network.evaluate([anchor_test, comparison_test], labels_test)
    print(f"Siamese results - Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
    
    encoder = siamese_network.layers[2]

    # Extract latent embeddings for training and test data
    latent_train = encoder.predict(Xtr)
    latent_test = encoder.predict(Xte)
    
    return encoder, siamese_network, latent_train, latent_test
    




# Helper function to save metrics and arrays
def save_metrics_and_arrays(ytrue, ypred, yscore, dataset_name, model_name, inverter, model_suffix, is_train=True, weighted=False, augmentation_type='random'):
    print("ytrue unique values:", np.unique(ytrue))
    print("ypred unique values:", np.unique(ypred))
    print("ytrue data type:", ytrue.dtype)
    print("ypred data type:", ypred.dtype)
    
    metric_type = 'Weighted' if weighted else 'Regular'
    data_type = 'Train' if is_train else 'Test'
    print('augmentation_type inside save metrics', augmentation_type)
    
    if augmentation_type == 'warping':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_warping/'
    elif augmentation_type == 'jittering':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_jittering/'
    elif augmentation_type == 'freqshifting':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_freqshifting/'
        
    elif augmentation_type == 'timeshifting':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_timeshifting/'
    elif augmentation_type == 'scaling':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_scaling/'
    elif augmentation_type == 'permuting':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_permuting/'
    
    
    elif augmentation_type == 'masking':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_masking/'
    elif augmentation_type == 'slicing':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_positives_on_slicing/'
        
    elif augmentation_type == 'random':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_negatives_on_randomseq/'
        
    elif augmentation_type == 'pairwise':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_negatives_on_raw_distances/'
    
    elif augmentation_type == 'recon_error':
        
        metrics_path = '/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_with_negatives_on_re_distances/'
    else:
        raise ValueError("Invalid augmentation_type. Must be 'random', 'pairwise', or 'recon_error'.")

    
    metrics_df = getMetrics(ytrue=ytrue, ypred=ypred, yscore=yscore, dataset=dataset_name, inverter=inverter, model=f'CL_{model_name}{model_suffix}', save_arrays=False, weighted=weighted)

    print('metrics_df.shape, ', metrics_df.shape)
    print(metric_type, data_type, augmentation_type, 'metrics:', metrics_df)
    #model_name_full = f'CL_{model_name}{model_suffix}_{metric_type}'
    metrics_filename = f'{metrics_path}{dataset_name}_CL{metric_type}{data_type}Metrics_PosAugOnlyJitter_v2.csv'
    arrays_filename = f'{predictions_path}{dataset_name}_CL{metric_type}{data_type}Arrays.csv'
    
    metrics_df.to_csv(metrics_filename, index=False, mode='a', header=False)
    #arrays_df.to_csv(arrays_filename, index=False, mode='a', header=False)

def kmeans_clustering_v3(model_name, dataset_name, latent_train, latent_test, Yte, inverter, n_clusters=2, Ytr=None, mixup=False, augmentation_type='random'):
    print('augmentation_type inside kmeans', augmentation_type)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=88)
    kmeans.fit(latent_train)

    # Predict clusters for training and test data
    train_clusters = kmeans.predict(latent_train)
    test_clusters = kmeans.predict(latent_test)

    train_cluster_count = Counter(train_clusters)
    test_cluster_count = Counter(test_clusters)

    print(f"Train cluster counts: {train_cluster_count}")
    print(f"Test cluster counts: {test_cluster_count}")
    
    if len(np.unique(test_clusters)) > 1:
    
      silhouette_avg_test = silhouette_score(latent_test, test_clusters)
    else:
    
      silhouette_avg_test = -1  # or some other value to indicate this scenario
    
    # Evaluate clustering
    silhouette_avg_train = silhouette_score(latent_train, train_clusters)
    #silhouette_avg_test = silhouette_score(latent_test, test_clusters)
    print(f"Train Silhouette Score: {silhouette_avg_train}")
    print(f"Test Silhouette Score: {silhouette_avg_test}")

    # Identify anomalies (assuming the smaller cluster is the anomaly)
    anomaly_cluster = np.argmin(np.bincount(train_clusters))
    anomalies_train = (train_clusters == anomaly_cluster).astype(int)
    anomalies_test = (test_clusters == anomaly_cluster).astype(int)
    
    print(f"Number of anomalies detected train: {np.sum(anomalies_train)}")
    print(f"Number of anomalies detected test: {np.sum(anomalies_test)}")




    # Calculate distance to the nearest cluster center as anomaly score
    train_distances = pairwise_distances(latent_train, kmeans.cluster_centers_)
    test_distances = pairwise_distances(latent_test, kmeans.cluster_centers_)
    
    train_scores = np.min(train_distances, axis=1)
    test_scores = np.min(test_distances, axis=1)

    if mixup:
        model_suffix = '_WithMixup_Kmeans'
    else:
        model_suffix = '_WithoutMixup_Kmeans'

    # Calculate and save metrics for train set if Ytr is provided
    if Ytr is not None:
        # Save regular metrics for train set
        save_metrics_and_arrays(Ytr.values, anomalies_train, train_scores, dataset_name, model_name, inverter, model_suffix, is_train=True, weighted=False, augmentation_type=augmentation_type)
        # Save weighted metrics for train set
        save_metrics_and_arrays(Ytr.values, anomalies_train, train_scores, dataset_name, model_name, inverter, model_suffix, is_train=True, weighted=True, augmentation_type=augmentation_type)

    # Calculate and save metrics for test set
    # Save regular metrics for test set
    save_metrics_and_arrays(Yte.values, anomalies_test, test_scores, dataset_name, model_name, inverter, model_suffix, is_train=False, weighted=False, augmentation_type=augmentation_type)
    # Save weighted metrics for test set
    save_metrics_and_arrays(Yte.values, anomalies_test, test_scores, dataset_name, model_name, inverter, model_suffix, is_train=False, weighted=True, augmentation_type=augmentation_type)


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

def dbscan_clustering_v3(model_name, dataset_name, latent_train, latent_test, Yte, eps, min_samples, inverter, Ytr=None, mixup=False, augmentation_type='random'):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(latent_train)
    labels = dbscan.labels_
    
    # Identify the most common cluster (considered as normal) and treat others as anomalies
    labels_counter = Counter(labels[labels != -1])
    most_common_label = labels_counter.most_common(1)[0][0]
    labels = np.where(labels == most_common_label, 0, 1)
    
    clust_normal = np.mean(latent_train[labels == 0], axis=0)
    clust_anomaly = np.mean(latent_train[labels != 0], axis=0)
    
    # Calculate distances to the cluster centers for test data
    vals_clust1 = pairwise_distances(latent_test, [clust_normal], metric='cosine').flatten()
    vals_clust2 = pairwise_distances(latent_test, [clust_anomaly], metric='cosine').flatten()
    
    # Use the distances to the normal cluster center as anomaly scores
    yscore = np.minimum(vals_clust1, vals_clust2)
    ypred = np.where(vals_clust1 < vals_clust2, 0, 1)
    
    # Assuming you have ground truth labels for test data to calculate metrics
    anomalies_train = labels
    anomalies_test = ypred

    if mixup:
        model_suffix = '_WithMixup_DBScan'
    else:
        model_suffix = '_WithoutMixup_DBScan'
    

    # Calculate and save metrics for train set if Ytr is provided
    if Ytr is not None:
        # Calculate distances to the cluster centers for train data
        train_scores = np.minimum(
            pairwise_distances(latent_train, [clust_normal], metric='cosine').flatten(),
            pairwise_distances(latent_train, [clust_anomaly], metric='cosine').flatten()
        )

        # Save regular metrics for train set
        save_metrics_and_arrays(Ytr.values, anomalies_train, train_scores, dataset_name, model_name, inverter, model_suffix, is_train=True, weighted=False, augmentation_type=augmentation_type)
        # Save weighted metrics for train set
        save_metrics_and_arrays(Ytr.values, anomalies_train, train_scores, dataset_name, model_name, inverter, model_suffix, is_train=True, weighted=True, augmentation_type=augmentation_type)

    # Calculate and save metrics for test set
    # Save regular metrics for test set
    save_metrics_and_arrays(Yte.values, anomalies_test, yscore, dataset_name, model_name, inverter, model_suffix, is_train=False, weighted=False, augmentation_type=augmentation_type)
    # Save weighted metrics for test set
    save_metrics_and_arrays(Yte.values, anomalies_test, yscore, dataset_name, model_name, inverter, model_suffix, is_train=False, weighted=True, augmentation_type=augmentation_type)



import numpy as np
from sklearn.model_selection import train_test_split

def build_ae_siamese_network_v3(params, Xtr, Xte, Yte, dataset_name, inverter, model_name, Ytr=None, mixup=False, augmentation_type='random'):
    # Define the model based on the model_name
    model_mapping = {
        'CNNAE': CNNAEModel,
        'CNNATTNAE': ConvAttnAEModel,
        'LSTMAE': LSTMAEModel,
        'LSTMATTNAE': LstmAttnAEModel,
        'BiLSTMAE': BidirectionalLSTMAEModel,
        'BiLSTMATTNAE': BidirectionalLSTMATTNAEModel,
        'TCNAE': TCNAEModel,
        'TCNATTNAE': TCNAttnAEModel,
        'MaskedBiLSTMATTNAE': BidirectionalLSTMATTNAEModel,
        'MultiHeadLSTMATTNAE': LstmMultiHeadAttnAEModel,
        'TransformerAE': TransformerAEModel
    }

    ModelClass = model_mapping[model_name]
    if model_name == 'TransformerAE':
        model = TransformerAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], encoding_dim=params['encoding_dim'], num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
        model.compile(optimizer='adam', loss='mse')
        model.fit([Xtr, Xtr], Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = model.evaluate([Xtr, Xtr], Xtr, verbose=0)
        test_re = model.evaluate([Xte, Xte], Xte, verbose=0)

        train_enc = model.predict([Xtr, Xtr])
        test_enc = model.predict([Xte, Xte])
        
    else:
        
        model = ModelClass(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        model.compile(optimizer='adam', loss='mse')   
            
        if model_name == 'MaskedBiLSTMATTNAE':
            Xtr = mask_sequences(Xtr, mask_fraction=0.3)
    
        model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    
        train_re = model.evaluate(Xtr, Xtr, verbose=0)
        test_re = model.evaluate(Xte, Xte, verbose=0)
    
        train_enc = model.predict(Xtr)
        test_enc = model.predict(Xte)        
    

    # Apply SJP augmentations
    n_augmentations = 3  # Number of positive augmentations per data point
    positive_augmentations = augment_positive_sequences(train_enc, n_augmentations)
    positive_augmentations_new = augment_positive_sequences_new(train_enc, n_augmentations)

    print(positive_augmentations.shape, positive_augmentations_new.shape)

    if augmentation_type == 'random':
        negative_augmentations = hard_negative_mining_with_random(Xtr, num_negatives=3)
        
    elif augmentation_type == 'pairwise':
        negative_augmentations = hard_negative_mining(Xtr, train_enc)
        
    elif augmentation_type == 'recon_error':
        negative_augmentations = hard_negative_mining_with_error(Xtr, train_enc)
        
    else:
        raise ValueError("Invalid augmentation_type. Must be 'random', 'pairwise', or 'recon_error'.")

    print('Positive Augmentations shape after SJP:', positive_augmentations.shape)
    print('Negative Augmentations shape with selected augmentation type:', negative_augmentations.shape)

    # Create pairs
    pairs, labels = create_pairs_with_augmentations(Xtr, positive_augmentations, negative_augmentations)
    print('Pairs and labels shapes', pairs.shape, labels.shape)

    if mixup:
        # Create pairs with mixup on the pairs made with augmentations
        pairs, labels = create_pairs_with_mixup_and_augmentations(Xtr, positive_augmentations, negative_augmentations, alpha=0.2)
        print('After Mixup, Pairs and labels shapes', pairs.shape, labels.shape)

    # Training siamese network on the augmented data
    input_shape = Xtr.shape[1:]  # Shape of a single input sequence
    siamese_network = build_siamese_network(input_shape)
    siamese_network.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

    # Unpack the pairs into two separate arrays for training
    anchor_data = pairs[:, 0]
    comparison_data = pairs[:, 1]

    # Ensure the data has the correct shape
    anchor_data = np.array(anchor_data)
    comparison_data = np.array(comparison_data)

    # Split the data into training and testing sets
    anchor_train, anchor_test, comparison_train, comparison_test, labels_train, labels_test = train_test_split(anchor_data, comparison_data, labels, test_size=0.2, random_state=42)

    # Train the Siamese network
    history = siamese_network.fit([anchor_train, comparison_train], labels_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
    # Evaluate the model
    evaluation = siamese_network.evaluate([anchor_train, comparison_train], labels_train)
    print(f"Siamese results - Train Loss: {evaluation[0]}, Train Accuracy: {evaluation[1]}")

    evaluation = siamese_network.evaluate([anchor_test, comparison_test], labels_test)
    print(f"Siamese results - Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    encoder = siamese_network.layers[2]

    # Extract latent embeddings for training and test data
    latent_train = encoder.predict(Xtr)
    latent_test = encoder.predict(Xte)

    return encoder, siamese_network, latent_train, latent_test

   
   
# Define create_batch_pairs outside the main function
def create_batch_pairs(X_batch, model, augmentation_type='random'):
    # Encode the current batch
    encodings = model.predict(X_batch)

    # Generate augmentations for this batch
    n_augmentations = 3  # Number of positive augmentations per data point
    positive_augmentations = augment_positive_sequences(encodings, n_augmentations)

    if augmentation_type == 'random':
        negative_augmentations = hard_negative_mining_with_random(X_batch, num_negatives=3)
    elif augmentation_type == 'pairwise':
        negative_augmentations = hard_negative_mining(X_batch, encodings)
    elif augmentation_type == 'recon_error':
        negative_augmentations = hard_negative_mining_with_error(X_batch, encodings)
    else:
        raise ValueError("Invalid augmentation_type. Must be 'random', 'pairwise', or 'recon_error'.")

    pairs, labels = create_pairs_with_augmentations(X_batch, positive_augmentations, negative_augmentations)
    return pairs, labels     
    


import numpy as np

def train_siamese_network_with_batches(siamese_network, anchor_train, comparison_train, labels_train, anchor_test, comparison_test, labels_test, epochs=20, batch_size=64):
    num_samples = anchor_train.shape[0]
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle the data for each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        anchor_train = anchor_train[indices]
        comparison_train = comparison_train[indices]
        labels_train = labels_train[indices]
        
        # Train on batches
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_anchor = anchor_train[start:end]
            batch_comparison = comparison_train[start:end]
            batch_labels = labels_train[start:end]
            siamese_network.train_on_batch([batch_anchor, batch_comparison], batch_labels)
        
        # Evaluate after each epoch
        train_evaluation = siamese_network.evaluate([anchor_train, comparison_train], labels_train)
        print(f"Epoch {epoch+1} - Train Loss: {train_evaluation[0]}, Train Accuracy: {train_evaluation[1]}")

        test_evaluation = siamese_network.evaluate([anchor_test, comparison_test], labels_test)
        print(f"Epoch {epoch+1} - Test Loss: {test_evaluation[0]}, Test Accuracy: {test_evaluation[1]}")

    # After training, extract the encoder and latent embeddings
    encoder = siamese_network.layers[2]

    latent_train = encoder.predict(anchor_train)
    latent_test = encoder.predict(anchor_test)

    return encoder, siamese_network, latent_train, latent_test



def build_ae_siamese_network_v4(params, Xtr, Xte, Yte, dataset_name, inverter, model_name, Ytr=None, mixup=False, augmentation_type='random'):
    # Define the model based on the model_name
    model_mapping = {
        'CNNAE': CNNAEModel,
        'CNNATTNAE': ConvAttnAEModel,
        'LSTMAE': LSTMAEModel,
        'LSTMATTNAE': LstmAttnAEModel,
        'BiLSTMAE': BidirectionalLSTMAEModel,
        'BiLSTMATTNAE': BidirectionalLSTMATTNAEModel,
        'TCNAE': TCNAEModel,
        'TCNATTNAE': TCNAttnAEModel,
        'MaskedBiLSTMATTNAE': BidirectionalLSTMATTNAEModel,
        'MultiHeadLSTMATTNAE': LstmMultiHeadAttnAEModel,
        'TransformerAE': TransformerAEModel
    }

    ModelClass = model_mapping[model_name]
    if model_name == 'TransformerAE':
        model = TransformerAEModel(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], encoding_dim=params['encoding_dim'], num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
        model.compile(optimizer='adam', loss='mse')
        model.fit([Xtr, Xtr], Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        train_re = model.evaluate([Xtr, Xtr], Xtr, verbose=0)
        test_re = model.evaluate([Xte, Xte], Xte, verbose=0)

        train_enc = model.predict([Xtr, Xtr])
        test_enc = model.predict([Xte, Xte])
        
    else:
        model = ModelClass(timesteps=Xtr.shape[1], n_features=Xtr.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
        model.compile(optimizer='adam', loss='mse')   
            
        if model_name == 'MaskedBiLSTMATTNAE':
            Xtr = mask_sequences(Xtr, mask_fraction=0.3)
    
        model.fit(Xtr, Xtr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    
        train_re = model.evaluate(Xtr, Xtr, verbose=0)
        test_re = model.evaluate(Xte, Xte, verbose=0)
    
        train_enc = model.predict(Xtr)
        test_enc = model.predict(Xte)        

    input_shape = Xtr.shape[1:]  # Shape of a single input sequence
    siamese_network = build_siamese_network(input_shape)
    siamese_network.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

    # Prepare pairs for training
    anchor_data, comparison_data, labels = [], [], []
    
    num_batches = int(np.ceil(len(Xtr) / params['batch_size']))

    for batch_idx in range(num_batches):
        start = batch_idx * params['batch_size']
        end = min((batch_idx + 1) * params['batch_size'], len(Xtr))
        X_batch = Xtr[start:end]
        
        # Encode the current batch
        encodings = model.predict(X_batch)

        # Apply SJP augmentations
        n_augmentations = 3  # Number of positive augmentations per data point
        positive_augmentations = augment_positive_sequences(encodings, n_augmentations)

        if augmentation_type == 'random':
            negative_augmentations = hard_negative_mining_with_random(X_batch, num_negatives=3)
        elif augmentation_type == 'pairwise':
            negative_augmentations = hard_negative_mining(X_batch, encodings)
        elif augmentation_type == 'recon_error':
            negative_augmentations = hard_negative_mining_with_error(X_batch, encodings)
        else:
            raise ValueError("Invalid augmentation_type. Must be 'random', 'pairwise', or 'recon_error'.")

        # Create pairs for the current batch
        pairs, batch_labels = create_pairs_with_augmentations(X_batch, positive_augmentations, negative_augmentations)
        
        if mixup:
          # Create pairs with mixup on the pairs made with augmentations
          anchor_data, comparison_data, labels = create_pairs_with_mixup_and_augmentations(X_batch, positive_augmentations, negative_augmentations, alpha=0.2)


        # Unpack pairs
        anchor_data.extend(pairs[:, 0])
        comparison_data.extend(pairs[:, 1])
        labels.extend(batch_labels)

    # Convert lists to numpy arrays
    anchor_data = np.array(anchor_data)
    comparison_data = np.array(comparison_data)
    labels = np.array(labels)
    
    

    # Split the data into training and testing sets
    anchor_train, anchor_test, comparison_train, comparison_test, labels_train, labels_test = train_test_split(
        anchor_data, comparison_data, labels, test_size=0.2, random_state=42
    )

    # Train the Siamese network using the batch-wise training function
    encoder, siamese_network, latent_train, latent_test = train_siamese_network_with_batches(
        siamese_network,
        anchor_train, comparison_train, labels_train,
        anchor_test, comparison_test, labels_test,
        epochs=params['epochs'], batch_size=params['batch_size']
    )

    return encoder, siamese_network, latent_train, latent_test

       

def build_augdata_siamese_network(params, Xtr, Xte, Yte, dataset_name, inverter, Ytr=None, mixup=False, augmentation_type='random'):
    # Apply SJP augmentations
    n_augmentations = 3  # Number of positive augmentations per data point

    if augmentation_type == 'warping':
        positive_augmentations = time_warp(Xtr, n_augmentations=3)
        
    elif augmentation_type == 'timeshifting':
        positive_augmentations = time_shift(Xtr, n_augmentations=3)
        
    elif augmentation_type == 'masking':
        positive_augmentations = time_mask(Xtr, n_augmentations=3)

    elif augmentation_type == 'jittering':
        positive_augmentations = jitter(Xtr, n_augmentations=3)

    elif augmentation_type == 'scaling':
        positive_augmentations = scaling(Xtr, n_augmentations=3)

    elif augmentation_type == 'permuting':
        positive_augmentations = permute_sequence(Xtr, n_augmentations=3)

    elif augmentation_type == 'freqshifting':
        positive_augmentations = frequency_shift(Xtr, n_augmentations=3)

    elif augmentation_type == 'slicing':
        positive_augmentations = window_slicing(Xtr, n_augmentations=3)
        
    else:
        raise ValueError("Invalid augmentation_type. Must be 'random', 'pairwise', or 'recon_error'.")
    
    negative_augmentations = hard_negative_mining_with_random(Xtr, num_negatives=3)

    print('Positive Augmentations shape after SJP:', positive_augmentations.shape)
    print('Negative Augmentations shape with selected augmentation type:', negative_augmentations.shape)

    # Create pairs
    pairs, labels = create_pairs_with_augmentations(Xtr, positive_augmentations, negative_augmentations)
    print('Pairs and labels shapes', pairs.shape, labels.shape)

    if mixup:
        # Create pairs with mixup on the pairs made with augmentations
        pairs, labels = create_pairs_with_mixup_and_augmentations(Xtr, positive_augmentations, negative_augmentations, alpha=0.2)
        print('After Mixup, Pairs and labels shapes', pairs.shape, labels.shape)

    # Training siamese network on the augmented data
    input_shape = Xtr.shape[1:]  # Shape of a single input sequence
    siamese_network = build_siamese_network(input_shape)
    siamese_network.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

    # Unpack the pairs into two separate arrays for training
    anchor_data = pairs[:, 0]
    comparison_data = pairs[:, 1]

    # Ensure the data has the correct shape
    anchor_data = np.array(anchor_data)
    comparison_data = np.array(comparison_data)

    # Split the data into training and testing sets
    anchor_train, anchor_test, comparison_train, comparison_test, labels_train, labels_test = train_test_split(anchor_data, comparison_data, labels, test_size=0.2, random_state=42)

    # Train the Siamese network
    history = siamese_network.fit([anchor_train, comparison_train], labels_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
    # Evaluate the model
    evaluation = siamese_network.evaluate([anchor_train, comparison_train], labels_train)
    print(f"Siamese results - Train Loss: {evaluation[0]}, Train Accuracy: {evaluation[1]}")

    evaluation = siamese_network.evaluate([anchor_test, comparison_test], labels_test)
    print(f"Siamese results - Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    encoder = siamese_network.layers[2]

    # Extract latent embeddings for training and test data
    latent_train = encoder.predict(Xtr)
    latent_test = encoder.predict(Xte)

    print(latent_train.shape, latent_test.shape)

    return encoder, siamese_network, latent_train, latent_test






