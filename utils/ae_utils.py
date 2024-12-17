from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from keras.models import Model
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Input, LSTM, Conv1D, Conv1DTranspose, RepeatVector, TimeDistributed, Dense, Bidirectional,  AveragePooling1D, UpSampling1D, Activation, Attention, GlobalMaxPooling1D, Reshape, MultiHeadAttention, LayerNormalization, Add, Dropout
from plots import plot_loss_curve
from utils import getHyperParams
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from tcn import TCN
from utils import *


def CNNAEModel(timesteps=8, n_features=70, layer1=60, layer2=40, encoding_dim=20):
    input_layer = Input(shape=(timesteps, n_features))
    
    # Encoder
    x = Conv1D(filters=layer1, kernel_size=3, activation="relu", padding="same")(input_layer)
    x = Conv1D(filters=layer2, kernel_size=3, activation="relu", padding="same")(x)
    x = Conv1D(filters=encoding_dim, kernel_size=3, activation="relu", padding="same")(x)
    
    # Decoder
    x = Conv1DTranspose(filters=layer2, kernel_size=3, activation="relu", padding="same")(x)
    x = Conv1DTranspose(filters=layer1, kernel_size=3, activation="relu", padding="same")(x)
    output_layer = Conv1DTranspose(filters=n_features, kernel_size=3, activation="sigmoid", padding="same")(x)
    
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    
    return autoencoder



def ConvAttnAEModel(timesteps=8, n_features=70, layer1=60, layer2=40, encoding_dim=20):
    input = Input(shape=(timesteps, n_features))
    
    x = Conv1D(filters=layer1, activation="relu",kernel_size=3, padding="same")(input)
    x = Conv1D(filters=layer2,  activation="relu", kernel_size=3,padding="same")(x)
    x = Conv1D(filters=encoding_dim,  activation="relu",kernel_size=3, padding="same")(x)
    
    attention = Attention()([x, x])
    attention = GlobalMaxPooling1D()(attention)
    attention = Reshape((1, encoding_dim))(attention)
    
    x = Conv1DTranspose(filters=layer2, activation="relu", kernel_size=3,padding="same")(x)
    x = Conv1DTranspose(filters=layer1,  activation="relu",kernel_size=3, padding="same")(x)
    x = Conv1DTranspose(filters=n_features,  activation="sigmoid",kernel_size=3, padding="same")(x)
    
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="mse")
    
    return autoencoder

def LSTMAEModel(timesteps = 8, n_features=70,layer1=60, layer2=40,encoding_dim=20):
    lstm_autoencoder = Sequential()
    lstm_autoencoder.add(LSTM(layer1, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    lstm_autoencoder.add(LSTM(layer2, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(encoding_dim, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(timesteps))
    lstm_autoencoder.add(LSTM(encoding_dim, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(layer2, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(layer1, activation='sigmoid', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
    lstm_autoencoder.compile(loss='mse', optimizer="adam")  
    return lstm_autoencoder

# def LstmAttnAEModel(timesteps = 8, n_features=137, layer1=20, layer2=40, encoding_dim=20):
#     encoder_inputs = Input(shape=(timesteps, n_features))
#     encoder = LSTM(layer1, activation='relu', return_sequences=True)(encoder_inputs)
#     encoder = LSTM(layer2, activation='relu', return_sequences=True)(encoder)
#     encoder = LSTM(encoding_dim, activation='relu', return_sequences=True)(encoder)
#     encoder_outputs = RepeatVector(timesteps)(encoder)
#     attention = Attention()([encoder_outputs, encoder_outputs])
#     attention = GlobalMaxPooling1D()(attention)
#     attention = Reshape((1, encoding_dim))(attention)
#     decoder = LSTM(layer2, activation='relu', return_sequences=True)(attention)
#     decoder = LSTM(layer1, activation='relu', return_sequences=True)(decoder)
#     decoder_outputs = TimeDistributed(Dense(n_features,activation='sigmoid'))(decoder)
#     autoencoder = Model(encoder_inputs, decoder_outputs)
#     autoencoder.compile(optimizer='adam', loss='mse')
#     return autoencoder


def LstmAttnAEModel(timesteps=8, n_features=137, layer1=20, layer2=40, encoding_dim=20):
    # Encoder
    encoder_inputs = Input(shape=(timesteps, n_features))
    encoder = LSTM(layer1, activation='relu', return_sequences=True)(encoder_inputs)
    encoder = LSTM(layer2, activation='relu', return_sequences=True)(encoder)
    encoder = LSTM(encoding_dim, activation='relu', return_sequences=True)(encoder)
    
    # Attention mechanism
    attention = Attention()([encoder, encoder])
    
    # Decoder
    decoder = LSTM(layer2, activation='relu', return_sequences=True)(attention)
    decoder = LSTM(layer1, activation='relu', return_sequences=True)(decoder)
    decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)
    
    # Autoencoder model
    autoencoder = Model(encoder_inputs, decoder_outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder


# def LstmAttnAEModel(timesteps=8, n_features=137, layer1=20, layer2=40, encoding_dim=20):
#     encoder_inputs = Input(shape=(timesteps, n_features))
#     encoder = LSTM(layer1, activation='relu', return_sequences=True)(encoder_inputs)
#     encoder = LSTM(layer2, activation='relu', return_sequences=True)(encoder)
#     encoder = LSTM(encoding_dim, activation='relu', return_sequences=True)(encoder)  # Ensure return_sequences=True
#     attention = Attention()([encoder, encoder])
#     attention = GlobalMaxPooling1D()(attention)
#     decoder = LSTM(layer2, activation='relu', return_sequences=True)(attention)
#     decoder = LSTM(layer1, activation='relu', return_sequences=True)(decoder)
#     decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)
#     autoencoder = Model(encoder_inputs, decoder_outputs)
#     autoencoder.compile(optimizer='adam', loss='mse')
#     return autoencoder


def LstmMultiHeadAttnAEModel(timesteps=8, layer1=20, layer2=4, encoding_dim=4, n_features=137, num_heads=4):
    # Encoder
    encoder_inputs = Input(shape=(timesteps, n_features))
    encoder = LSTM(layer1, activation='relu', return_sequences=True)(encoder_inputs)
    encoder = LSTM(layer2, activation='relu', return_sequences=True)(encoder)
    encoder = LSTM(encoding_dim, activation='relu', return_sequences=False)(encoder)
    encoder_outputs = RepeatVector(timesteps)(encoder)
    
    # Multi-Head Attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=encoding_dim)(encoder_outputs, encoder_outputs)
    attention = Reshape((timesteps, encoding_dim))(attention)
    
    # Decoder
    decoder = LSTM(layer2, activation='relu', return_sequences=True)(attention)
    decoder = LSTM(layer1, activation='relu', return_sequences=True)(decoder)
    decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)
    
    # Autoencoder Model
    autoencoder = Model(encoder_inputs, decoder_outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder


def BidirectionalLSTMAEModel(timesteps=8, n_features=70, layer1=60, layer2=40, encoding_dim=20):
    # Input layer
    encoder_inputs = Input(shape=(timesteps, n_features))
    
    # Encoder
    x = Bidirectional(LSTM(layer1, activation='relu', return_sequences=True))(encoder_inputs)
    x = Bidirectional(LSTM(layer2, activation='relu', return_sequences=True))(x)
    x = Bidirectional(LSTM(encoding_dim, activation='relu', return_sequences=False))(x)
    
    # Bottleneck
    x = RepeatVector(timesteps)(x)
    
    # Decoder
    x = Bidirectional(LSTM(encoding_dim, activation='relu', return_sequences=True))(x)
    x = Bidirectional(LSTM(layer2, activation='relu', return_sequences=True))(x)
    x = Bidirectional(LSTM(layer1, activation='sigmoid', return_sequences=True))(x)
    
    # Output layer
    decoder_outputs = TimeDistributed(Dense(n_features))(x)
    # Model
    bilstm_autoencoder = Model(encoder_inputs, decoder_outputs)
    # Compile the model
    bilstm_autoencoder.compile(loss='mse', optimizer='adam')
    return bilstm_autoencoder


def LstmAttnAEModel(timesteps=8, n_features=137, layer1=20, layer2=40, encoding_dim=20):
    # Encoder
    encoder_inputs = Input(shape=(timesteps, n_features))
    encoder = LSTM(layer1, activation='relu', return_sequences=True)(encoder_inputs)
    encoder = LSTM(layer2, activation='relu', return_sequences=True)(encoder)
    encoder = LSTM(encoding_dim, activation='relu', return_sequences=True)(encoder)
    
    # Attention mechanism
    attention = Attention()([encoder, encoder])
    
    # Decoder
    decoder = LSTM(layer2, activation='relu', return_sequences=True)(attention)
    decoder = LSTM(layer1, activation='relu', return_sequences=True)(decoder)
    decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)
    
    # Autoencoder model
    autoencoder = Model(encoder_inputs, decoder_outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def BidirectionalLSTMATTNAEModel(timesteps=8, n_features=70, layer1=60, layer2=40, encoding_dim=20):
    # Input layer
    encoder_inputs = Input(shape=(timesteps, n_features))
    
    # Encoder
    encoder = Bidirectional(LSTM(layer1, activation='relu', return_sequences=True))(encoder_inputs)
    encoder = Bidirectional(LSTM(layer2, activation='relu', return_sequences=True))(encoder)
    encoder = Bidirectional(LSTM(encoding_dim, activation='relu', return_sequences=True))(encoder)
    
    # Bottleneck
    #encoder_outputs = RepeatVector(timesteps)(encoder_bi_lstm3)
    
    # Attention mechanism
    attention = Attention()([encoder, encoder])
    #attention = GlobalMaxPooling1D()(attention)
    #attention = Reshape((1, encoding_dim * 2))(attention) # Adjust reshape dimensions based on BiLSTM
    
    # Decoder
    #decoder = Bidirectional(LSTM(encoding_dim, activation='relu', return_sequences=True))(attention)
    decoder = Bidirectional(LSTM(layer2, activation='relu', return_sequences=True))(attention)
    decoder = Bidirectional(LSTM(layer1, activation='sigmoid', return_sequences=True))(decoder)
    
    # Output layer
    decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)    
    # Model
    bilstm_autoencoder = Model(encoder_inputs, decoder_outputs)    
    # Compile the model
    bilstm_autoencoder.compile(loss='mse', optimizer='adam')    
    return bilstm_autoencoder




def TCNAEModel(timesteps=8, n_features=70, layer1=20, layer2=6, encoding_dim=20):
    model = Sequential([
        TCN(input_shape=(timesteps, n_features), nb_filters=layer1, kernel_size=3, padding='same', activation='relu', return_sequences=True),
        Conv1D(filters=layer2, kernel_size=3, activation='relu', padding='same'),
        Conv1D(filters=encoding_dim, kernel_size=3, activation='relu', padding='same'),
        AveragePooling1D(pool_size=4, strides=None, padding='valid'),
        Activation("linear"),
        UpSampling1D(size=4),
        Conv1D(filters=layer2, kernel_size=3, activation='relu', padding='same'),
        TCN(nb_filters=layer1, kernel_size=3, padding='same', activation='relu', return_sequences=True),
        Dense(n_features, activation='sigmoid')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def TCNAttnAEModel(timesteps=8, n_features=137, layer1=100, layer2=64, encoding_dim = 4):
    encoder_inputs = Input(shape=(timesteps, n_features))
    encoder = TCN(nb_filters=layer1, kernel_size=3, padding='same',activation='relu', return_sequences=True)(encoder_inputs)
    encoder = Conv1D(filters=layer2, kernel_size=3, activation='relu', padding='same')(encoder)
    encoder = Conv1D(filters=encoding_dim, kernel_size=3, activation='relu', padding='same')(encoder)
    attention = Attention()([encoder, encoder])
    attention = AveragePooling1D(pool_size=4, strides=None, padding='valid')(attention)
    decoder = UpSampling1D(size=4)(attention)
    decoder = Conv1D(filters=layer2, kernel_size=3, activation='relu', padding='same')(decoder)
    decoder = TCN(nb_filters=layer1, kernel_size=3, padding='same',activation='relu',return_sequences=True)(decoder)
    decoder_outputs = Dense(n_features, activation='sigmoid')(decoder)
    autoencoder = Model(encoder_inputs, decoder_outputs)
    autoencoder.compile(loss='mse', optimizer='adam')    
    return autoencoder

def encoder(input_dim, encoding_dim, layer1, layer2):
    x = Dense(layer1, activation = 'relu')(input_dim)
    x = Dense(layer2, activation = 'relu')(x)
    x = Dense(encoding_dim, activation = 'relu')(x)
    return x

def decoder(encoding_dim, decod_dim, layer1, layer2):
    x = Dense(layer2, activation = 'relu')(encoding_dim)
    x = Dense(layer1, activation = 'relu')(x)
    x = Dense(decod_dim, activation = 'sigmoid')(x)
    return x

def MLPAEModel(n_features, encoding_dim, layer1, layer2):
    input_dim = Input(shape = (n_features, ))
    encoder_out = encoder(input_dim, encoding_dim, layer1, layer2)
    decoder_out = decoder(encoder_out, n_features, layer1, layer2)
    autoencoder = Model(inputs = input_dim, outputs = decoder_out)
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    return autoencoder

## Transformer Autoencoder Architecture

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    # Feed Forward Network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    return Add()([x, res])

def transformer_decoder(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Masked Multi-Head Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    # Normalization and Multi-Head Attention
    x = LayerNormalization(epsilon=1e-6)(res)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, encoder_outputs)
    x = Dropout(dropout)(x)
    res = Add()([x, res])
    
    # Feed Forward Network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    return Add()([x, res])

def TransformerAEModel(timesteps=8, n_features=137, encoding_dim=64, num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1):
    # Adjust encoding_dim to be divisible by num_heads
    if encoding_dim % num_heads != 0:
        encoding_dim = num_heads * (encoding_dim // num_heads)
        
    head_size = encoding_dim // num_heads  # Calculate the head size
    # head_size: This refers to the size of each individual attention head in the multi-head attention mechanism. Each head projects the input into a different subspace, which allows the model to capture different aspects of the input data.

#encoding_dim: This refers to the overall dimension of the encoded representation in the model. This dimension is used in various parts of the transformer architecture, including the size of the vectors used in the multi-head attention mechanism.

#In many implementations, encoding_dim is used to determine the head_size by dividing it by the number of heads (num_heads). For example, if encoding_dim is 128 and num_heads is 8, then each head would have a head_size of 128 / 8 = 16. 

    encoder_inputs = Input(shape=(timesteps, n_features))
    x = encoder_inputs
    for _ in range(num_encoder_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    encoder_outputs = x
    
    decoder_inputs = Input(shape=(timesteps, n_features))
    x = decoder_inputs
    for _ in range(num_decoder_layers):
        x = transformer_decoder(x, encoder_outputs, head_size, num_heads, ff_dim, dropout)
    decoder_outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(x)
    
    # Model
    autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder



param_grid ={
            'n_features': [137],
            'encoding_dim':[30,20,10],
            'layer1':[50, 40,30],
            'layer2':[40,30,20],
            'epochs': [100, 200],
            'batch_size': [256, 128, 64]
            }

callback1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1) 



def tuneHyperparameters(xtrain, param_grid, dataset_name, inverter, model_name, params_path):
    ''' Tune Hypereparameters for the autoencoder model and save to the parameters to a csv file'''
    
    if model_name == 'MLPAE':
        regressor_keras = KerasRegressor(build_fn = MLPAEModel, n_features=xtrain.shape[1],  encoding_dim = 30, layer1 = 50, layer2 = 30, verbose= 0)
    elif model_name == 'CNNAE':
        regressor_keras = KerasRegressor(build_fn=CNNAEModel, n_features=xtrain.shape[2], encoding_dim = 30, layer1 = 50, layer2 = 30, verbose=0)
    elif model_name == 'LSTMAE':
        regressor_keras = KerasRegressor(build_fn = LSTMAEModel, n_features=xtrain.shape[2], encoding_dim = 30, layer1 = 50, layer2 = 30, verbose=0)
    elif model_name == 'TCNAE':
        regressor_keras = KerasRegressor(build_fn = TCNAEModel, n_features=xtrain.shape[2], encoding_dim = 30, layer1 = 50, layer2 = 30, verbose=0)
        
    # grid_search = GridSearchCV(estimator=regressor_keras, param_grid=param_grid, scoring='neg_mean_squared_error' if model_name=='MLPAE' else scorer, cv=3, verbose=False)
    
    grid_search = GridSearchCV(estimator=regressor_keras, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=False)
    print(f"KerasRegressor params: {regressor_keras.get_params().keys()}")
    #xtrain_flat = xtrain.reshape((xtrain.shape[0], -1))
    print(xtrain.shape)

    grid_search.fit(xtrain, xtrain, callbacks=[callback1])
    
    df = pd.DataFrame({'file':[dataset_name],'inverter':[inverter],'model':[model_name], 'Hyperparameters':[grid_search.best_params_]})
    
    print(grid_search.best_params_)
    df.to_csv(f'{params_path}{dataset_name}_{model_name}.csv', index=False, header=False, mode='a')

def trainAEModel(xtrain, dataset_name, model_name, inverter, hyp_path,model_save_path,save_model=False):
    ''' load the saved hyperparameters and train the model, save the model to a path
    '''
    df_hyp = pd.read_csv(f'{hyp_path}{dataset_name}_{model_name}.csv', names=['file', 'model', 'Hyperparameters'])
    batch_size, encoding_dim, epochs, layer1, layer2 = getHyperParams(dataset_name,df_hyp,inverter, model_name) 
    if model_name == 'MLPAE':
        model = MLPAEModel(layer1=layer1, layer2=layer2, encoding_dim=encoding_dim, n_features=xtrain.shape[1])
    elif model_name == 'CNNAE':
        model = CNNAEModel(layer1=layer1, layer2=layer2, encoding_dim=encoding_dim, n_features=xtrain.shape[2])
    elif model_name == 'LSTMAE':
        model = LSTMAEModel(layer1=layer1, layer2=layer2, encoding_dim=encoding_dim, n_features=xtrain.shape[2])
    elif model_name == 'BiLSTMAE':
        model = BidirectionalLSTMAEModel(layer1=layer1, layer2=layer2, encoding_dim=encoding_dim, n_features=xtrain.shape[2])
    if save_model:
        history = model.fit(xtrain, xtrain, epochs=epochs, batch_size=batch_size, verbose=False,validation_split=0.2, callbacks=callback1).history
        model.save(f'{model_save_path}{dataset_name}_{model_name}.h5')
        plot_loss_curve(history, path=f'./Plots/AE/{model_name}_{dataset_name}_AccLoss')
    else:
        model.load_weights(f'{model_save_path}{dataset_name}_{model_name}.h5')
    return model



def custom_hyperparameter_search(param_grid, X_train, dataset_name, inverter, model_name, params_path, num_folds=3):
    best_params = None
    best_score = float('inf')
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    
    for params in ParameterGrid(param_grid):
        fold_scores = []
        print(params)
        # Cross-validation loop
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            print(X_train_fold.shape, X_val_fold.shape)

            if model_name == 'CNNAE':
                cnnae_model = CNNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                cnnae_model.compile(optimizer='adam', loss='mse')
                cnnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = cnnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)

            elif model_name == 'CNNATTNAE':                
                cnnattnae_model = ConvAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                cnnattnae_model.compile(optimizer='adam', loss='mse')
                cnnattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = cnnattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)
            
            elif model_name == 'LSTMAE':
                lstmae_model = LSTMAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                lstmae_model.compile(optimizer='adam', loss='mse')
                lstmae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = lstmae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)

            elif model_name == 'LSTMATTNAE':
                lstmattnae_model = LstmAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                lstmattnae_model.compile(optimizer='adam', loss='mse')
                lstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = lstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)
                
            elif model_name == 'BiLSTMAE':
                bilstmae_model = BidirectionalLSTMAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmae_model.compile(optimizer='adam', loss='mse')
                bilstmae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = bilstmae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)

            elif model_name == 'BiLSTMATTNAE':
                bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmattnae_model.compile(optimizer='adam', loss='mse')
                bilstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = bilstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)
                
            elif model_name == 'TCNAE':
                tcnae_model = TCNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                tcnae_model.compile(optimizer='adam', loss='mse')
                tcnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = tcnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)  

            elif model_name == 'TCNATTNAE':
                tcnattnae_model = TCNAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                tcnattnae_model.compile(optimizer='adam', loss='mse')
                tcnattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = tcnattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)  

            elif model_name == 'MaskedBiLSTMATTNAE':
                masked_X_train_fold = mask_sequences(X_train_fold, mask_fraction=0.3)
                # Count the number of zero entries in the original and masked sequences
                num_zeros_original = np.sum(X_train_fold == 0)
                num_zeros_masked = np.sum(masked_X_train_fold == 0)
                print(f"Number of zero entries in original sequences: {num_zeros_original}")
                print(f"Number of zero entries in masked sequences: {num_zeros_masked}")

                bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmattnae_model.compile(optimizer='adam', loss='mse')
                # Here passing input masked values and other input stays the same original data and it tries to reconstruct 
                bilstmattnae_model.fit(masked_X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                # the validation is done on original val data itself, no need of masking here 
                val_loss = bilstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)  

            elif model_name == 'MultiHeadLSTMATTNAE':
                num_heads=4 # considering 4 heads 
                multiheadlstmattnae_model = LstmMultiHeadAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'], num_heads = num_heads )
                multiheadlstmattnae_model.compile(optimizer='adam', loss='mse')
                multiheadlstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = multiheadlstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                fold_scores.append(val_loss)
           
        
        avg_score = np.mean(fold_scores)
        print('avg_score', avg_score)
        
        if avg_score < best_score:
            best_params = params
            best_score = avg_score
    
    df = pd.DataFrame({'file': [dataset_name], 'inverter': [inverter], 'model': [model_name], 'Hyperparameters': [best_params]})
    print(df.values)
    df.to_csv(f'{params_path}{dataset_name}_{model_name}.csv', index=False, header=False, mode='a')
    
    return best_params


transformer_param_grid ={
            'n_features': [137],
            'encoding_dim':[30,20,10],
            #'layer1':[50, 40,30],
            #'layer2':[40,30,20],
            'epochs': [100, 200],
            'batch_size': [256, 128, 64]
            }

def transformer_custom_hyperparameter_search(transformer_param_grid, X_train, dataset_name, inverter, model_name, params_path, num_folds=3):
    best_params = None
    best_score = float('inf')
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    
    for params in ParameterGrid(transformer_param_grid):
        fold_scores = []
        print(params)
        # Cross-validation loop
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            print(X_train_fold.shape, X_val_fold.shape)


            if model_name == 'TransformerAE':
                transformerae_model = TransformerAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], encoding_dim=params['encoding_dim'], num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
                transformerae_model.compile(optimizer='adam', loss='mse')
                transformerae_model.fit([X_train_fold, X_train_fold], X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                val_loss = transformerae_model.evaluate([X_val_fold, X_val_fold], X_val_fold, verbose=0)
                fold_scores.append(val_loss)
        
        avg_score = np.mean(fold_scores)
        print('avg_score', avg_score)
        
        if avg_score < best_score:
            best_params = params
            best_score = avg_score
    
    df = pd.DataFrame({'file': [dataset_name], 'inverter': [inverter], 'model': [model_name], 'Hyperparameters': [best_params]})
    print(df.values)
    df.to_csv(f'{params_path}{dataset_name}_{model_name}.csv', index=False, header=False, mode='a')
    
    return best_params



def compare_ae_models(custom_param_grid, X_train, dataset_name, inverter, model_name, params_path, num_folds=3):
    best_params = None
    best_score = float('inf')
    
    #kf = KFold(n_splits=num_folds, shuffle=True)
    tscv = TimeSeriesSplit(n_splits=num_folds)
    
    for params in ParameterGrid(custom_param_grid):
        train_scores = []
        fold_scores = []
        print(params)
        # Cross-validation loop
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            print(X_train_fold.shape, X_val_fold.shape)


            if model_name == 'CNNAE':
                cnnae_model = CNNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                cnnae_model.compile(optimizer='adam', loss='mse')
                cnnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = cnnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = cnnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)

            elif model_name == 'CNNATTNAE':                
                cnnattnae_model = ConvAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                cnnattnae_model.compile(optimizer='adam', loss='mse')
                cnnattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = cnnattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = cnnattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)
            
            elif model_name == 'LSTMAE':
                lstmae_model = LSTMAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                lstmae_model.compile(optimizer='adam', loss='mse')
                lstmae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = lstmae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = lstmae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)

            elif model_name == 'LSTMATTNAE':
                lstmattnae_model = LstmAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])                
                lstmattnae_model.compile(optimizer='adam', loss='mse')
                lstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = lstmattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = lstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)
                
            elif model_name == 'BiLSTMAE':
                bilstmae_model = BidirectionalLSTMAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmae_model.compile(optimizer='adam', loss='mse')
                bilstmae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = bilstmae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = bilstmae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)

            elif model_name == 'BiLSTMATTNAE':
                bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmattnae_model.compile(optimizer='adam', loss='mse')
                bilstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = bilstmattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = bilstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)
                
            elif model_name == 'TCNAE':
                tcnae_model = TCNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                tcnae_model.compile(optimizer='adam', loss='mse')
                tcnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = tcnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = tcnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)
                
            elif model_name == 'TCNATTNAE':
                tcnattnae_model = TCNAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                tcnattnae_model.compile(optimizer='adam', loss='mse')
                tcnattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = tcnattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = tcnattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re) 

            elif model_name == 'MaskedBiLSTMATTNAE':
                masked_X_train_fold = mask_sequences(X_train_fold, mask_fraction=0.3)
                # Count the number of zero entries in the original and masked sequences
                num_zeros_original = np.sum(X_train_fold == 0)
                num_zeros_masked = np.sum(masked_X_train_fold == 0)
                print(f"Number of zero entries in original sequences: {num_zeros_original}")
                print(f"Number of zero entries in masked sequences: {num_zeros_masked}")

                bilstmattnae_model = BidirectionalLSTMATTNAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'])
                bilstmattnae_model.compile(optimizer='adam', loss='mse')
                # Here passing input masked values and other input stays the same original data and it tries to reconstruct 
                bilstmattnae_model.fit(masked_X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                # the validation is done on original val data itself, no need of masking here 
                train_re = bilstmattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = bilstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)  

            elif model_name == 'MultiHeadLSTMATTNAE':
                num_heads=4 # considering 4 heads 
                multiheadlstmattnae_model = LstmMultiHeadAttnAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], layer1=params['layer1'], layer2=params['layer2'], encoding_dim=params['encoding_dim'], num_heads = num_heads )
                multiheadlstmattnae_model.compile(optimizer='adam', loss='mse')
                multiheadlstmattnae_model.fit(X_train_fold, X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = multiheadlstmattnae_model.evaluate(X_train_fold, X_train_fold, verbose=0)
                val_re = multiheadlstmattnae_model.evaluate(X_val_fold, X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)  

            elif model_name == 'TransformerAE':
                transformerae_model = TransformerAEModel(timesteps=X_train_fold.shape[1], n_features=X_train_fold.shape[2], encoding_dim=params['encoding_dim'], num_heads=4, ff_dim=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
                transformerae_model.compile(optimizer='adam', loss='mse')
                transformerae_model.fit([X_train_fold, X_train_fold], X_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                train_re = transformerae_model.evaluate([X_train_fold, X_train_fold], X_train_fold, verbose=0)
                val_re = transformerae_model.evaluate([X_val_fold, X_val_fold], X_val_fold, verbose=0)
                train_scores.append(train_re)
                fold_scores.append(val_re)
                
           
        avg_train_score = np.mean(train_scores)
        print('avg_train_re', avg_train_score)
        
        avg_val_score = np.mean(fold_scores)
        print('avg_val_re', avg_val_score)

    
    df = pd.DataFrame({'file': [dataset_name], 'inverter': [inverter], 'model': [model_name], 'avg_train_re': [avg_train_score],'avg_val_re': [avg_val_score]})
    print(df.values)
    df.to_csv(f'{params_path}{dataset_name}_AEs_comparison.csv', index=False, header=False, mode='a')
    
    return avg_train_score, avg_val_score






def getmetrics_ae_models(params, Xtr, Xte, Yte, dataset_name, inverter, model_name, Ytr=None):
    

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

                
    print('calculating metrics')       
    threshold = np.percentile(np.mean(np.power(flatten(Xtr)- flatten(train_enc), 2), axis=1), 99)
    train_mses = np.mean(np.power(flatten(Xtr)- flatten(train_enc), 2), axis=1)
    test_mses = np.mean(np.power(flatten(Xte)- flatten(test_enc), 2), axis=1)
    ypred_train = np.where(train_mses>threshold,1,0)
    ypred_test = np.where(test_mses>threshold,1,0)
    
    


    if Ytr is not None:
        #train_metrics_wt = getWeightedMetrics(ytrue=Ytr.values, ypred=ypred_train, yscore=train_mses, dataset=dataset_name, inverter = inverter, model=model_name)
        train_metrics_wt = getMetrics(ytrue=Ytr.values, ypred=ypred_train, yscore=train_mses, dataset=dataset_name, inverter=inverter, model=model_name, save_arrays=False, weighted=True)
        train_metrics_wt.to_csv(f'/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_baseline_ae_models/{dataset_name}WeightedTrainMetrics_v2.csv', index=False, mode='a', header=False)

        #train_metrics = getMetrics(ytrue=Ytr.values, ypred=ypred_train,  yscore=train_mses, dataset=dataset_name, inverter=inverter, model=model_name)
        train_metrics = getMetrics(ytrue=Ytr.values, ypred=ypred_train, yscore=train_mses, dataset=dataset_name, inverter=inverter, model=model_name, save_arrays=False, weighted=False)
        train_metrics.to_csv(f'/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_baseline_ae_models/{dataset_name}TrainMetrics_v2.csv', index=False, mode='a', header=False)

    #test_metrics_wt = getWeightedMetrics(ytrue=Yte.values, ypred=ypred_test,  yscore=test_mses, dataset=dataset_name, inverter=inverter,model=model_name)
    test_metrics_wt = getMetrics(ytrue=Yte.values, ypred=ypred_test, yscore=test_mses, dataset=dataset_name, inverter=inverter, model=model_name, save_arrays=False, weighted=True)
    test_metrics_wt.to_csv(f'/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_baseline_ae_models/{dataset_name}WeightedTestMetrics_v2.csv', index=False, mode='a', header=False)

    #test_metrics = getMetrics(ytrue=Yte.values, ypred=ypred_test,  yscore=test_mses, dataset=dataset_name, inverter=inverter, model=model_name)
    test_metrics = getMetrics(ytrue=Yte.values, ypred=ypred_test, yscore=test_mses, dataset=dataset_name, inverter=inverter, model=model_name, save_arrays=False, weighted=False)
    test_metrics.to_csv(f'/mnt/work/digitwin/vpulagura_work/CheckingCode_CN/ThesisCode/Metrics/Metrics_baseline_ae_models/{dataset_name}TestMetrics_v2.csv', index=False, mode='a', header=False)

        



