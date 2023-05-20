import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout, Conv1D, MaxPooling1D, Flatten

import tensorflow as tf
from keras.optimizers import adam_v2


# split a univariate sequence into samples
def gen_time_series_data(X_train, y_train, X_val, y_val, n_input, n_future):
    batch_X = []
    batch_y = []
    val_X = []
    val_y = []
    test_X = []

    df_train = X_train.copy()
    df_train['pm'] = y_train
    df_val = X_val.copy()
    df_val['pm'] = y_val
    df_train = pd.concat([df_train, df_val])
    df_train = df_train.sort_values(by=['session_id', 'rank'])

    for id in df_train['session_id'].unique().tolist():
        tmp = df_train[df_train['session_id'] == id]
        tmp = tmp.drop('session_id', axis=1)
        max_rank = tmp['rank'].max()

        train_mask = tmp['rank'] > 24
        val_X_mask = (tmp['rank'] > 24) & (tmp['rank'] < 24 + n_input + 1)  # 25-64
        val_y_mask = (tmp['rank'] > 12) & (tmp['rank'] < 25)

        test_mask = tmp['rank'] < 12 + n_input + 1  # 13-52

        front_12_mask = tmp['rank'] < (max_rank - 11)  # 最后12个不要
        final_12_mask = tmp['rank'] > 36  # 预留25-36
        X_train = tmp[train_mask][final_12_mask]
        y_train = tmp[train_mask][front_12_mask]['pm']

        X_val = tmp[val_X_mask].values[np.newaxis, :, :]
        y_val = tmp[val_y_mask]['pm'].values[np.newaxis, :12]
        val_X.append(X_val)
        val_y.append(y_val)

        X_test = tmp[test_mask].values[np.newaxis, :, :]
        test_X.append(X_test)

        for end in range(len(X_train), n_input, -3):
            batch_X.append(X_train[end - 40:end].values[np.newaxis, :, :])
            batch_y.append(y_train[end - 40:end].values[np.newaxis, :12])

    batch_X = np.vstack(batch_X)  # batch_x的shape是(N, 40, fea)
    batch_y = np.vstack(batch_y)  # batch_y的shape是(N, 12)
    val_X = np.vstack(val_X)  # batch_x的shape是(295, 40, fea)
    val_y = np.vstack(val_y)
    test_X = np.vstack(test_X)
    return batch_X, batch_y, val_X, val_y, test_X


def train_val_lstm(X_train, y_train, X_val, y_val, in_n_features, out_dim, epoch, callbacks_list, verbose=1):
    # 13-24: val
    # 1-12: test
    # 用n_input个历史值预测未来n_future个值
    n_input = 40
    n_future = 12

    batch_X, batch_y, val_X, val_y, test_X = gen_time_series_data(X_train, y_train, X_val, y_val, n_input, n_future)
    # batch_x的shape是(N, 40, fea)
    # batch_y的shape是(N, 12)

    model_name = 'lstm'
    model = Sequential()
    if model_name == 'lstm':
        # First LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
        model.add(LSTM(units=16, activation='relu', return_sequences=True, input_shape=(n_input, in_n_features)))
        # model.add(Dropout(0.2))
        #
        # # Second LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
        # model.add(LSTM(units=16, activation='relu', return_sequences=True))
        # model.add(Dropout(0.2))
        #
        # # Final LSTM layer with Dropout regularisation; Set return_sequences to False since now we will be predicting with the output layer
        model.add(LSTM(units=32))
        # model.add(Dropout(0.2))

        # The output layer with linear activation to predict Open stock price
        # model.add(Dense(units=out_dim, activation="linear"))
        model.add(Dense(32, activation='softmax'))
        # model.add(Dense(32, activation='softmax'))
        model.add(Dense(n_future))
    elif model_name == 'CNN1D':
        model.add(
            Conv1D(filters=16, kernel_size=3, activation="tanh", use_bias=True, input_shape=(n_step, in_n_features)))
        model.add(MaxPooling1D(pool_size=n_step - 3 + 1))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(out_dim, activation='relu'))
    print(model.summary())

    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs)
    adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam)  # metrics = ['accuracy']
    history = model.fit(batch_X, batch_y, epochs=epoch, verbose=verbose,
                        validation_data=(val_X, val_y), callbacks=callbacks_list)  # verbose=1表示带进度条
    pred_y = model.predict(test_X)
    return pred_y


def predict_series(model, X_test, n_step, b_size):
    test_data_gen = gen_time_series_data(X_test, np.zeros(len(X_test)), n_step, b_size)
    # for i in zip(*test_data_gen[0]):
    #     print(*i)
    #     print('******************')
    return model.predict(test_data_gen, verbose=0)
