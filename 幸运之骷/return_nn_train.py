#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import gc
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import PCA
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input,Embedding, Dense, Activation, Dropout, Flatten
from keras import regularizers
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

################################################评分函数#######################################################
def my_loss(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return mae+tmape


def my_loss2(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return tmape

################################################数据导入#######################################################
clear_train_return = pd.read_csv("./train_data/all_fund_return.csv").set_index('ID')
################################################数据切分#####################################################
# 利用历史120天的数据，预测未来61天的数据
train_size = 120
pred_size = 61


final_train_feature = pd.DataFrame()
train_target = pd.DataFrame()
for i in range(419):
    train = clear_train_return.iloc[: , i : train_size + i].copy().astype('float32')
    train.columns = range(train.shape[1])
    final_train_feature = pd.concat([final_train_feature, train])
    test = clear_train_return.iloc[:, train_size + i : (train_size + pred_size) + i].copy().astype('float32')
    test.columns = range(test.shape[1])
    train_target = pd.concat([train_target, test])

########################################### 特征工程 #######################################################
train_diff = final_train_feature - final_train_feature.shift(-1, axis=1)
train_diff5 = final_train_feature - final_train_feature.shift(-5, axis=1)
final_train_feature = final_train_feature.iloc[:, ::-1]
train_diff5m = final_train_feature - final_train_feature.rolling(window=5, axis=1).median()
final_train_feature = final_train_feature.iloc[:, ::-1]
train_diff5m = train_diff5m.iloc[:, ::-1]

def add_median(train, train_diff, train_diff5, train_diff5m,periods):
    train_median = train.median(axis=1).fillna(0).values
    for (w1, w2) in periods:
        c = 'median_%d_%d' % (w1, w2)
        cm = 'mean_%d_%d' % (w1, w2)
        cmax = 'max_%d_%d' % (w1, w2)
        cd = 'median_diff_%d_%d' % (w1, w2)
        cd5 = 'median_diff5_%d_%d' % (w1, w2)
        cd5m = 'median_diff5m_%d_%d' % (w1, w2)
        cd5mm = 'mean_diff5m_%d_%d' % (w1, w2)
        ##### train数据部分
        train[c] = train.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        train[cm] = train.iloc[:, 5 * w1:5 * w2].mean(axis=1, skipna=True).values
        train[cmax] = train.iloc[:, 5 * w1:5 * w2].max(axis=1, skipna=True).values
        train[cd] = train_diff.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        train[cd5] = train_diff5.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        train[cd5m] = train_diff5m.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        train[cd5mm] = train_diff5m.iloc[:, 5 * w1:5 * w2].mean(axis=1, skipna=True).values
        train[c] = (train[c] - train_median).fillna(0).astype('float32')
        train[cm] = (train[cm] - train_median).fillna(0).astype('float32')
        train[cmax] = (train[cmax] - train_median).fillna(0).astype('float32')
        train[cd] = (train[cd]).fillna(0).astype('float32')
        train[cd5] = (train[cd5]).fillna(0).astype('float32')
        train[cd5m] = (train[cd5m]).fillna(0).astype('float32')
        train[cd5mm] = (train[cd5mm]).fillna(0).astype('float32')
        print(train.shape)
    return train

# 一个周期加7个特征，总共24周
periods = [(23, 24), (22, 24), (21, 24), (20, 24),
           (22, 23), (21, 23), (20, 23), (19, 23),
           (21, 22), (20, 22), (19, 22), (18, 22),
           (20, 21), (19, 21), (18, 21), (17, 21),
           (19, 20), (18, 20), (17, 20),(16, 20),
           ]
eg_train = add_median(final_train_feature, train_diff, train_diff5, train_diff5m,periods)


# train_feature = np.concatenate((train_feature, x_ext))
eg_train = pd.concat([eg_train, train_diff, train_diff5, train_diff5m], axis=1)
eg_train = eg_train.fillna(0)
eg_train = eg_train.as_matrix()
train_target = train_target.as_matrix()
train_feature = eg_train.reshape((eg_train.shape[0], 1, eg_train.shape[1]))

def rnn_model(input_dim, output_dim):

    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(1, input_dim),  dtype='float32', name='main_input')
    x = Bidirectional(LSTM(180, return_sequences=False))(main_input)
    x = Dropout(dropout)(x)

    x = Dense(output_dim, activation='linear',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    model = Model(inputs=main_input, outputs=x)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_size = eg_train.shape[1]
model = rnn_model(train_size, pred_size)
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10)
reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
model.fit(train_feature, train_target,
          batch_size=256,
          nb_epoch=100,
          validation_split=0.1,
          callbacks=[early_stopping, reduce]
          )

model.save("./model/self_rnn_return.h5")