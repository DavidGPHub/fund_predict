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
from sklearn.metrics import mean_squared_error

################################################评分函数#######################################################
def my_loss(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return mae+tmape


def my_loss2(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return tmape

################################################数据清洗#######################################################
# 将train和val拼接起来
train_corr = pd.read_csv("./offical_data/train_correlation.csv")
train_corr = train_corr.rename(columns={train_corr.columns[0]: "ID"})
train_corr = train_corr.set_index('ID')
test_corr = pd.read_csv('./offical_data/test_correlation.csv')
test_corr = test_corr.rename(columns={test_corr.columns[0]: "ID"})
test_corr = test_corr.set_index('ID')

all_train_corr = pd.concat([train_corr, test_corr], axis=1)
del train_corr, test_corr
gc.collect()

# 最终的时间范围：'2017-06-01' --> '2018-03-16'
clear_train_corr = all_train_corr.loc[:, all_train_corr.columns >= '2017-06-01']
pred_index = clear_train_corr.index
pred_column = ['value']
################################################数据切分：非常关键#####################################################
train_size = 60
pred_size = 1
interval_time = 61


train_feature = pd.DataFrame()
train_target = pd.DataFrame()
for i in range(15):
    train = clear_train_corr.iloc[: , i : train_size + i].copy().astype('float32')
    train.columns = range(train.shape[1])
    train_feature = pd.concat([train_feature, train])
    test = clear_train_corr.iloc[:, train_size + interval_time + i: (train_size + interval_time + pred_size) + i].copy().astype('float32')
    test.columns = range(test.shape[1])
    train_target = pd.concat([train_target, test])

pred_X = clear_train_corr.iloc[: , -train_size:].copy().astype('float32')
pred_X.columns = range(pred_X.shape[1])

########################################### 特征工程 #######################################################
train_diff = train_feature - train_feature.shift(-1, axis=1)
pred_diff = pred_X - pred_X.shift(-1, axis=1)
train_diff5 = train_feature - train_feature.shift(-5, axis=1)
pred_diff5 = pred_X - pred_X.shift(-5, axis=1)
train_feature = train_feature.iloc[:, ::-1]
train_diff5m = train_feature - train_feature.rolling(window=5, axis=1).median()
train_feature = train_feature.iloc[:, ::-1]
train_diff5m = train_diff5m.iloc[:, ::-1]
pred_X = pred_X.iloc[:, ::-1]
pred_diff5m = pred_X - pred_X.rolling(window=5, axis=1).median()
pred_X = pred_X.iloc[:, ::-1]
pred_diff5m = pred_diff5m.iloc[:, ::-1]

def add_median(train, train_diff, train_diff5, train_diff5m,
               pred, pred_diff, pred_diff5, pred_diff5m,
               periods):
    train_median = train.median(axis=1).fillna(0).values
    pred_median = pred.median(axis=1).fillna(0).values
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
        ##### pred数据部分
        pred[c] = pred.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        pred[cm] = pred.iloc[:, 5 * w1:5 * w2].mean(axis=1, skipna=True).values
        pred[cmax] = pred.iloc[:, 5 * w1:5 * w2].max(axis=1, skipna=True).values
        pred[cd] = pred_diff.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        pred[cd5] = pred_diff5.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        pred[cd5m] = pred_diff5m.iloc[:, 5 * w1:5 * w2].median(axis=1, skipna=True).values
        pred[cd5mm] = pred_diff5m.iloc[:, 5 * w1:5 * w2].mean(axis=1, skipna=True).values
        pred[c] = (pred[c] - pred_median).fillna(0).astype('float32')
        pred[cm] = (pred[cm] - pred_median).fillna(0).astype('float32')
        pred[cmax] = (pred[cmax] - pred_median).fillna(0).astype('float32')
        pred[cd] = (pred[cd]).fillna(0).astype('float32')
        pred[cd5] = (pred[cd5]).fillna(0).astype('float32')
        pred[cd5m] = (pred[cd5m]).fillna(0).astype('float32')
        pred[cd5mm] = (pred[cd5mm]).fillna(0).astype('float32')
    return train, pred

# 一个周期加7个特征，总共12周
periods = [(11, 12), (10, 12), (9, 12), (8, 12),
           (10, 11), (9, 11), (8, 11), (7, 11),
           (9, 10), (8, 10), (7, 10), (6, 10),
           (8, 9), (7, 9), (6, 9), (5, 9),
           (7, 8), (6, 8), (5, 8),(4, 8),
           ]
eg_train, eg_pred = add_median(train_feature, train_diff, train_diff5, train_diff5m,
                               pred_X, pred_diff, pred_diff5, pred_diff5m,
                               periods)

eg_train = pd.concat([eg_train, train_diff, train_diff5, train_diff5m], axis=1)
eg_pred = pd.concat([eg_pred, pred_diff, pred_diff5, pred_diff5m], axis=1)
eg_train = eg_train.fillna(0)
eg_pred = eg_pred.fillna(0)
eg_train = eg_train.as_matrix()
eg_pred = eg_pred.as_matrix()
train_target = train_target.as_matrix()
train_feature = eg_train.reshape((eg_train.shape[0], 1, eg_train.shape[1]))
pred_feature = eg_pred.reshape((eg_pred.shape[0], 1, eg_pred.shape[1]))

##############################################attention rnn 模型代码################################################
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU
import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]


def attention_rnn_model(input_dim, output_dim):
    dropout = 0.7
    regularizer = 0.0005
    main_input = Input(shape=(1, input_dim),  dtype='float32', name='main_input')
    x = Bidirectional(LSTM(280, return_sequences=True))(main_input)
    x = Dropout(dropout)(x)

    sent_att_vec, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(x)
    x = Dropout(dropout)(sent_att_vec)

    x = Dense(output_dim, activation='linear',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    model = Model(inputs=main_input, outputs=x)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  # loss=my_loss2,
                  metrics=['mse'])
    return model

train_size = eg_train.shape[1]
model = attention_rnn_model(train_size, pred_size)
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10)
reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
model.fit(train_feature, train_target,
          batch_size=256,
          nb_epoch=20,
          validation_split=0.1,
          callbacks=[early_stopping, reduce]
          )
y_pred = model.predict(pred_feature)

### 生成结果
result = pd.DataFrame(y_pred, columns=['values'], index=pred_index)
result.to_csv('./submit_result/rnn_corr_v1.csv', index=True, index_label='ID', header=True)
