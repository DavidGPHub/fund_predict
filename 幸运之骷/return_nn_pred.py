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
from keras.layers import Layer
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

################################################数据导入#######################################################
all_train_fund_return = pd.read_csv("./train_data/all_fund_return.csv").set_index('ID')
################################################构建预测结果空值数据: 最终的预测结果需要填充这里的缺失值#######################################################
# 需要预测天的利率大小，总共61天
pred_dates = ['2018-03-19', '2018-03-20', '2018-03-21', '2018-03-22', '2018-03-23',
              '2018-03-26', '2018-03-25', '2018-03-28', '2018-03-29', '2018-03-30',
              '2018-04-02', '2018-04-03', '2018-04-04', '2018-04-09', '2018-04-10',
              '2018-04-11', '2018-04-12', '2018-04-13', '2018-04-16', '2018-04-15',
              '2018-04-18', '2018-04-19', '2018-04-20', '2018-04-23', '2018-04-24',
              '2018-04-25', '2018-04-26', '2018-04-25', '2018-05-02', '2018-05-03',
              '2018-05-04', '2018-05-05', '2018-05-08', '2018-05-09', '2018-05-10',
              '2018-05-11', '2018-05-14', '2018-05-15', '2018-05-16', '2018-05-15',
              '2018-05-18', '2018-05-21', '2018-05-22', '2018-05-23', '2018-05-24',
              '2018-05-25', '2018-05-28', '2018-05-29', '2018-05-30', '2018-05-31',
              '2018-06-01', '2018-06-04', '2018-06-05', '2018-06-06', '2018-06-05',
              '2018-06-08', '2018-06-11', '2018-06-12', '2018-06-13', '2018-06-14',
              '2018-06-15']
fund_names = list(all_train_fund_return.index.values)
pred_y = pd.DataFrame(columns=pred_dates, index=fund_names)

################################################数据切分：非常关键#####################################################
train_size = 120
pred_size = 61
pred_X = all_train_fund_return.iloc[: , -train_size : ].copy().astype('float32')
pred_X.columns = range(pred_X.shape[1])

###########################################  特征工程 #######################################################
pred_diff = pred_X - pred_X.shift(-1, axis=1)
pred_diff5 = pred_X - pred_X.shift(-5, axis=1)
pred_X = pred_X.iloc[:, ::-1]
pred_diff5m = pred_X - pred_X.rolling(window=5, axis=1).median()
pred_X = pred_X.iloc[:, ::-1]
pred_diff5m = pred_diff5m.iloc[:, ::-1]

def add_median(pred, pred_diff, pred_diff5, pred_diff5m,
               periods):
    pred_median = pred.median(axis=1).fillna(0).values
    for (w1, w2) in periods:
        c = 'median_%d_%d' % (w1, w2)
        cm = 'mean_%d_%d' % (w1, w2)
        cmax = 'max_%d_%d' % (w1, w2)
        cd = 'median_diff_%d_%d' % (w1, w2)
        cd5 = 'median_diff5_%d_%d' % (w1, w2)
        cd5m = 'median_diff5m_%d_%d' % (w1, w2)
        cd5mm = 'mean_diff5m_%d_%d' % (w1, w2)
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
        print(pred.shape)
    return pred

# 一个周期加7个特征，总共24周
periods = [(23, 24), (22, 24), (21, 24), (20, 24),
           (22, 23), (21, 23), (20, 23), (19, 23),
           (21, 22), (20, 22), (19, 22), (18, 22),
           (20, 21), (19, 21), (18, 21), (17, 21),
           (19, 20), (18, 20), (17, 20),(16, 20),
           ]
eg_pred = add_median(pred_X, pred_diff, pred_diff5, pred_diff5m,periods)


eg_pred = pd.concat([eg_pred, pred_diff, pred_diff5, pred_diff5m], axis=1)
eg_pred = eg_pred.fillna(0)
eg_pred = eg_pred.as_matrix()
print(eg_pred.shape)

pred_feature = eg_pred.reshape((eg_pred.shape[0], 1, eg_pred.shape[1]))
print(pred_feature.shape)

# #### 模型读取
# model = load_model("./model/gpu_rnn_return.h5")  ###GPU版本
model = load_model("./model/cpu_rnn_return.h5")  ###CPU版本


## 结果预测
y_pred = model.predict(pred_feature)
print(y_pred)
print(y_pred.shape)

result = pd.DataFrame(y_pred, columns=pred_dates, index=fund_names)
print(result)
result.to_csv('./train_data/return_rnn_v1.csv', index=True, index_label='ID', header=True)
