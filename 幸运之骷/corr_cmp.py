#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import math
import pprint
import json
from time import time
import re
from random import sample
from time import sleep
import multiprocessing
import pandas as pd

build_train_fund_return = pd.read_csv('./train_data/return_rnn_v1.csv')
build_train_fund_return = build_train_fund_return.set_index('ID')
build_train_fund_return = build_train_fund_return.T
date_lst = list(build_train_fund_return.index.values)

def get_return_end_day(corr_day, date_lst=date_lst, window_size=60):
    corr_day_index = date_lst.index(corr_day)
    end_day_index = corr_day_index + window_size
    try:
        end_day = date_lst[end_day_index]
    except:
        return 0
    return end_day

corr_day = '2018-03-19'

def get_fund_corr_value(fund_name_1, fund_name_2, corr_day, base_data=build_train_fund_return):
    end_day = get_return_end_day(corr_day)
    build_train_fund_return = base_data.loc[(base_data.index >= corr_day) &
                                       (base_data.index <= end_day)]
    raw_corr = build_train_fund_return[[fund_name_1, fund_name_2]].corr()
    corr_result = raw_corr.as_matrix()[0][1]
    return corr_result

###########################################计算结果####################################
submit_result = pd.read_csv('./submit_result/last_day_result.csv')
fund_corr_names = list(submit_result['ID'].values)
result_column_names = ['ID','value']
result_data = []
for fund_corr_name in fund_corr_names:
    fund_name_1, fund_name_2 = fund_corr_name.split('-')
    print(fund_name_1, fund_name_2)
    fund_corr = get_fund_corr_value(fund_name_1, fund_name_2, corr_day='2018-03-19')
    result_data.append([fund_corr_name, fund_corr])
pprint.pprint(result_data)
result_df = pd.DataFrame(result_data, columns=result_column_names)
result_df = result_df.set_index('ID')
result_df.to_csv('./submit_result/rnn_return_v1.csv',index=True, index_label='ID', header=True)