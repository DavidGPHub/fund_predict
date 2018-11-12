#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

# 将train和val拼接起来
train_fund_return = pd.read_csv("./offical_data/train_fund_return.csv")
train_fund_return = train_fund_return.rename(columns={train_fund_return.columns[0]: "ID"})
train_fund_return = train_fund_return.set_index('ID')
test_fund_return = pd.read_csv('./offical_data/test_fund_return.csv')
test_fund_return = test_fund_return.rename(columns={test_fund_return.columns[0]: "ID"})
test_fund_return = test_fund_return.set_index('ID')
all_train_fund_return = pd.concat([train_fund_return, test_fund_return], axis=1)
all_train_fund_return.to_csv('./train_data/all_fund_return.csv', index=True, index_label='ID', header=True)
