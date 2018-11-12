#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
res_1 = pd.read_csv('./submit_result/rnn_corr_v1.csv').set_index('ID')
res_2 = pd.read_csv('./submit_result/rnn_return_v1.csv').set_index('ID')
sub = 0.95*res_2 + 0.05*res_1
res_2.to_csv('./submit_result/sub_2.csv', index=True, index_label='id', header=True)
