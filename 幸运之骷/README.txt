一、利用corr数据直接进行训练，进行数据滑窗处理，采用Attention-RNN模型
直接运行corr_nn.py即可， attention-rnn可以保存模型，但是读取模型时会出错。所以直接模型训练预测直接进行了。

二、利用return数据直接进行训练，进行数据滑窗处理，采用RNN模型
1、运行nn_process_data.py ，将train和val拼接成所有数据，并进行保存。
2、运行return_nn_train.py， 进行模型训练。
3、运行return_nn_pred.py， 进行模型预测。
4、运行corr_cmp.py 计算出相关性结果。

三、模型融合
1、运行result_blend.py。计算出模型融合的结果。