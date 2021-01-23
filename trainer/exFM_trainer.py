'''
@author: natsu初夏倾城
@time: 2021/1/20 5:33 下午
@desc:
'''
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from process.processUtils import *
from process.feature import SparseFeat, DenseFeat
from model.exfm import exFM

if __name__ == "__main__":
    data = pd.read_csv('../criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征

    data[sparse_features] = data[sparse_features].fillna('-1', )  # 填充缺失值
    data[dense_features] = data[dense_features].fillna(0, )  # 填充缺失值
    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()  # LabelEncoder 对离散数据进行标签化 字典序列 如['dog','cat','mouse'] 变为[1,0,2]
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))  # 将连续型变量放缩到(0,1)范围
    data[dense_features] = mms.fit_transform(data[dense_features])


    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    feature_index = build_input_features(linear_feature_columns)

    model = exFM(feature_columns=linear_feature_columns, feature_index=feature_index)  # 初始化Deep模型


    model.before_train()
    model.fit(train_model_input, train[target].values, batch_size=32, epochs=30, validation_split=0.2)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
