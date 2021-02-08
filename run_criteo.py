'''
@author: natsu初夏倾城
@time: 2021/1/20 5:33 下午
@desc: criteo数据集测试
'''
# -*- coding: utf-8 -*-
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from process.processUtils import build_input_features
from process.feature import SparseFeat, DenseFeat, DenseBucketFeat
from config.exFM_config import exFM_config as configs
import pickle as pkl


def train_criteo(model):
    if model == 'exFM':
        from trainer.exFM_trainer import run_criteo
    elif model == 'fm':
        from trainer.fm_trainer import run_criteo
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
             'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
             'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    feat_size = pkl.load(open('../criteo/bucket/feature_size.pkl', 'rb'))
    if configs['general']['data'] == -1:
        data_train = pd.read_csv('../criteo/train_bucket.csv', sep=',', dtype=int)  # 服务器上读取数据的方法
        data_test = pd.read_csv('../criteo/test_bucket.csv', sep=',',dtype=int)  # 服务器上读取数据的方法
    else:
        data_train = pd.read_csv('../criteo/train_bucket.csv', nrows=configs['general']['data'], sep=',',
                                 dtype=int)  # 服务器上读取数据的方法
        data_test = pd.read_csv('../criteo/test_bucket.csv', nrows=configs['general']['data'], sep=',',
                                dtype=int)  # 服务器上读取数据的方法
    sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征
    fixlen_feature_columns = [SparseFeat(feat, feat_size[feat], 20)
                              for feat in sparse_features] + [DenseBucketFeat(feat, feat_size[feat], 20)
                                                              for feat in dense_features]
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:3'
    feature_index = build_input_features(fixlen_feature_columns)
    run_criteo(feature_columns=fixlen_feature_columns, feature_index=feature_index, data_train=data_train,
               data_test=data_test, device=device)
