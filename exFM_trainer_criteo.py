'''
@author: natsu初夏倾城
@time: 2021/1/20 5:33 下午
@desc: criteo数据集测试
'''
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from process.processUtils import *
from process.feature import SparseFeat, DenseFeat
from model.exfm import exFM
from config.exFM_config import exFM_config as configs

filename = 'log-c'+str(configs['gRDA']['c'])+'.log'
logging.basicConfig(filename=filename,level=logging.INFO,filemode='w')

if __name__ == "__main__":
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
             'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
             'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    data_train = pd.read_csv('../criteo/train.txt', nrows=500000,sep='\t', names=names)  # 服务器上读取数据的方法
    data_test = pd.read_csv('../criteo/test.txt', nrows=10000, sep='\t', names=names)  # 服务器上读取数据的方法

    # data = pd.read_csv('../criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征
    data_train[sparse_features] = data_train[sparse_features].fillna('-1', )  # 填充缺失值
    data_train[dense_features] = data_train[dense_features].fillna(0, )  # 填充缺失值
    data_test[sparse_features] = data_test[sparse_features].fillna('-1', )  # 填充缺失值
    data_test[dense_features] = data_test[dense_features].fillna(0, )  # 填充缺失值
    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()  # LabelEncoder 对离散数据进行标签化 字典序列 如['dog','cat','mouse'] 变为[1,0,2]
        data_train[feat] = lbe.fit_transform(data_train[feat])
        data_test[feat] = lbe.fit_transform(data_test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))  # 将连续型变量放缩到(0,1)范围
    data_train[dense_features] = mms.fit_transform(data_train[dense_features])
    data_test[dense_features] = mms.fit_transform(data_test[dense_features])

    fixlen_feature_columns = [SparseFeat(feat, data_train[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]


    feature_names = get_feature_names(fixlen_feature_columns)

    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:1'

    feature_index = build_input_features(fixlen_feature_columns)

    model = exFM(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                 net_learning_rate=configs['exFm']['learning_rate'], c=configs['gRDA']['c'],
                 mu=configs['gRDA']['mu'], structure_learing_rate=configs['gRDA']['learning_rate'],device=device)  # 初始化模型
    model.to(device)

    model.before_train()
    model.fit(train_model_input, data_train[target].values, batch_size=configs['general']['batch_size'],
              epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])

    pred_ans = model.predict(test_model_input, 256)
    logging.info("test LogLoss:{}".format(round(log_loss(data_test[target].values, pred_ans), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(data_test[target].values, pred_ans), 4)))
    model.afterTrain()
