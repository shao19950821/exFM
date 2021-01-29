'''
@author: natsu初夏倾城
@time: 2021/1/28 12:06 下午
@desc:
'''
import math
import time
from model.fm import FM
from model.exfm import exFM
from config.exFM_config import exFM_config as configs
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from process.processUtils import *

filename = 'log-c' + str(configs['gRDA']['c']) + '.log'
logging.basicConfig(filename=filename, level=logging.INFO, filemode='w')


def run_criteo(feature_columns, feature_index, data, device='cpu'):
    target = ['label']
    feature_names = get_feature_names(feature_columns)
    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    logging.info("data num:{}".format(configs['general']['data']))
    logging.info("epoch num:{}".format(configs['general']['epochs']))
    period1_model = exFM(feature_columns=feature_columns, feature_index=feature_index,
                         net_learning_rate=configs['general']['learning_rate'], c=configs['gRDA']['c'],
                         mu=configs['gRDA']['mu'], structure_learing_rate=configs['gRDA']['learning_rate'],
                         device=device)  # 初始化模型
    period1_model.to(device)
    period1_model.before_train()
    period1_train_start_time = time.time()
    period1_model.fit(train_model_input, train[target].values, batch_size=configs['general']['batch_size'],
                      epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])
    period1_train_cost_time = (int)(time.time() - period1_train_start_time)
    period1_test_start_time = time.time()
    period1_model_pred_ans = period1_model.predict(test_model_input, 256)
    period1_test_cost_time = (int)(time.time() - period1_test_start_time)
    logging.info("period1:test LogLoss:{}".format(round(log_loss(test[target].values, period1_model_pred_ans), 4)))
    logging.info("period1:AUC:{}".format(round(roc_auc_score(test[target].values, period1_model_pred_ans), 4)))
    logging.info("period1:train cost time:{}".format(period1_train_cost_time))
    logging.info("period1:test cost time:{}".format(period1_test_cost_time))
    period1_model.afterTrain()

    # mask多余特征和多余特征交互
    alpha = period1_model.linear.alpha.data.numpy()
    beta = period1_model.fm.beta.data.numpy()

    linear_mask_idx = []
    pair_mask_idx = []
    for idx, score in enumerate(alpha):
        if math.fabs(score) <= configs['exFM']['mask_threshold']:
            linear_mask_idx.append(idx)

    for idx, score in enumerate(beta):
        if math.fabs(score) <= configs['exFM']['mask_threshold']:
            pair_mask_idx.append(idx)

    period2_model = FM(feature_columns=feature_columns, feature_index=feature_index,
                       learning_rate=configs['general']['learning_rate'],
                       device=device,linear_mask_idx=linear_mask_idx,pair_mask_idx=pair_mask_idx)  # 初始化模型
    period2_model.to(device)
    period2_model.before_train()
    period2_train_start_time = time.time()
    period2_model.fit(train_model_input, train[target].values, batch_size=configs['general']['batch_size'],
                      epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])
    period2_train_cost_time = (int)(time.time() - period2_train_start_time)
    period2_test_start_time = time.time()
    period2_model_pred_ans = period2_model.predict(test_model_input, 256)
    period2_test_cost_time = (int)(time.time() - period2_test_start_time)
    logging.info("test LogLoss:{}".format(round(log_loss(test[target].values, period2_model_pred_ans), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(test[target].values, period2_model_pred_ans), 4)))
    logging.info("period2:train cost time:{}".format(period2_train_cost_time))
    logging.info("period2:test cost time:{}".format(period2_test_cost_time))
