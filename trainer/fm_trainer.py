'''
@author: natsu初夏倾城
@time: 2021/1/28 12:19 下午
@desc:
'''
import logging
from model.fm import FM
from config.exFM_config import exFM_config as configs
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from process.processUtils import *

filename = 'log-fm.log'
logging.basicConfig(filename=filename, level=logging.INFO, filemode='w')


def run_criteo(feature_columns, feature_index, data, device='cpu'):
    target = ['label']
    feature_names = get_feature_names(feature_columns)
    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    model = FM(feature_columns=feature_columns, feature_index=feature_index,
               learning_rate=configs['general']['learning_rate'],
               device=device)  # 初始化模型
    model.to(device)
    model.before_train()
    model.fit(train_model_input, train[target].values, batch_size=configs['general']['batch_size'],
              epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])

    pred_ans = model.predict(test_model_input, 256)
    logging.info("test LogLoss:{}".format(round(log_loss(test[target].values, pred_ans), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(test[target].values, pred_ans), 4)))
