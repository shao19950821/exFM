'''
@author: natsu初夏倾城
@time: 2021/1/28 12:19 下午
@desc:
'''
import time
from model.fm import FM
from config.exFM_config import exFM_config as configs
from sklearn.metrics import log_loss, roc_auc_score
from process.processUtils import *

filename = 'log-fm.log'
logging.basicConfig(filename=filename, level=logging.INFO, filemode='w')


def run_criteo(feature_columns, feature_index, data_train, data_test, device='cpu'):
    target = ['label']
    feature_names = get_feature_names(feature_columns)
    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}
    logging.info("data num:{}".format(configs['general']['data']))
    logging.info("epoch num:{}".format(configs['general']['epochs']))
    model = FM(feature_columns=feature_columns, feature_index=feature_index,
               learning_rate=configs['general']['learning_rate'],
               device=device)  # 初始化模型
    model.to(device)
    model.before_train()
    train_start_time = time.time()
    model.fit(train_model_input, data_train[target].values, batch_size=configs['general']['batch_size'],
              epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])
    train_cost_time = (int)(time.time() - train_start_time)
    test_start_time = time.time()
    pred_ans = model.predict(test_model_input, 256)
    test_cost_time = (int)(time.time() - test_start_time)
    logging.info("test LogLoss:{}".format(round(log_loss(data_test[target].values, pred_ans), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(data_test[target].values, pred_ans), 4)))
    logging.info("train cost time:{}".format(train_cost_time))
    logging.info("test cost time:{}".format(test_cost_time))
