'''
@author: natsu初夏倾城
@time: 2021/2/16 9:53 上午
@desc:
'''
import os
import time
from model.alphaBetaFM import AlphaBetaFM
from config.exFM_config import exFM_config as configs
from sklearn.metrics import log_loss, roc_auc_score
from process.processUtils import *
from model.weightFM import WeightFM

filename = 'weightFM-batchsize'+ str(configs['general']['batch_size']) +'-c-' + str(configs['gRDA']['c']) + '.log'
logging.basicConfig(filename=filename, level=logging.INFO, filemode='w')


def run_criteo(feature_columns, feature_index, data_train, data_test, device='cpu'):
    alpha = ""
    beta = ""
    target = ['label']
    feature_names = get_feature_names(feature_columns)
    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}
    logging.info("data num:{}".format(configs['general']['data']))
    logging.info("epoch num:{}".format(configs['general']['epochs']))
    if not os.path.exists('paramc-' + str(configs['gRDA']['c']) + '.pth'):
        period1_model = AlphaBetaFM(feature_columns=feature_columns, feature_index=feature_index,
                                    net_learning_rate=configs['general']['learning_rate'], c=configs['gRDA']['c'],
                                    mu=configs['gRDA']['mu'], structure_learing_rate=configs['gRDA']['learning_rate'],
                                    device=device)  # 初始化模型
        period1_model.to(device)
        period1_model.before_train()
        period1_train_start_time = time.time()
        period1_model.fit(train_model_input, data_train[target].values, batch_size=configs['general']['batch_size'],
                          epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])
        period1_train_cost_time = (int)(time.time() - period1_train_start_time)
        period1_test_start_time = time.time()
        period1_model_pred_ans = period1_model.predict(test_model_input, 256)
        period1_test_cost_time = (int)(time.time() - period1_test_start_time)
        logging.info(
            "period1:test LogLoss:{}".format(round(log_loss(data_test[target].values, period1_model_pred_ans), 4)))
        logging.info("period1:AUC:{}".format(round(roc_auc_score(data_test[target].values, period1_model_pred_ans), 4)))
        logging.info("period1:train cost time:{}".format(period1_train_cost_time))
        logging.info("period1:test cost time:{}".format(period1_test_cost_time))
        period1_model.afterTrain()
        alpha = period1_model.linear.alpha
        beta = period1_model.fm.beta
    else:
        print("load the structure param")
        checkpoint = torch.load('paramc-' + str(configs['gRDA']['c']) + '.pth')
        alpha = checkpoint['alpha']
        beta = checkpoint['beta']
    period2_model = WeightFM(feature_columns=feature_columns, feature_index=feature_index,
                             net_learning_rate=configs['general']['learning_rate'], alpha=alpha, beta=beta,
                             device=device)  # 初始化模型,
    period2_model.to(device)
    period2_model.before_train()
    period2_train_start_time = time.time()
    period2_model.fit(train_model_input, data_train[target].values, batch_size=configs['general']['batch_size'],
                      epochs=configs['general']['epochs'], validation_split=configs['general']['validation_split'])
    period2_train_cost_time = (int)(time.time() - period2_train_start_time)
    period2_test_start_time = time.time()
    period2_model_pred_ans = period2_model.predict(test_model_input, 256)
    period2_test_cost_time = (int)(time.time() - period2_test_start_time)
    logging.info("test LogLoss:{}".format(round(log_loss(data_test[target].values, period2_model_pred_ans), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(data_test[target].values, period2_model_pred_ans), 4)))
    logging.info("period2:train cost time:{}".format(period2_train_cost_time))
    logging.info("period2:test cost time:{}".format(period2_test_cost_time))
