'''
@author: natsu初夏倾城
@time: 2021/1/20 10:26 上午
@desc:
'''

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from optimizer.gRDA import gRDA
from sklearn.metrics import *
import torch.utils.data as Data
import logging
from layer.fmLayer import NormalizedWeightedFMLayer
from layer.linearLayer import NormalizedWeightedLinearLayer
from process.processUtils import slice_arrays
from tqdm import tqdm
from .baseModel import BaseModel
import time


class AlphaBetaFM(BaseModel):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, net_learning_rate=1e-3,
                 alpha_init_mean=0.5, alpha_init_radius=0.001, beta_init_mean=0.5, beta_init_radius=0.001,
                 activation='tanh', selected_pairs=None,
                 c=0.0005, mu=0.8, structure_learing_rate=1e-3, seed=1024, device='cpu'):
        super(AlphaBetaFM, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.linear = NormalizedWeightedLinearLayer(feature_columns=feature_columns, feature_index=feature_index,
                                                    init_std=init_std, alpha_init_mean=alpha_init_mean,
                                                    alpha_init_radius=alpha_init_radius,
                                                    alpha_activation=activation,
                                                    device=device)
        self.fm = NormalizedWeightedFMLayer(feature_columns=feature_columns, feature_index=feature_index,
                                            init_std=init_std, beta_init_mean=beta_init_mean,
                                            beta_init_radius=beta_init_radius,
                                            beta_activation=activation, selected_pairs=selected_pairs,
                                            seed=seed,
                                            device=device)

        self.net_lr = net_learning_rate  # 学习率
        self.structure_lr = structure_learing_rate  # 学习率
        self.c = c  # gRDA c
        self.mu = mu  # gRDA mu

    def forward(self, x):
        linear_out = self.linear(x)
        fm_out = self.fm(x)
        out = linear_out + fm_out
        return torch.sigmoid(out)

    def before_train(self):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_params = set([self.linear.alpha, self.fm.beta])
        net_params = [i for i in all_parameters if i not in structure_params]
        self.structure_optim = self.get_structure_optim(structure_params)
        self.net_optim = self.get_net_optim(net_params)
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_net_optim(self, learnable_params):
        logging.info("init net optimizer, lr = {}".format(self.net_lr))
        optimizer = optim.Adam(learnable_params, lr=float(self.net_lr))
        logging.info("init net optimizer finish.")
        return optimizer

    def get_structure_optim(self, learnable_params):
        logging.info("init structure optimizer, lr = {}, c = {}, mu = {}".format(self.structure_lr, self.c, self.mu))
        optimizer = gRDA(learnable_params, lr=float(self.structure_lr), c=self.c, mu=self.mu)
        logging.info("init structure optimizer finish.")
        return optimizer

    def get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def fit(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
            shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        print(self.device, end="\n")
        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        y_pred = model(x).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        total_loss_epoch += loss.item()
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            print('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s - loss: {1: .4f}".format(
                epoch_time, epoch_logs["loss"])

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            print(eval_str)


    def get_linearlayer_feature_interaction_score(self):
        linear_interaction_score = {}
        for idx, feature in enumerate(self.linear.feature_columns):
            name = feature.name
            score = self.linear.alpha[idx].item()
            logging.info("feature {} => importance score {}".format(name, score))
            linear_interaction_score[name] = score
        return linear_interaction_score

    def get_fmlayer_feature_interaction_score(self):
        feature_interaction_score = {}
        feat_i, feat_j = self.fm.pair_indexes.tolist()
        feat_name_i = [self.fm.feature_columns[i].name for i in feat_i]
        feat_name_j = [self.fm.feature_columns[j].name for j in feat_j]
        for idx, pair in enumerate(zip(feat_name_i, feat_name_j)):
            score = self.fm.beta[idx].item()
            logging.info("pair {} => importance score {}".format(pair, score))
            feature_interaction_score[pair] = score
        return feature_interaction_score

    def save(self):
        print("save model structure param")
        state = {'alpha': self.linear.alpha,
                 'beta': self.fm.beta,
                 }
        torch.save(state, 'paramc-' + str(self.c) + '.pth')

    # 训练后输出权重
    def afterTrain(self):
        self.get_linearlayer_feature_interaction_score()
        self.get_fmlayer_feature_interaction_score()
        self.save()
