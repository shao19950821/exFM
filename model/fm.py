'''
@author: natsu初夏倾城
@time: 2021/1/28 12:49 下午
@desc:
'''

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import *
import torch.utils.data as Data
import torch.nn as nn
import logging
from layer.fmLayer import FactorizationMachineLayer
from layer.linearLayer import LinearLayer
from process.processUtils import slice_arrays
from tqdm import tqdm
import time


class FM(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, learning_rate=1e-3,
                 seed=1024, device='cpu', linear_mask_idx=[], pair_mask_idx=[]):
        super(FM, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.linear = LinearLayer(feature_columns=feature_columns, feature_index=feature_index, init_std=init_std,
                                  device=device, linear_mask_idx=linear_mask_idx)
        self.fm = FactorizationMachineLayer(feature_columns=feature_columns, init_std=init_std, seed=seed,
                                            device=device, pair_mask_idx=pair_mask_idx)
        self.lr = learning_rate

    def forward(self, x):
        linear_out = self.linear(x)
        fm_out = self.fm(x)
        out = linear_out + fm_out
        return torch.sigmoid(out)

    def before_train(self):
        self.metrics_names = ["loss"]
        self.net_optim = self.get_net_optim(self.parameters())
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_net_optim(self, learnable_params):
        logging.info("init net optimizer, lr = {}".format(self.lr))
        optimizer = optim.Adam(learnable_params, lr=float(self.lr))
        logging.info("init net optimizer finish.")
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
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
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
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        total_loss_epoch += loss.item()
                        loss.backward()
                        net_optim.step()
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

            epoch_time = int(time.time() - start_time)
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

    def evaluate(self, x, y, batch_size=256):
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")
