'''
@author: natsu初夏倾城
@time: 2021/2/16 10:12 上午
@desc:
'''

import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def before_train(self):
        """Be called before the training process."""

    def fit(self):
        """training process."""

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

    def after_train(self):
        """Be called after the training process."""
