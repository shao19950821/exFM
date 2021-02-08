'''
@author: natsu初夏倾城
@time: 2021/1/20 9:42 上午
@desc: 线性模型
'''

import torch
import torch.nn as nn
from process.feature import *
from process.processUtils import create_embedding_matrix, create_structure_param


class LinearLayer(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu', linear_mask_idx=[]):
        super(LinearLayer, self).__init__()
        if len(linear_mask_idx) > 0:
            feature_columns = self.mask_feature(feature_columns, linear_mask_idx)
        self.feature_index = feature_index
        self.feature_columns = feature_columns

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

    def mask_feature(self, feature_columns, linear_mask_idx):
        filter_feature_columns = []
        for idx, feature in enumerate(feature_columns):
            if idx not in linear_mask_idx:
                filter_feature_columns.append(feature)
        return filter_feature_columns

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        if len(embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(embedding_list, dim=-1), dim=-1, keepdim=False)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit


class NormalizedWeightedLinearLayer(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, alpha_init_mean=0.5, alpha_init_radius=0.001,
                 alpha_activation='tanh', device='cpu', ):
        super(NormalizedWeightedLinearLayer, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)
        self.alpha = create_structure_param(len(self.feature_columns), alpha_init_mean, alpha_init_radius, device)
        self.activate = nn.Tanh() if alpha_activation == 'tanh' else nn.Identity()

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        if len(embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(embedding_list, dim=1).squeeze(-1) * (self.activate(self.alpha)), dim=-1,
                keepdim=True)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit
