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
    def __init__(self, feature_columns, feature_index,init_std=0.0001, device='cpu',linear_mask_idx = []):
        super(LinearLayer, self).__init__()
        if len(linear_mask_idx) > 0:
            feature_columns =self.mask_feature(feature_columns,linear_mask_idx)
        self.feature_index = feature_index
        self.sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
            feature_columns) else []
        self.dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(
            feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        if len(self.dense_feat_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feat_columns), 1).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def mask_feature(self,feature_columns,linear_mask_idx):
        filter_feature_columns = []
        for idx,feature in enumerate(feature_columns):
            if idx not in linear_mask_idx:
                filter_feature_columns.append(feature)
        return filter_feature_columns

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.sparse_feat_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feat_columns]

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit


class NormalizedWeightedLinearLayer(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, alpha_init_mean=0.5, alpha_init_radius=0.001,
                 alpha_activation='tanh', device='cpu', ):
        super(NormalizedWeightedLinearLayer, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
            feature_columns) else []
        self.dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(
            feature_columns) else []
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)
        self.alpha = create_structure_param(
            len(self.sparse_feat_columns) + sum(fc.dimension for fc in self.dense_feat_columns), alpha_init_mean,
            alpha_init_radius, device)
        if len(self.dense_feat_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feat_columns), 1).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)
        self.activate = nn.Tanh() if alpha_activation == 'tanh' else nn.Identity()

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.sparse_feat_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feat_columns]

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.cat(sparse_embedding_list, dim=1).squeeze(-1)
            linear_dense_logit = torch.cat(dense_value_list, dim=-1) * (self.weight).transpose(1, 0)
            linear_logit = torch.sum(
                torch.cat([linear_sparse_logit, linear_dense_logit], dim=1) * (self.activate(self.alpha)), dim=-1,
                keepdim=True)
        elif len(sparse_embedding_list) > 0:
            linear_sparse_logit = torch.cat(sparse_embedding_list, dim=1).squeeze(-1)
            linear_logit = torch.sum(linear_sparse_logit * (self.activate(self.alpha)), dim=-1, keepdim=True)
        elif len(dense_value_list) > 0:
            linear_dense_logit = torch.cat(dense_value_list, dim=-1) * (self.weight).transpose(1, 0)
            linear_logit = torch.sum(linear_dense_logit * (self.activate(self.alpha)), dim=-1, keepdim=True)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit
