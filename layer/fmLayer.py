'''
@author: natsu初夏倾城
@time: 2021/1/20 10:15 上午
@desc:
'''

import torch
import torch.nn as nn
from process.processUtils import build_input_features, create_embedding_matrix, create_structure_param, \
    generate_pair_index
from process.feature import SparseFeat, DenseFeat, DenseBucketFeat


class FactorizationMachineLayer(nn.Module):
    def __init__(self, feature_columns, init_std=0.0001, seed=1024, device='cpu', pair_mask_idx=[]):
        super(FactorizationMachineLayer, self).__init__()
        torch.manual_seed(seed)
        self.pair_mask_idx = pair_mask_idx
        self.feature_columns = feature_columns
        self.feature_index = build_input_features(feature_columns)
        # embedding 矩阵
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, sparse=False,
                                                      device=device)
        self.device = device
        self.inputdim = len(self.feature_columns)
        if len(self.pair_mask_idx) > 0:
            self.register_buffer('pair_indexes',
                                 torch.tensor(
                                     generate_pair_index(self.inputdim, 2)).to(device))
            self.pair_mask_idx = pair_mask_idx

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        embed_matrix = torch.cat(embedding_list, dim=1)
        square_of_sum = torch.sum(embed_matrix, dim=1) ** 2
        sum_of_square = torch.sum(embed_matrix ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        if len(self.pair_mask_idx) > 0:
            mask_feat_i, mask_feat_j = torch.index_select(self.pair_indexes, 1,
                                                          torch.LongTensor(self.pair_mask_idx).to(self.device))
            mask_embed_i = torch.index_select(embed_matrix, 1, mask_feat_i)
            mask_embed_j = torch.index_select(embed_matrix, 1, mask_feat_j)
            mask_embed_product = torch.sum(torch.mul(mask_embed_i, mask_embed_j), dim=2)
            return 0.5 * ix - torch.sum(mask_embed_product, dim=1, keepdim=True)
        else:
            return 0.5 * ix


class NormalizedWeightedFMLayer(torch.nn.Module):

    def __init__(self, feature_columns, feature_index, init_std=0.0001, beta_init_mean=0.5, beta_init_radius=0.001,
                 beta_activation='tanh', selected_pairs=None,
                 seed=1024, device='cpu'):
        super(NormalizedWeightedFMLayer, self).__init__()
        self.feature_columns = feature_columns
        torch.manual_seed(seed)
        self.feature_index = feature_index
        self.inputdim = len(self.feature_columns)
        self.register_buffer('pair_indexes',
                             torch.tensor(generate_pair_index(self.inputdim, 2, selected_pairs)).to(device))
        interaction_pair_number = len(self.pair_indexes[0])

        # embedding 矩阵
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, sparse=False,
                                                      device=device)

        self.beta = create_structure_param(
            interaction_pair_number, beta_init_mean,
            beta_init_radius, device)
        self.activate = nn.Tanh() if beta_activation == 'tanh' else nn.Identity()
        self.batch_norm = torch.nn.BatchNorm1d(interaction_pair_number, affine=False, momentum=0.01, eps=1e-3)

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        embed_matrix = torch.cat(embedding_list, dim=1)
        feat_i, feat_j = self.pair_indexes
        embed_i = torch.index_select(embed_matrix, 1, feat_i)
        embed_j = torch.index_select(embed_matrix, 1, feat_j)
        embed_product = torch.sum(torch.mul(embed_i, embed_j), dim=2)
        normed_emded_product = self.batch_norm(embed_product)
        weighted_embed_product = torch.mul(normed_emded_product, self.activate(self.beta.unsqueeze(0)))
        return torch.sum(weighted_embed_product, dim=1, keepdim=True)
