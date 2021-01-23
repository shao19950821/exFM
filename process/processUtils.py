'''
@author: natsu初夏倾城
@time: 2021/1/19 11:27 下午
@desc:
'''

import numpy as np
import logging
import torch
import torch.nn as nn
from itertools import combinations
from .feature import SparseFeat, DenseFeat
from collections import OrderedDict


# Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns}
    )

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)  # 正态分布初始化权重，N（mean，std)

    return embedding_dict.to(device)

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


#Return OrderedDict: {feature_name:(start, start+dimension)}
def build_input_features(feature_columns):
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def generate_pair_index(n, order=2, selected_pairs=None):
    """Return enumeration of feature combination pair index.

    :param n: number of valid features, usually equals to `input_dim4lookup`
    :type n: int
    :param order: order of interaction. defaults to 2
    :type order: int
    :param selected_pairs: specifying selected pair of index
    :type selected_pairs: sequence of tuples, optional
    :return: a list of tuple, each containing feature index
    :rtype: list of tuple

    :Example:

    >>> generate_pair_index(5, 2)
    >>> [(0, 0, 0, 0, 1, 1, 1, 2, 2, 3),
         (1, 2, 3, 4, 2, 3, 4, 3, 4, 4)]
    >>> generate_pair_index(5, 3)
    >>> [(0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
         (1, 1, 1, 2, 2, 3, 2, 2, 3, 3),
         (2, 3, 4, 3, 4, 4, 3, 4, 4, 4)]
    >>> generate_pair_index(5, 2, [(0,1),(1,3),(2,3)])
    >>> [(0, 1, 2), (1, 3, 3)]
    """
    if n < 2:
        raise ValueError("undefined. please ensure n >= 2")
    pairs = list(combinations(range(n), order))
    if selected_pairs is not None and len(selected_pairs) > 0:
        valid_pairs = set(selected_pairs)
        pairs = list(filter(lambda x: x in valid_pairs, pairs))
        logging.info("Using following selected feature pairs \n{}".format(pairs))
        if len(pairs) != len(selected_pairs):
            logging.warning("Pair number {} != specified pair number {}".format(len(pairs), len(selected_pairs)))
    return list(zip(*pairs))


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError

def slice_arrays(arrays, start=None, stop=None):

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]

