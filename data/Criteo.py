'''
@author: natsu初夏倾城
@time: 2021/2/4 4:40 下午
@desc:
'''

import csv
import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import numpy as np


dense_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征
sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散型特征
names = [' ', 'label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
         'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
         'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
test_names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
              'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
              'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']


class CriteoProcessor(object):

    def __init__(self, filepath):
        super(CriteoProcessor, self).__init__()
        self.dataDir = '../../criteo/'
        self.feature_map_dir = os.path.join(self.dataDir, 'feature_map')
        self.bucket_dir = os.path.join(self.dataDir, 'bucket')
        self.filepath = filepath
        self.raw_data = self.load_raw_data(filepath)

    def make_feat_map(self):
        dense_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征
        sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散型特征
        self.raw_data[dense_features] = self.raw_data[dense_features].fillna(-1.0, )  # dense填充缺失值
        self.raw_data[sparse_features] = self.raw_data[sparse_features].fillna(0, )  # 填充缺失值
        feat_map = []
        for feature_names in tqdm(dense_features + sparse_features):
            dict = self.raw_data[feature_names].value_counts().to_dict()
            feat_map.append(dict)
        pkl.dump(feat_map, open(os.path.join(self.feature_map_dir, 'feature_map.pkl'), 'wb'))

    def bucket(self):
        feat_map = pkl.load(open(os.path.join(self.feature_map_dir, 'feature_map.pkl'), 'rb'))
        feat_sizes = {}
        num_feat = []
        for i in tqdm(range(13)):
            kv = []
            for k, v in feat_map[i].items():
                if k == '':
                    kv.append([-1, v])
                else:
                    kv.append([k, v])
            kv = sorted(kv, key=lambda x: x[0])
            kv = np.array(kv)
            _s = 0
            thresholds = []
            for j in range(len(kv) - 1):
                _k, _v = kv[j]
                _s += _v
                if _s > 20:
                    thresholds.append(_k)
                    _s = 0
            thresholds = np.array(thresholds)
            num_feat.append(thresholds)
            feat_sizes[dense_features[i]] = len(num_feat[i]) + 1

        cat_feat = []
        for i in tqdm(range(13, 39)):
            cat_feat.append({})
            for k, v in feat_map[i].items():
                if v > 20:
                    cat_feat[i - 13][k] = len(cat_feat[i - 13])
            cat_feat[i - 13]['other'] = len(cat_feat[i - 13])
            feat_sizes[sparse_features[i - 13]] = len(cat_feat[i - 13])
        pkl.dump(num_feat, open(os.path.join(self.bucket_dir, 'dense_feature_bucket.pkl'), 'wb'))
        pkl.dump(cat_feat, open(os.path.join(self.bucket_dir, 'sparse_feature_bucket.pkl'), 'wb'))
        pkl.dump(feat_sizes, open(os.path.join(self.bucket_dir, 'feature_size.pkl'), 'wb'))

    def process_train_sampling_data(self):
        bucket_file = open('../../criteo/train_bucket.csv', 'w', encoding='utf-8', errors='ignore', newline="")
        num_feat = pkl.load(open(os.path.join(self.bucket_dir, 'dense_feature_bucket.pkl'), 'rb'))
        cat_feat = pkl.load(open(os.path.join(self.bucket_dir, 'sparse_feature_bucket.pkl'), 'rb'))
        i = 0
        writer = csv.writer(bucket_file)
        train_data = pd.read_csv('../../criteo/train_sample.csv', sep=',')  # 舍弃unnamed
        for index, row in tqdm(train_data.iterrows()):
            if i == 0:
                writer.writerow(names)
            for dense_index, dense in enumerate(dense_features):
                value = row[dense]
                row[dense] = len(
                    np.where((num_feat[dense_index] < value) & (abs(num_feat[dense_index] - value) > 1e-8))[0])
            for sparse_index, sparse in enumerate(sparse_features):
                value = row[sparse]
                if value in cat_feat[sparse_index]:
                    row[sparse] = cat_feat[sparse_index][value]
                else:
                    row[sparse] = cat_feat[sparse_index]['other']
            writer.writerow(row)
            i += 1

    def process_test_raw_data(self):
        bucket_file = open('../../criteo/test_bucket.csv', 'w', encoding='utf-8', errors='ignore', newline="")
        num_feat = pkl.load(open(os.path.join(self.bucket_dir, 'dense_feature_bucket.pkl'), 'rb'))
        cat_feat = pkl.load(open(os.path.join(self.bucket_dir, 'sparse_feature_bucket.pkl'), 'rb'))
        i = 0
        writer = csv.writer(bucket_file)
        test_data = pd.read_csv('../../criteo/test.csv', sep=',')  # 舍弃unnamed
        for index in tqdm(range(len(test_data))):
            row = test_data.iloc[index]
            if i == 0:
                writer.writerow(test_names)
            for dense_index, dense in enumerate(dense_features):
                value = row[dense]
                row[dense] = len(
                    np.where((num_feat[dense_index] < value) & (abs(num_feat[dense_index] - value) > 1e-8))[0])
            for sparse_index, sparse in enumerate(sparse_features):
                value = row[sparse]
                if value in cat_feat[sparse_index]:
                    row[sparse] = cat_feat[sparse_index][value]
                else:
                    row[sparse] = cat_feat[sparse_index]['other']
            writer.writerow(row)
            i += 1

    def load_raw_data(self, filepath):
        data = pd.read_csv(filepath, sep=',')  # 舍弃unnamed
        return data

    def negative_down_sampling(self):
        data = pd.DataFrame(self.raw_data)  # 服务器上读取数据的方法
        neg_cnt = 0
        pos_cnt = 0
        for row in tqdm(data.itertuples()):
            if getattr(row, 'label') == 1:
                pos_cnt += 1
            else:
                neg_cnt += 1
        neg_threshold = pos_cnt * 1. / neg_cnt
        print('pos_cnt', pos_cnt, 'neg_cnt', neg_cnt, 'neg_threshold:', neg_threshold)
        sample_file = open('../../criteo/train_sample.csv', 'w', encoding='utf-8', errors='ignore', newline="")
        writer = csv.writer(sample_file)
        i = 0
        neg_cnt = 0
        pos_cnt = 0
        for row in tqdm(data.itertuples()):
            if i == 0:
                writer.writerow(names)
            if getattr(row, 'label') == 1.:
                writer.writerow(row)
                pos_cnt += 1
            elif np.random.random() < neg_threshold:
                writer.writerow(row)
                neg_cnt += 1
            i = i + 1
        neg_threshold = pos_cnt * 1. / neg_cnt
        print('pos_cnt', pos_cnt, 'neg_cnt', neg_cnt, 'neg_threshold:', neg_threshold)
        sample_file.close()


if __name__ == '__main__':
    dataProcessor = CriteoProcessor('../../criteo/train.csv')
    dataProcessor.negative_down_sampling()
    dataProcessor.make_feat_map()
    dataProcessor.bucket()
    dataProcessor.process_train_sampling_data()
    dataProcessor.process_test_raw_data()
