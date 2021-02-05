'''
@author: natsu初夏倾城
@time: 2021/1/27 4:39 下午
@desc:
'''

import argparse
from run_criteo import train_criteo

parser = argparse.ArgumentParser(description='PyTorch exFm example')
parser.add_argument('--model', type=str, default='exFM',
                    help='fm - Factorization Machine  exFM - explainable Factorization Machine')
args = parser.parse_args()
model = args.model
train_criteo(model)