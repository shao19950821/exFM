'''
@author: natsu初夏倾城
@time: 2021/1/23 9:35 下午
@desc: 配置调整
'''

exFM_config = {
    'general': {
        'batch_size': 2000,
        'data': -1,  # -1 代表全部读取
        'epochs': 1,
        'validation_split': 0.1,
        'learning_rate': 1e-3,
    },
    'exFM': {
        'mask_threshold': 0.,
    },
    'gRDA': {
        'c': 0.05,
        'mu': 0.8,
        'learning_rate': 1e-3,
    }
}
