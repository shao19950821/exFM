'''
@author: natsu初夏倾城
@time: 2021/1/23 9:35 下午
@desc: 配置调整
'''

exFM_config = {
    'general':{
        'batch_size':32,
        'epochs':30,
        'validation_split':0.2
    },
    'exFm': {
        'learning_rate': 1e-3,
    },
    'gRDA': {
        'c': 0.0005,
        'mu': 0.8,
        'learning_rate': 1e-3,
    }
}
