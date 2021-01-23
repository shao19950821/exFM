'''
@author: natsu初夏倾城
@time: 2021/1/20 10:29 上午
@desc:
'''

import torch

if __name__ == '__main__':

    a = torch.Tensor([[0,1,2],[1,2,3]])
    b = torch.Tensor([[0,1,2],[1,2,3]])
    print(torch.cat([a,b] ,dim = 1).shape)
    print(torch.cat([a,b] ,dim = 1))