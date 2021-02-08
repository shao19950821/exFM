'''
@author: natsu初夏倾城
@time: 2021/2/2 2:18 下午
@desc:
'''

import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    def __init__(self, params, alpha=0.001, beta=0.0005, l1=0.8, l2=0.2):
        """
        Constuct gRDA class.

        :param params:  learnable  params
        :type params: list object
        :param lr:  learning rate
        :type lr: float
        :param c:  initial sparse control constant
        :type c: float
        :param mu:  sparsity control
        :type mu: float

        :return: optimizer object
        :rtype: class
        """
        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super(FTRL, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Setstate."""
        super(FTRL, self).__setstate__(state)

    def step(self, closure=None):
        """
        Optimizer gRDA performs a single optimization step.

        :param closure:  a closure that reevaluates the model
        :type closure: callable object
        :return: loss
        :rtype: float
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            lambda1 = group['lambda1']
            lambda2 = group['lambda2']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'iter_num' not in param_state:
                    iter_num = param_state['iter_num'] = torch.zeros(1)
                    accumulator = param_state['accumulator'] = torch.FloatTensor(p.shape).to(p.device)
                    l1_accumulation = param_state['l1_accumulation'] = torch.zeros(1)
                    accumulator.data = p.clone()
                else:
                    iter_num = param_state['iter_num']
                    accumulator = param_state['accumulator']
                    l1_accumulation = param_state['l1_accumulation']
                iter_num.add_(1)
                accumulator.data.add_(-lr, d_p)
                l1_diff = c * torch.pow(torch.tensor(lr), mu + 0.5) * torch.pow(iter_num, mu) - c * torch.pow(
                    torch.tensor(lr), mu + 0.5) * torch.pow(iter_num - 1, mu)
                l1_accumulation += l1_diff
                new_a_l1 = torch.abs(accumulator.data) - l1_accumulation.to(p.device)
                p.data = torch.sign(accumulator.data) * new_a_l1.clamp(min=0)
        return loss
