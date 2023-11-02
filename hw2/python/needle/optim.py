"""Optimization module"""
from collections import defaultdict
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        # 相比于self.u={}，这样创建字典不会出现key不存在的错误，可以少写点判断
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        # BEGIN YOUR SOLUTION
        for parm in self.params:
            if self.weight_decay > 0:
                grad = parm.grad.data + self.weight_decay * parm.data
            else:
                grad = parm.grad.data
            self.u[parm] = self.momentum * \
                self.u[parm] + (1 - self.momentum) * grad
            parm.data = parm.data - self.lr * self.u[parm]
        # END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
            self,
            params,
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        # BEGIN YOUR SOLUTION
        self.t += 1
        for parm in self.params:
            if self.weight_decay > 0:
                grad = parm.grad.data + self.weight_decay * parm.data
            else:
                grad = parm.grad.data
            self.m[parm] = self.beta1 * self.m[parm] + (1 - self.beta1) * grad
            self.v[parm] = self.beta2 * self.v[parm] + \
                (1 - self.beta2) * (grad * grad)
            bm = self.m[parm] / (1 - self.beta1**self.t)
            bv = self.v[parm] / (1 - self.beta2**self.t)
            parm.data = parm.data - self.lr * bm / (bv**(1 / 2) + self.eps)
        # END YOUR SOLUTION
