from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        maxZ = array_api.amax(Z, axis=self.axes, keepdims=True)
        z_exp_minus = array_api.exp(Z - maxZ)
        z_sum = array_api.sum(z_exp_minus, axis=self.axes)
        z_log = array_api.log(z_sum)
        z_ans = z_log + maxZ.reshape(z_log.shape)
        return z_ans
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        maxz = z.realize_cached_data().max(axis=self.axes, keepdims=True)
        zexp = exp(z - maxz)
        zsumexp = summation(zexp, self.axes)
        grad_div_zse = out_grad / zsumexp
        grad_div_zse_b = grad_div_zse.reshape(maxz.shape).broadcast_to(z.shape)
        return grad_div_zse_b * zexp


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
