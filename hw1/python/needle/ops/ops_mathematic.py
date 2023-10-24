"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * self.scalar * mul_scalar(node.inputs[0], self.scalar - 1)
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a_d = divide(out_grad, node.inputs[1])
        b_d = negate(multiply(out_grad, divide(
            node.inputs[0], multiply(node.inputs[1], node.inputs[1]))))
        return a_d, b_d
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar

        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        d = a.ndim
        axis = [i for i in range(d)]
        x, y = (
            d - 1, d - 2) if self.axes is None else (self.axes[0], self.axes[1])
        axis[x], axis[y] = axis[y], axis[x]
        self.t = (x, y)
        return array_api.transpose(a, axes=axis)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return transpose(out_grad, self.t)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION

        return numpy.reshape(a, self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, shape=self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        # END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     # BEGIN YOUR SOLUTION
    #     a1 = matmul(out_grad, transpose(node.inputs[1]))
    #     a2 = matmul(transpose(node.inputs[0]), out_grad)
    #     return a1, a2
    #     # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(
            lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(
                tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(
                tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
    # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.negative(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return negate(out_grad)
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.realize_cached_data().copy()
        data[data > 0] = 1
        return multiply(out_grad, Tensor(data))
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
