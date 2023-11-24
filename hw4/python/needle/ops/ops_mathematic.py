"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


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
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a**self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0]**(self.scalar - 1))
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a_d = out_grad / node.inputs[1]
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
        return a.permute(axis)
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
        return a.compact().reshape(self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape).compact()
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
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        b = array_api.sum(a, axis=self.axes)
        return b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError(
                "Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a @ b
        # END YOUR SOLUTION

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
        return -a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return -out_grad
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.log()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.exp()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0].exp())
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0])**2)
        # END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        # BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        # END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i + 1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        # END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        # END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)
        arr = a.device.full(new_shape, 0)
        slices = [slice(0, n) for n in arr.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, arr.shape[axis], self.dilation + 1)
        arr[tuple(slices)] = a
        return arr
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return UnDilate(self.axes, self.dilation)(out_grad)
        # END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        slices = [slice(0, n) for n in a.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)].compact()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding),
                  (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, K_, C_in_, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        assert K == K_, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"

        inner_dim = K * K * C_in
        out_H, out_W = (H - K + 1) // self.stride, (W - K + 1) // self.stride
        im2col = A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                              strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))\
            .compact()\
            .reshape((N * out_H * out_W, inner_dim))
        out = im2col @ B.compact().reshape((K * K_ * C_in_, C_out))
        return out.compact().reshape((N, out_H, out_W, C_out))
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        W_permute = transpose(flip(W, (0, 1)), (2, 3))  # K * K * C_out * C_in
        # out_grad: # N * (H+2P-K+1) * (W+2P-K+1) * C_out
        X_grad = conv(out_grad, W_permute, padding=K - 1 - self.padding)

        X_permute = transpose(X, (0, 3))  # C_in * H * W * N
        # (H+2P-K+1) * (W+2P-K+1) * N * C_out
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # C_in * H * W * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding)
        W_grad = transpose(transpose(W_grad, (0, 1)),
                           (1, 2))  # H * W * C_in * C_out

        return X_grad, W_grad
        # END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
