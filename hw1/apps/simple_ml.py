"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    # 读取数据
    with gzip.open(image_filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(-1, 28 * 28)
    data = data.astype('float32')
    with gzip.open(label_filename, 'rb') as f:
        label = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    # 归一化,就是简单的映射
    data = data / 255
    return data, label
    # END YOUR CODE


# def parse_mnist(image_filesname, label_filename):
#     """Read an images and labels file in MNIST format.  See this page:
#     http://yann.lecun.com/exdb/mnist/ for a description of the file format.

#     Args:
#         image_filename (str): name of gzipped images file in MNIST format
#         label_filename (str): name of gzipped labels file in MNIST format

#     Returns:
#         Tuple (X,y):
#             X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
#                 data.  The dimensionality of the data should be
#                 (num_examples x input_dim) where 'input_dim' is the full
#                 dimension of the data, e.g., since MNIST images are 28x28, it
#                 will be 784.  Values should be of type np.float32, and the data
#                 should be normalized to have a minimum value of 0.0 and a
#                 maximum value of 1.0.

#             y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
#                 labels of the examples.  Values should be of type np.int8 and
#                 for MNIST will contain the values 0-9.
#     """
#     # BEGIN YOUR SOLUTION
#     raise NotImplementedError()
#     # END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    # BEGIN YOUR SOLUTION
    e_z = ndl.exp(Z)
    e_s = ndl.summation(e_z, (1,))
    e_l = ndl.log(e_s)
    e_s = ndl.summation(e_l)
    y_s = ndl.summation(y_one_hot * Z)
    return (e_s - y_s) / Z.shape[0]
    # END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    # BEGIN YOUR SOLUTION
    batch_num = X.shape[0] // batch
    Y = gethot(y, W2.shape[1])
    for i in range(batch_num):
        x_d = ndl.Tensor(X[i * batch:(i + 1) * batch])
        y_d = ndl.Tensor(Y[i * batch:(i + 1) * batch])
        x_w1 = ndl.matmul(x_d, W1)
        th_x_w1 = ndl.relu(x_w1)
        o = ndl.matmul(th_x_w1, W2)
        loss = softmax_loss(o, y_d)
        loss.backward()
        W1 = ndl.Tensor(W1.realize_cached_data() - lr *
                        W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr *
                        W2.grad.realize_cached_data())
    return W1, W2
    # END YOUR SOLUTION


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT
def gethot(y, n):
    one_hot = np.eye(n)[y]
    return one_hot


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
