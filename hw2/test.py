import sys
sys.path.append("./python")
sys.path.append("./apps")
from apps.mlp_resnet import MLPResNet, epoch, train_mnist


import numpy as np
import needle as ndl
import needle.nn as nn


from mlp_resnet import *


def test_mlp_train_epoch_1():
    np.testing.assert_allclose(
        train_epoch_1(5, 250, ndl.optim.Adam, lr=0.01, weight_decay=0.1),
        np.array([0.675267, 1.84043]),
        rtol=0.0001,
        atol=0.0001,
        )


def train_epoch_1(hidden_dim, batch_size, optimizer, **kwargs):
    np.random.seed(1)
    train_dataset = ndl.data.MNISTDataset(
        "./data/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
        )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), **kwargs)
    model.eval()
    return np.array(epoch(train_dataloader, model, opt))

# test_mlp_train_epoch_1()


if __name__ == '__main__':
    a = ndl.Tensor([1, 2, 3])
    b = ndl.Tensor([5, 6, 7])
    c = a + b
