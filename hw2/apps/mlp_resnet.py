import sys
sys.path.append("../python")
from needle.data.data_basic import DataLoader

from needle.data.datasets.mnist_dataset import MNISTDataset


import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(
        ), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    # END YOUR SOLUTION


def MLPResNet(
        dim,
        hidden_dim=100,
        num_blocks=3,
        num_classes=10,
        norm=nn.BatchNorm1d,
        drop_prob=0.1,
        ):
    # BEGIN YOUR SOLUTION
    res = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2,
                                        norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                        nn.Linear(hidden_dim, num_classes))

    return res


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    tot_loss, tot_err = [], 0.0
    loss_fc = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in dataloader:
            logit = model(X)
            tot_loss.append(loss_fc(logit, y).numpy())
            tot_err += np.sum(logit.numpy().argmax(axis=1) != y.numpy())
    else:
        model.train()
        for X, y in dataloader:
            logit = model(X)
            loss = loss_fc(logit, y)
            tot_loss.append(loss.numpy())
            tot_err += np.sum(logit.numpy().argmax(axis=1) != y.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    return tot_err / len(dataloader.dataset), np.mean(tot_loss)
    # END YOUR SOLUTION


def train_mnist(
        batch_size=100,
        epochs=10,
        optimizer=ndl.optim.Adam,
        lr=0.001,
        weight_decay=0.001,
        hidden_dim=100,
        data_dir="data",
        ):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt=opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
