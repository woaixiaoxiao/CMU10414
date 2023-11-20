import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


def parse_CIFAR10(base_folder: str, train: bool):
    # 获得要读取的所有文件的路径
    data_paths = []
    if train:
        for i in range(1, 6):
            data_paths.append(f'data_batch_{i}')
    else:
        data_paths.append('test_batch')
    # 读取数据
    data = []
    label = []
    for data_path in data_paths:
        with open(os.path.join(base_folder, data_path), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            data.append(data_dict[b'data'])
            label.append(data_dict[b'labels'])
    # 数据预处理：先去除多余的维度，再归一化和调整形状
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=None)
    data = data / 255
    data = data.reshape((-1, 3, 32, 32))
    # 返回结果
    return np.array(data), np.array(label)


class CIFAR10Dataset(Dataset):
    def __init__(
            self,
            base_folder: str,
            train: bool,
            p: Optional[int] = 0.5,
            transforms: Optional[List] = None
            ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        # BEGIN YOUR SOLUTION
        self.X, self.Y = parse_CIFAR10(base_folder, train)
        self.transforms = transforms
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        # BEGIN YOUR SOLUTION
        image, label = self.X[index], self.Y[index]
        if self.transforms:
            image = np.array([self.apply_transforms(i) for i in image])
        return (image, label)
        # END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        # BEGIN YOUR SOLUTION
        return len(self.Y)
        # END YOUR SOLUTION
