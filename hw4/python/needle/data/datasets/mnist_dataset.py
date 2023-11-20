import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


def parse_mnist(image_filename, label_filename):
    # BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
        file_content = f.read()
        # >I 代表大端无符号整数
        num = struct.unpack('>I', file_content[4:8])[0]
        # 第一个参数代表要读出的格式
        # 第二个参数代表要读的东西
        X = np.array(struct.unpack(
            'B' * 784 * num, file_content[16:16 + 784 * num]
            ), dtype=np.float32)
        X.resize((num, 784))
    with gzip.open(label_filename, 'rb') as f:
        file_content = f.read()
        num = struct.unpack('>I', file_content[4: 8])[0]
        y = np.array([struct.unpack('B', file_content[8 + i:9 + i])[0]
                     for i in range(num)], dtype=np.uint8)

    X = X / 255.0
    return X, y
    # END YOUR CODE


class MNISTDataset(Dataset):
    def __init__(
            self,
            image_filename: str,
            label_filename: str,
            transforms: Optional[List] = None,
            ):
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(
            image_filename=image_filename,
            label_filename=label_filename
            )
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        img = self.images[index]
        if len(img.shape) > 1:
            img = np.array([self.apply_transforms(
                i.reshape(28, 28, 1)).reshape(28 * 28) for i in img])
        else:
            img = self.apply_transforms(
                img.reshape(28, 28, 1)).reshape(28 * 28)
        label = self.labels[index]
        return (img, label)

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        # END YOUR SOLUTION
