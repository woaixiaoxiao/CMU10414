import numpy as np
import sys
import numdifftools as nd
sys.path.append("./src")
import mugrade
from simple_ml import *
try:
    from simple_ml_ext import *
except:
    pass
import struct
import gzip
import idx2numpy


def parse_mnist(image_filename, label_filename):
    # 读取数据
    with gzip.open(image_filename,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data=data.reshape(-1,28*28)
    data=data.astype('float32')
    with gzip.open(label_filename,'rb') as f:
        label = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    # 归一化,就是简单的映射
    data=data/255
    return data,label

def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz","data/train-labels-idx1-ubyte.gz")
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

def main():
    # parse_mnist("data/train-images-idx3-ubyte.gz",
    #                   "data/train-labels-idx1-ubyte.gz")
    test_parse_mnist()

if __name__ == '__main__':
    main()