# 测试

-m 指定测试的程序为pytest

-k 指定名字中包含add的测试

```python
!python3 -m pytest -k "add"
```

-s 可以让print正常打印

通过下面指令可以方便地让jupyter编程Markdown

```python
jupyter nbconvert --to markdown faker.ipynb
```



# parse_mnist

函数的两个参数已经包含了文件夹，因此可以直接作为路径进行open

整体思路

1. 解压文件
2. 通过numpy从文件中读数据
3. 修改数据的格式和类型
4. 归一化

具体操作

1. 首先要用gzip将gz文件解压缩

    `rb`的意思是只读字节流的形式打开。对图像，音频等数据，用这种方式打开最合适

2. `data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)`

    1. dtype=uint8就够了，因为图像的每个像素值都是0-255，指定这个类型之后，会将f文件的每个字节都翻译成一个uint8类型的变量
    2. offset=16是这个数据集特有的特点，即前16个元数据的字节直接跳过即可，这个值应该是每个数据集都可能不一样

3. 将data的形状修改并修改数据类型，刚读入的形状是一个一维数组

4. 归一化操作很简单，将0-255映射到0-1即可

```python
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
```

# softmax_loss

使用numpy实现softmax交叉熵损失，关键在于numpy的并行化操作，广播特性，操作轴axis等

这是一维输入时的公式，其中每个$z_i$都是这个样本标签为i的概率
$$
\begin{equation}

\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.

\end{equation}
$$
实现要求的是二维输入的情况，思路如下

1. 对整个二维数组求exp指数
2. 按行对指数结果求和
3. 对求和的结果取对数
4. 每个样本都要减去对应的$z_i$

```python
def softmax_loss(Z, y):
    exp_Z = np.exp(Z)
    exp_S = np.sum(exp_Z,axis=1)
    exp_L = np.log(exp_S)
    y_Z = Z[np.arange(Z.shape[0]),y]
    return np.mean(exp_L-y_Z)
```

# SGD

更新公式
$$
\theta:=\theta-\frac\alpha BX^T(Z-I_y)
$$
其中$Z=\operatorname{softmax}(X\theta)$

而$softmax$操作就是对每一行normalize，即对每个标签i来说
$$
\begin{aligned}-1\{i=y\}+\frac{\exp h_i}{\sum_{j=1}^k\exp h_j}\end{aligned}
$$
而$I_y$是生成一个one_hot向量，只有y处为1

具体实现如下

```c++
def normalize_(x):
    ex=np.exp(x)
    ex_rowsum=ex.sum(axis=1)
    ex_n=ex/ex_rowsum[:,np.newaxis]
    return ex_n
def gethot(y,n):
    one_hot=np.eye(n)[y]
    return one_hot

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    batch_num = X.shape[0] // batch
    for i in range(batch_num):
        x_d = X[i*batch:(i+1)*batch] // m*n
        y_d = y[i*batch:(i+1)*batch] // m
        o = np.dot(x_d,theta) // m*k
        z = normalize_(o) // m*k
        oh = gethot(y_d,theta.shape[1]) // m*k
        # print(x_d.T.shape,z.shape,oh.shape)
        theta-=lr/batch*(np.dot(x_d.T,z-oh))
```

# nn_epoch

和上题一样，翻译公式即可

$W_2$的梯度
$$
\begin{aligned}\nabla_{W_2}\ell_{ce}(\sigma(XW_1)W_2,y)&=\sigma(XW_1)^T(S-I_y)\end{aligned}
$$
$W_1$的梯度
$$
\nabla_{W_1}\ell_{ce}(\sigma(XW_1)W_2,y)=X^T\left((S-I_y)W_2^T\circ\sigma^{\prime}(XW_1)\right)
$$
其中$S$为
$$
S=\text{ softmax}(\sigma(XW_1)W_2)
$$

```c++
def active(x):
    return np.maximum(0,x)
def deactivate(x):
    return np.where(x<=0,0,1)
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    ### BEGIN YOUR CODE
    batch_num = X.shape[0] // batch
    for i in range(batch_num):
        x_d = X[i*batch:(i+1)*batch]
        y_d = y[i*batch:(i+1)*batch]
        x_w1 = np.dot(x_d,W1)
        th_x_w1 = active(x_w1)
        o = np.dot(th_x_w1,W2)
        s = normalize_(o)
        iy = gethot(y_d,W2.shape[1])
        d_w2 = np.dot(active(x_w1).T,s-iy)
        d_w1 = np.dot(x_d.T,np.multiply(np.dot(s-iy,W2.T),deactivate(x_w1)))
        W2-=lr/batch*d_w2
        W1-=lr/batch*d_w1
```

## softmax_regression_epoch_cpp

一行一行翻译python代码就行了，关键在于矩阵操作

```c++
void matMul(const float *x, float *theta, float *o, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            o[i * k + j] = 0;
            for (int t = 0; t < n; t++) {
                // x: i t   theta: t j   o:i j
                // x: m n   theta: n k   o:m k
                o[i * k + j] += x[i * n + t] * theta[t * k + j];
            }
        }
    }
}
void sigmoid(float *mat, int m, int k) {
    for (int i = 0; i < m; i++) {
        float temp = 0;
        for (int j = 0; j < k; j++) {
            mat[i * k + j] = std::exp(mat[i * k + j]);
            temp += mat[i * k + j];
        }
        for (int j = 0; j < k; j++) {
            mat[i * k + j] /= temp;
        }
    }
}
void hotDeal(float *mat, int m, int k, const unsigned char *y) {
    for (int i = 0; i < m; i++) {
        char label = y[i];
        mat[i * k + label] -= 1;
    }
}
// x:m*n  o:m*k
// x的转置n*m o:m*k
// dg n*k
void calDg(const float *x, float *o, float *dg, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = i * k + j;
            dg[index] = 0;
            for (int t = 0; t < m; t++) {
                dg[index] += x[t * n + i] * o[t * k + j];
            }
        }
    }
}
void updateTheta(float *theta, float *dg, int n, int k, float lr, float batch) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            theta[i * k + j] -= lr / batch * dg[i * k + j];
        }
    }
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    int batch_num = m / batch;
    float *o = (float *)malloc(sizeof(float) * m * k);
    float *dg = (float *)malloc(sizeof(float) * n * k);
    for (int i = 0; i < batch_num; i++) {
        const float *x_d = X + n * batch * i;
        const unsigned char *y_d = y + batch * i;
        matMul(x_d, theta, o, batch, n, k);
        sigmoid(o, batch, k);
        hotDeal(o, batch, k, y_d);
        // theta-=lr/batch*(np.dot(x_d.T,z-oh))
        // theta-=lr/batch*(x_d.T,o)
        calDg(x_d, o, dg, batch, n, k);
        updateTheta(theta, dg, n, k, lr, batch);
    }
}
```




