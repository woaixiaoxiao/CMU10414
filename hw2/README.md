# Question 1

这部分是完成一些初始化的操作，我们会用到`init_basic`中已经准备好的函数，这部分主要是对numpy的操作进行了一些封装，可以指定设备为cpu还是gpu，可以指定数据类型，可以指定是否需要梯度，并且返回的是Tensor

## xavier_uniform

根据公式来就行，注意传参的方式，因为给我们提供的函数签名是这样`rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False)`，`*s`代表我们可以传入很多个变量，都会归到s中，正确的调用方式如下

```c++
def xavier_uniform(fan_in, fan_out, gain=1.0, kwargs):
    # BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    print("hello  ", a)
    return rand(fan_in, fan_out, low=-a, high=a, kwargs)
    # END YOUR SOLUTION
```

我一开始使用`rand((fan_in, fan_out), -a, a, kwargs)`，这样是错误的，Python不会自动将元组翻译为s，只能一个参数一个参数地传。

而因为rand函数第一个参数就是`*s`，导致后面就没法正常传入-a和a，必须制定`low=-a,high=a`，至于最后一个参数代表的是多个字典，因此直接`kwargs`传入即可

## others

通过第一个熟悉语法基础之后，其他的就是翻译一下公式

```c++
def xavier_normal(fan_in, fan_out, gain=1.0, kwargs):
    # BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std=std, kwargs)
    # END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, kwargs)
    # END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # BEGIN YOUR SOLUTION
    std = math.sqrt(2) * math.sqrt(1 / fan_in)
    return randn(fan_in, fan_out, std=std, kwargs)
    # END YOUR SOLUTION
```

# Question 2

## Linear

### init

直接调用之前写好的初始化函数即可，这里要手动传入device和dtype等信息

关键是bias函数需要自己手动reshape一下

```c++
self.weight = init.kaiming_uniform(
    in_features, out_features, device=device, dtype=dtype)
self.bias = init.kaiming_uniform(
    out_features, 1, device=device, dtype=dtype).reshape((1, out_features))

```

### forward

直接翻译公式即可

关键是要手动将bias广播成最终的shape，可以在乘法结束之后取出shape，这样省的自己拼拼凑凑了

```c++
def forward(self, X: Tensor) -> Tensor:
# BEGIN YOUR SOLUTION
	xat = X.matmul(self.weight)
    shape = xat.shape
    bb = self.bias.broadcast_to(shape)
    return ops.add(xat, bb)
    # END YOUR SOLUTION
```

## relu

### forward

这里好像是说之后反向传播时，直接将relu在0处的导数看为0

```c++
def forward(self, x: Tensor) -> Tensor:
# BEGIN YOUR SOLUTION
	return ops.relu(x)
    # END YOUR SOLUTION
```

## sequential

### forward

定义了一个批量化操作，到时候传入一堆模块，就可以按顺序执行这些模块了

```c++
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        c = x
        for m in self.modules:
            c = m(c)
        return c
        # END YOUR SOLUTION
```

## LogSumExp

这个算子就有点复杂了，特别是在反向传播的时候

不过反向传播的求导公式化简之后，普通的$z_i$和$maxz$的表达式是一样的，即
$$
\frac{e^{z_i-maxz}}{ze^{z_i-maxz}}
$$

### forward

前向传播只需要翻译公式，和hw1里的方法差不多，就是要注意维度的问题

```c++
    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        maxZ = array_api.amax(Z, axis=self.axes, keepdims=True)
        z_exp_minus = array_api.exp(Z - maxZ)
        z_sum = array_api.sum(z_exp_minus, axis=self.axes)
        z_log = array_api.log(z_sum)
        z_ans = z_log + maxZ.reshape(z_log.shape)
        return z_ans
        # END YOUR SOLUTION
```

### backward

求导公式在前面已经给出

1. 我们在backward的时候，要使用tensor，不能使用numpy的数组

2. 而目前还没有实现tensor的max操作，所以需要先将数据提取出来进行max，之后将maxz作为一个标量和tensor操作一下，结果就是tensor了

在这里我突然有点不理解为什么forward的时候，为什么返回的numpy数组，却可以生成tensor。调试了一下发现，我忘记了这些操作的类都是从TensorOp中继承的，而这个类将它的`__call__`方法设置为`make_from_op`。而在`make_from_op`中，它先初始化一个Tensor，然后通过`realize_cached_data`调用了`compute`方法，才真正的给这个tensor赋值。也就是说，forward只是构建一个tensor里的一小步，这里面提供了很多封装。

```c++
    def gradient(self, out_grad, node):
        z = node.inputs[0]
        maxz = z.realize_cached_data().max(self.axes, keepdims=True)
        zexp = exp(z - maxz)
        zsumexp = summation(zexp, self.axes)
        grad_div_zse = out_grad / zsumexp
        grad_div_zse_b = grad_div_zse.reshape(maxz.shape).broadcast_to(z.shape)
        return grad_div_zse_b * zexp
```

这里的坑点就是不能使用array_api的函数，要使用自己写的继承自TensorOp的函数，这样才可以生成并返回Tensor

## SoftmaxLoss

### forward

logits：$[m,k]$，y：$[m,]$

```c++
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        pre = ops.summation(ops.logsumexp(logits, axes=(1)))
        oh = init.one_hot(logits.shape[1], y)
        lat = ops.summation(ops.multiply(oh, logits))
        return (pre - lat) / logits.shape[0]
        # END YOUR SOLUTION
```

1. 调用之前写的加强版的logsumexp计算减号前面的之和
2. 调用init.one_hot生成one_hot矩阵，这里传入的y需要是tensor，因为在one_hot函数中，调用了y.numpy
3. 通过逐元素相乘然后求和，得到减号后面的
4. 最后记得除以样本数量，得到平均值

## LayerNorm1d

翻译公式即可，不过这里面形状的变化有点复杂

首先，我们的参数是需要自己自定义的，观察公式就可以知道，weight和bias就是一个形状为$(n,)$的向量，初始化时一个为全1，一个为全0

其次，我们的输入一定是一个二维的tensor，因此，在后面各种操作时，我们可以轻松将tensor给reshape成想要的形状，注意点

1. tensor是没有自动的广播功能的，因此，我们最好在操作之前，手动广播维度
2. 低维的矩阵，比如形状为$[a,b]$，可以直接广播成$[c,a,b]$

```c++
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        # self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        mean = (x.sum((1,)) /
                x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x - mean)2).sum((1,)) /
               x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        deno = (var + self.eps)0.5
        return self.weight.broadcast_to(x.shape) * (x - mean) / deno + self.bias.broadcast_to(x.shape)
        # END YOUR SOLUTION
```

## Flatten

这个简单，计算出维度之后直接reshape就行

```c++
class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        s = X.shape
        x = s[0]
        y = 1
        for i in s[1:]:
            y *= i
        return X.reshape((x, y))
        # END YOUR SOLUTION
```

## BatchNorm1d

这个归一化的操作很神奇，是对在同一个特征位置上的值进行归一化，而不是在同一个样本内部进行归一化，因此像sum这种操作的轴都是0，也就是列

并且可以在训练的时候不断地学习，最后在测试的时候使用统计好的数据

这里给的公式有一点歧义

![image-20231031200321862](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231031200321862.png)

上面是训练的，没啥问题，下面是测试的，这里直接就是y等于这一堆，这就有点问题，这只是对数据进行了归一化操作，因此，还需要乘w加b之后才对。并且上面的$mu$其实就是之前统计的平均数，下方的theta的平方，就是统计好的方差。

最后，这里要用$self.training$来判断当前是在训练还是在测试

```c++
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training:
            # ex = (x.sum((0,)) / x.shape[0]).broadcast_to(x.shape)
            # vx = (((x - ex)2).sum((0,)) /
            #       x.shape[0]).broadcast_to(x.shape)
            ex = (x.sum((0,)) / x.shape[0])
            vx = (((x - ex.broadcast_to(x.shape))2).sum((0,)) / x.shape[0])
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                self.momentum * ex
            self.running_var = (1 - self.momentum) * \
                self.running_var + self.momentum * vx
            norm = (x - ex.broadcast_to(x.shape)) / \
                ((vx.broadcast_to(x.shape) + self.eps)0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / (
                (self.running_var.broadcast_to(x.shape) + self.eps)(0.5))
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        # END YOUR SOLUTION
```

注意坑

1. 要用到axis的操作，最好是用元组传递参数，不要用整数，比如sum操作。因为我们在这里默认是认为axis可以迭代，即可以for循环访问的，不像numpy原生的操作

## Dropout

dropout的实现就是生成一个二进制数组，其中1的比例就是p

框架已经给我们提供了一种初始化方法`randb`，直接调用就行了，唯一的问题是题目传入的p代表的是0的比例，因此用1-p作为参数传入

最后要记得将数组放大，这样才能保证均值和方差不变

```c++
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        # END YOUR SOLUTION
```

## Residual

```c++
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return self.fn(x) + x
        # END YOUR SOLUTION
```

# Question 3

## SGD

主体部分翻译公式即可，不过这里要注意

1. `self.u`是一个字典，以parameter为键，以对应的被动量修正过的梯度为值

2. 这里只会更新Parameter类型的变量，所以之前各种模块里的要训练的参数都要声明为parameter，并且梯度需要是可更新的，就像这样

    ```c++
    self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
    ```

3. 这里多给了一个`weight_decay`，这个参数是用来正则化的，使用的是L2范数。具体在使用上，使用它和矩阵的变量相乘（在损失函数中是$1/2*w^2$，求导后就是$w$，这个weight_decay应该相当于一个系数。有点问题的就是，这里应该要用w的绝对值才准确，但是我没加绝对值也过了），然后加上梯度上即可。

```c++
    def step(self):
        # BEGIN YOUR SOLUTION
        for parm in self.params:
            if self.weight_decay > 0:
                grad = parm.grad.data + self.weight_decay * parm.data
            else:
                grad = parm.grad.data
            self.u[parm] = self.momentum * \
                self.u[parm] + (1 - self.momentum) * grad
            parm.data = parm.data - self.lr * self.u[parm]
        # END YOUR SOLUTION
```

最后这里有个奇怪的点就是，我这里一直报错显示有一部分数据是float64，在`parm.data = parm.data - self.lr * self.u[parm]`类型检查一直出错。后来检查到是`SoftmaxLoss`模块的问题，在这里计算loss时，我是先计算减号前的和，以及减号后的和，相减之后再除以样本数量，这样就会出错。而如果在求和之前先除样本数就不会错。问了gpt之后说是Python在溢出的时候会默认提升位数，估计是直接加起来太大了，导致从float32变成float64

# Adam

依旧是翻译公式，有以下注意点

1. `selr.t`每调用一次step就+1，目前还不是很懂这个原理是啥，应该是用来调节每次的梯度对u和v的影响程度的

```c++
    def step(self):
        # BEGIN YOUR SOLUTION
        self.t += 1
        for parm in self.params:
            if self.weight_decay > 0:
                grad = parm.grad.data + self.weight_decay * parm.data
            else:
                grad = parm.grad.data
            self.m[parm] = self.beta1 * self.m[parm] + (1 - self.beta1) * grad
            self.v[parm] = self.beta2 * self.v[parm] + \
                (1 - self.beta2) * (grad * grad)
            bm = self.m[parm] / (1 - self.beta1self.t)
            bv = self.v[parm] / (1 - self.beta2self.t)
            parm.data = parm.data - self.lr * bm / (bv(1 / 2) + self.eps)
        # END YOUR SOLUTION
```

这里有个坑，就是会有一个内存的检测，要求不能出现多余的tensor

也就是说，除了构建计算图之后，能用原生的data就用。我之前在`BatchNorm1d`中计算`self.running_mean`时就直接用的Tensor，造成了内存的问题。

# Question 4

## Transformations

### RandomFlipHorizontal

将数组的第二个维度翻转即可，~~具体为啥应该不是重点~~

```c++
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
```

### RandomCrop

随机在上下左右四个方向移动图片，用零填充

实现思路是先在四个方向填充很多0，再根据随机生成的偏移来对数组切片即可

```c++
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2)
        # BEGIN YOUR SOLUTION
        img_pad = np.pad(img, [(self.padding, self.padding),
                         (self.padding, self.padding), (0, 0)])
        h, w, _ = img_pad.shape
        return img_pad[self.padding + shift_x:h - self.padding + shift_x, self.padding + shift_y:w - self.padding + shift_y, :]
        # END YOUR SOLUTION

```

## Dataset

dataset类要完成的任务是根据index给出数据，其中index可能是整数，也可能是其他类型的索引，比如切片，比如列表(Python内置的列表不支持用列表访问列表，但是numpy的数组是支持的，而我们这里的images是用numpy的数组存储的，所以需要考虑到)。因此，`self.images[index]`可能是一个列表，其中有多个image

同时，我们这里可能需要进行transforms的操作，Dataset类已经给出了`apply_transforms`函数，我们只需要调用即可。不过调用的时候要注意，我们之前实现的各种transform的操作都是在(H,W,C)中的H和W上进行的，所以将数据放入`apply_transforms`之前，要将数据reshape成`(28, 28, 1)`，同时，在操作结束之后，将图片回复正常形状

最后，返回数据和标签组成的元组

```c++
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

```

这里参考大佬的代码换了个解析图片文件的函数，解析一个图片的思路如下

1. 首先得清楚这个图片文件的格式，通常在文件开始处有一些元信息，真正读取数据时已经跳过
2. 首先，打开文件并读入文件的所有内容
3. 读取文件的元信息，比如这个文件的数据有多长，或者文件包含多少个图片。可以通过`struct.unpack`函数完成这个操作，这个函数的第一个参数是读取数据的格式，第二个参数是需要读取的文件的区域
4. 读取文件的数据，按自己想要的格式存储起来
5. 对图片数据进行归一化（需要的话）

感觉读图片文件这种操作，有点繁琐且没意思，知道思路之后~~能用现成的就用现成的吧~~

## Dataloader

Dataloader这玩意提供了一层抽象，可以实现如下附加功能

1. 以batch的大小读取数据
2. 决定是否要打乱

其中以batch的大小读取数据可以轻松实现，因为在dataset中已经支持批量读取数据了

而打乱的操作可以用如下思路实现

1. 假设现在有n个数据
2. 生成0到n-1的全排列
3. 以batch为大小将全排列进行划分，得到m个列表
4. 这m个列表就是m个batch，直接用列表去访问dataset即可，numpy数组可以正确处理这里访问方式

最后，通过dataloader返回的数据和标签，都是tensor类型，因为直接放到神经网络中去训练了，这里总结一下dataloader返回的数据的格式，它返回的是一个元组，元组的第一个是数据，第二个标签

1. 数据的格式$(m,H,W,C)$，所以通过下标i就可以拿到第i个数据
2. 标签的格式是$(m,)$，通过下标i就可以拿到第i个标签

这里还涉及到了一些Python的语法，以及运算符的重载

```c++
class DataLoader:
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
            self,
            dataset: Dataset,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
            ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # BEGIN YOUR SOLUTION
        self.index = -1
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        # END YOUR SOLUTION
        return self

    def __next__(self):
        # BEGIN YOUR SOLUTION
        self.index += 1
        if self.index >= len(self.ordering):
            raise StopIteration
        samples = self.dataset[self.ordering[self.index]]
        return Tensor(samples[0]), Tensor(samples[1])
        # END YOUR SOLUTION

```

# Question 5

## ResidualBlock

直接照着图翻译即可，先构造出残差网络的主函数，然后再构建残差网络，最后加上一个relu

```c++
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(
        ), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    # END YOUR SOLUTION
```

## MLPResNet

继续翻译图

不过这里有个玄学，那numblocks个残差网络，如果我先定义好，然后在拿到这里面来，最后就会报错，或者我在这个函数里多定义其他的变量，但是不放到res中，也会报错，明明新定义的模块都没有使用，目前还不清楚为啥

```c++
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
```

## epoch

epoch就是一次迭代需要做的事情，那就需要使用模型，损失函数，优化器

如果现在是测试模式，那连优化器都不用了，只需要使用模型得到输出，计算以下损失和准确率即可

这里我们根据opt判断是训练或者还是测试模式，可以通过`model.eval`或者`model.train`设置这些模式，本质上在这两个函数中，就是将这模块的所有组成部分的training属性设置为false或者true。通过traning可以控制一些模块在forward时的表现，比如那个batch-norm

要注意的是，这个模式和是否需要进行梯度下降并不等价，这应该是单纯控制forward时的作用

在计算损失的时候，不能直接用tensor来计算了，要用tensor里真实的data

```c++
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
```

## Train Mnist

这其实就是最终用户使用的接口，自己定义模型，优化器，损失函数，得到数据集

其中有个细节就是在定义优化器的时候，我们传入了模型的所有需要更新的参数，这才是决定梯度下降时到底要更新哪些参数的位置

还有个有意思的点，在使用l2惩罚项的时候，看起来是在损失函数上加上了这个惩罚项，而在真正实现的时候，是在优化器中定义weight_decay，这应该也是为了节省资源的一种做法，可以省去很多不必要的计算

```c++
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
```










































