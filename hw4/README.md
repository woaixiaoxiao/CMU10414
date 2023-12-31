# Part 1: ND Backend

这一部分是要在NDarray的基础上，重新实现之前的一系列ops

区别就在于之前我们是以numpy的数组来支持的，现在需要改为自己抽象的ndarray。因此，之前在array_api使用的一些函数，需要检查在自己的NDarray中是否提供，以及各种参数是否和numpy的一致

这里面文档还是有些坑没说的

1. 文档直接让复制autograd函数，但是如果直接复制hw2的autograd函数，arrayapi依然是numpy，所以这一行还是要保留

    ```python
    from .backend_selection import Device, array_api, NDArray, default_device
    ```

2. 在hw3中实现的求和函数中，只允许指定一个轴，并且必须指定一个轴，但是在hw1中实现各种ops时，numpy的sum是允许对多个轴进行求和的，因此这里也需要修改hw3的求和函数

3. 在hw4中使用到了autograd中的TensorTuple，这个类的detach函数中调用make_const时，需要用`TensorTuple`，而项目给的代码是`Tuple`

可能还有一些坑不记得了

最后，tanh这个ops一直过不去精度，不太清楚是什么原因

除去这些坑，在具体的实现层面

1. 在修改形状，或者以任何形式修改元素的组织方式时，一定要提前调用`compact`函数
2. 要交换轴的顺序，比如转置时，要通过permute函数实现
3. numpy的广播机制是支持，将(5,)直接广播成(n,5)的，但是在我们的permute中，是不支持的，所以需要先reshape再broadcast
4. 新增加的stack和split，是为了用来取出tensor中的第n项，因为对于tensor是没有办法直接通过下标访问的
    1. 这时候通过对指定的轴split，就可以得到一个列表，列表中的每一项就是想访问的tensor
    2. 同时，对一个列表在指定维度stack，可以将列表转为一整个tensor


整体来说，这个part体现了一个思想，写代码时，除了保证一个模块的功能是正确的，还要保证不同模块之间的对接是正确的，否则的话照样有bug。

# Part 2: CIFAR-10 dataset

又出bug了，说cpudevice没有array函数，怀疑是因为colab的gpu环境时长用完了，导致make阶段没有成功生成cpython文件

这里的思路

1. 在init函数中，将数据给读到numpy数组中
    1. 文件里面数据的格式是字典，键分别是data和label。其中训练数据有5个文件，需要手动将它们都读取
    2. 最终读取的效果，data的形状是(m,单个数据的形状)，label的形状就是(m,)
2. 在getitem函数中，按照索引取值，一般索引是切片的形式，因此会一般会读出多个数据，就算不是多个数据，也会在单个数据的形状前多出一个维度。这里最好返回numpy数组

`AttributeError: 'CPUDevice' object has no attribute 'Array'`未解决

# Part 3: Convolutional neural network [40 points]

## Padding ndarrays

输出一个被padding的NDarray对象

1. 首先根据padding的参数和原来的shape构造出new_shape
2. 构造切片的数组，用于后面来将原来的数组复制到新数组的指定位置
3. 赋值

这里面赋值是通过`__setitem__`函数实现，这个函数先通过`__getitem__`函数获取一个不compact的数组对象，然后对它进行赋值

## Flipping ndarrays & FlipOp

翻转某个维度只需要修改stride和offset，对于要翻转的维度i

1. `new_offset += (self.shape[i] - 1) * new_stride[i]`
2. `new_stride[i] *= -1`

在make之后马上调用compact就不会在后面的各种操作中出错。同时在调用flip之前，最好也是compact的，否则上面这个公式感觉会有问题

## Dilation

传入参数中，axes代表要在哪些轴上膨胀，dilation表示膨胀多少格，如果是1，代表增加一个空格

1. 在forward操作中
    1. 先求出new_shape
    2. 构造出新的数组
    3. 构造切片
    4. 赋值
2. 在backward操作中
    1. 构造切片
    2. 直接返回

## Convolution forward

将卷积操作转为乘法操作，假设矩阵为A，卷积核为B

1. 先取出A和B的各种维度，包括n，h，w，c_in，k，c_out
2. 将A转为矩阵乘法需要的样子
    1. 先通过调整shape和stride，转为形状`(N, H - K + 1, W - K + 1, K, K, C_in)`，对应的stride为`(Ns, Hs, Ws, Hs, Ws, Cs))`、
    2. 先compact，再reshape成`(N, H - K + 1, W - K + 1, -1)`。如果不先compact，就会出错
3. 将B转为乘法需要的样子，即`(-1, C_out)`
4. A和B相乘得到结果out

如果要加上padding操作，则在第1步之前，调用pad操作，将h和w维度扩展

如果要加上stride，那在得到out之后，在h和w维度，以stride为步长进行切片

## Convolution backward

凑维度

1. 需要用到flip将维度进行翻转
2. 需要用到dilation恢复stride造成的影响

## nn.Conv

1. 初始化W矩阵，初始化偏置b
2. 调整X的形状，计算卷积

## Implementing "ResNet9"

用sequential搭积木

# Part 4: Recurrent neural network [10 points]

## RNNCell

1. 先准备好参数数组（先tensor再parameter），包括两个矩阵W和两个偏置，其中一个矩阵的大小为(n,h)，一个矩阵为(h,h)，两个偏置均为(h,)
2. 计算forward结果，**输出隐层状态**

## RNN

这里实现了一个多层RNN，多层RNN的参数并不多，假设层数为L，这总共就L个RNNCell，L个隐藏状态（只需要记录某一个t的隐层状态），t个输出的h（总共t个时刻，每个时刻的最后一层隐藏）

因此在实现时，会有一个输入的X和一个初始化的h

1. 首先遍历所有时刻，边遍历每个时刻，边记录当前时刻横向的每一个h，和纵向的最后一个h
2. 当前时刻结束时，更新横向的h，给下一个t使用，将最后一个h加入out，最后需要返回
3. 最后返回out和h，分别是横向和纵向的最后一层的隐层状态

# Part 5: Long short-term memory network [10 points]

## LSTMCell

基本和rnn差不多，不过为了方便计算，隐层的长度直接乘4，这样所有的计算都可以用一个式子计算完，之后再拆分为四个中间变量即可

在拆解的时候，刚经过计算的out应该是(batch,4*hs)

1. 首先以第二个维度进行split，得到(4*hs,batch)
2. 以hs为步长，分别取出4个(hs,batch)
3. 对每个的第二个维度进行stack，得到(batch,hs)

## LSTM

在init阶段，构造好每一层的那个rnn_cell，第一层的输入是input_size，输出是hidden_size，其他层的输入输出都是hidden_size

在forward阶段

1. 首先将输入给解构，先判断是否给了初始化的c和h，如果给了的话，和输入的X一样，都split，之后才能正常访问
2. 遍历每一个时刻，这里只需要取出当前时刻的输入，即(m,input_size)。然后每经过一个时刻都要更新这个时刻的所有层的c和h，因为边遍历需要边记录每一层的c和h，这样才能在时刻结束的时候更新
3. 遍历每一层，这里需要取出当前位于第几层，以及这一层的model。输入x，h，c之后，得到输出的h，c，其中输出的h继续用作下一层的x，遍历完所有层时，记录最后一个隐层输出到out数组
4. 最后需要返回out，h，c

# Part 6: Penn Treebank dataset [10 points]

## Dictionary

给每个不同的单词一个id，这个id就是它们输入的顺序

有一个字典一个列表

1. 字典：记录了每个单词对应的id，word->idx
2. 列表：按出现顺序记录了所有不同的单词，通过下标idx可以访问对应的word

## tokenize in Corpus

将数据集中的每个单词都转为一个id，其中以每一行为单位，插入一个<eos>换行符

1. 先以utf-8的格式打开文件，读取每一行到lines
2. 遍历lines中的每一个line
    1. 如果超出了要读的数量，则直接退出
    2. 遍历line中的每一个word，加入ids
    3. 加入换行符的id

## batchify

将输入的data的形状转为`(nbatch, batch_size)`，其中字母的顺序是以列为单位的。因此先reshape再转置即可

## get_batch

将batchify的结果按指定的下标取出，构造成输入data`(bptt, bs)`和标签target`(bptt*bs,)`的tensor。其中`bptt`指的是一个样本的长度，即一个样本包含多少个单词，`bs`指的是一个batch有多少个样本，即多少个样本并行

~~目前还不太懂为什么target的形状是这样的~~，现在知道了。因为在后面训练语言模型时，输入的形状是`(bptt,bs)`，最后输出的形状是`(bptt,bs`。当target的形状是`(bptt*bs,)`时，直接就可以一对一的比较结果是否正确，属于一种实现上的选择。

# Part 7: Training a word-level language model [10 points]

## Embedding

输入为`x of shape (seq_len, bs)`，x中的每一个元素都是一个word的id

1. 先将这个id转为one_hot，这个one_hot的长度通过num_embeddings指定
2. 将one_hot转嵌入向量，就是通过一个矩阵乘法，将num_embeddings转为embedding_dim，可以节省空间，而且可以捕捉到更多的前后文语义

## LanguageModel

在模型的初始化层面，直接调用写好的nn的模块搭积木就好了

初始化模块的各种参数，就两类

1. 模块特有的参数，比如lstm和rnn的num_layers
2. 和样本特征相关的参数，比如ebedding的大小，隐层的大小

在调用模块的层面，完全不用管一个batch有多少个样本什么什么的，只需要关注样本特征的变化，就可以搭出正确的积木

## epoch_general_ptb

1. 设置模式为训练or评估
2. 开始训练
    1. 获取输入的数据和标签
    2. 通过模型，得到输出
    3. 计算损失（如果是softmax，一般输出的形状为$(m,n)$，标签的形状为$(m,)$，这样就可以直接softmax了。这应该也是求loss的一种思想，即标签就是一个一维数组，其中每个数字代表一个标签，而模型的输出的通常是一个二维数组，输出和标签一一对应）
    4. 计算各种指标

在这里，还需要将h给detach，因为模型输出的是一个已经存在于一个网络中的tensor。它还要作为下一次训练的输入，因此，将它detach成一个独立的tensor

## train_ptb

1. 定义优化器
2. 训练n_epoch轮

## evaluate_ptb

返回一次训练的结果即可，opt为None

## 使用ndl训练

```python
import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=ndl.cpu(), dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='rnn', device=ndl.cpu())
train_ptb(model, train_data, seq_len=1, n_epochs=1, device=device)
evaluate_ptb(model, train_data, seq_len=40, device=device)
```


