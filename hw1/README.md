# Introduction to `needle`

两个重要的问题

1. `python/needle/autograd.py`，定义了框架，包括但不限于以下部分
    1. class Op
    2. class TensorOp(Op)
    3. class Value
    4. class Tensor(Value)
    5. compute_gradient_of_variables
2. `python/needle/ops/ops_mathematic.py`，包含多种操作的实现

推荐先熟悉以下类，其他的没那么重要

1. Value

2. Op

    关键在于两个函数

    compute()会计算输出

    gradient()会根据输出的梯度，求出输入的梯度（反向传播）

3. Tensor

4. TensorOp

# 实现forward操作

很多操作就是一行numpy函数解决，但是最后一个需要自己再处理一下逻辑

在numpy中，转置维度必须将所有维度包括了，加入维度是5，那么输入的axes的长度一定是五，然后把要交换的两个维度交换位置，比如这里就是将2和4对应的维度交换，[0,1,4,3,2]

```c++
    def compute(self, a):
        d = a.ndim
        axis = [i for i in range(d)]
        x, y = (d - 1, d - 2) if self.axes is None 						       else(self.axes[0], self.axes[1])
        axis[x], axis[y] = axis[y], axis[x]
        return array_api.transpose(a, axes=axis)
```

# 实现backward计算

## power_scalar

$$
y=x^a\\
\frac{dy}{dx}=ax^{a-1}
$$

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * self.scalar * mul_scalar(node.inputs[0], self.scalar - 1)
```

## divide

$$
y=\frac{a}{b}\\
\frac{dy}{da}=\frac{1}{b}\\
\frac{dy}{db}=-\frac{a}{b^2}
$$

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a_d = divide(out_grad, node.inputs[1])
        b_d = negate(multiply(out_grad, divide(
            node.inputs[0], multiply(node.inputs[1], node.inputs[1]))))
        return a_d, b_d
        # END YOUR SOLUTION
```

## divide_scalar

标量操作很简单

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar
```

## matmul

这个操作就有点恶心了，特别是两个相乘的矩阵存在高纬（超过两维的情况）

高维矩阵相乘，本质上是多个二维矩阵相乘，它们除了最后两个维度要能够相乘外，前面的维度必须完全相同，比如一个四维矩阵A[a,b,c,d]和一个三维矩阵[e,f,g]可以相乘的条件就是[c,d]和[f,g]要能够相乘，并且b要等于e

而求导，也就是将前面维度全部求和，比如C=AB，其中A的维度是[a,b,c,d]，B的维度是[d,e]，C的维度就是[a,b,c,e]

1. 现在损失对A求导的结果的形状肯定也是[a,b,c,d]，这个维度很容易通过C的梯度的形状[a,b,c,e]和B的形状[d,e]凑出来
2. 而损失函数对B求导的形状肯定是[d,e]，但是这个不能直接由C的梯度[a,b,c,e]和A的梯度[a,b,c,d]凑出来，因为多了前面两维[a,b]，将这两维给除去的方法，就是将对应位置给加起来，通过numpy.sum(0,1)就可以达到这个效果。（为什么相加之后结果就是B的导数，就和矩阵求导相关了，我也不会证明）

最后，对一个高维矩阵，比如[a,b,c,d]进行转置，好像只会将它转为[a,b,d,c]

```c++
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
```

## reshape

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        # END YOUR SOLUTION
```

## negate

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return negate(out_grad)
```

## transpose

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return transpose(out_grad, self.t)
```

## broadcast

1. 将变化的维度给找出来，比如原始数据是[3,5,1,4]，被广播成了[3,5,4,4]，那么要做的就是将第3维给找出来，也就是1->4的这一维，然后将这一维的梯度通过sum相加
2. 最后还要reshape成原来的样子

```c++
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
```

## summation

summation是对维度进行了压缩，那么在反向传播时，肯定要将其进行广播

所以

1. 先找出哪些维度被压缩了
2. reshape成被压缩的形状（之前通过sum操作，可能直接将维度给干没了，所以要reshape恢复维度，即使那个维度的长度是1）
3. 广播成输入的形状（为啥可以直接广播就是正确的梯度，和矩阵求导相关）

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
```

# 拓扑排序

这个部分感觉题意有点难理解，本质上很简单

```c++
def find_topo_sort(node_list: List[Value]) -> List[Value]:
```

这个函数给出了一个node_list，这里面可能有多个node，输出以这个node为终点的拓扑排序（正拓扑排序）

> 以目前我的理解，不太明白为什么是node_list，一般不就是用最后lost的函数的结点为终点找拓扑排序吗。
>
> 我看了一下测试的代码，也都是只用了一个节点，先留个坑

下面这个函数给了一个dfs的模板

1. 第一个参数node代表当前遍历到了哪个结点
2. 第二个参数visited代表已经有哪些结点被访问过了
3. 第三个参数topo_order代表目前已经找到的拓扑排序，我们最后也就是要返回这个作为拓扑排序的答案

```c++
def topo_sort_dfs(node, visited, topo_order):
```

实现代码很简单，就是题意对我来说有一点晦涩，感觉和不太熟悉怎么用终点推出拓扑排序有关，菜狗

```c++
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    ans = []
    vis = set()
    for node in node_list:
        topo_sort_dfs(node, vis, ans)
    return ans
def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    for ne in node.inputs:
        topo_sort_dfs(ne, visited, topo_order)
    visited.add(node)
    topo_order.append(node)
```

# 反向模式微分

函数签名

```c++
def compute_gradient_of_variables(output_tensor, out_grad)
```

1. 第一个参数是最后输出的那个tensor
2. 第二个参数是这个tensor的梯度

函数内已有的变量

```c++
node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
```

这个字典表示，当前tensor对应的，所有输出tensor对它的偏导数

使用起来也很简单，只要将对应的偏导数全部相加，就是当前tensor的梯度

```c++
reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
```

这就是反向的拓扑排序

实现思路基本就是下面这个图

![image-20231024161041846](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231024161041846.png)

但是需要加入一些对当前框架的理解

1. 首先，这个函数的作用就是计算出每一个tensor对应的梯度，而通过上图也可以看到，我们在反向传播中是通过新建了一个tensor来保存梯度的。按照这个框架的意思，好像是将这个新建的tensor作为原来tensor的一个变量存入原来的tensor，这里可以用self.grad来存。感觉这个参数可以随便命名，可能后面的调用都是自己写
2. 这里要用到之前写的各种op中的gradient函数了，可以通过`gradient_as_tuple`来调用。这个函数也正是生成了对输入的所有tensor对应的偏导数tensor，核心操作了属于是

因此，实现思路

1. 按照反向拓扑排序的顺序来遍历所有tensor
2. 根据`node_to_output_grads_list`对应的值，计算这个tensor的梯度，这里可以用框架提供的`sum_node_list`来计算梯度之和。然后将计算好的梯度存入当前node
3. 如果当前tensor没有input结点，也就是op为None，直接continue
4. 如果当前tensor有input结点
    1. 先通过`gradient_as_tuple`取出这个tensor对所有input的偏导数tensor
    2. 然后将input和偏导数tensor一一对应的访问，加入字典`node_to_output_grads_list`中

```c++
def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        grads = sum_node_list(node_to_output_grads_list[node])
        node.grad = grads
        if node.op is None:
            continue
        node_to_input_grad = node.op.gradient_as_tuple(node.grad, node)
        for input_node, input_grad in zip(node.inputs, node_to_input_grad):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
 node_to_output_grads_list[input_node].append(input_grad)
```

# softmax loss

先实现两个op，要注意，在反向传播的过程中，要全程调用tensor的op，不能再用numpy的操作了

```c++
class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)
    def gradient(self, out_grad, node):
        return divide(out_grad, node.inputs[0])
class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)
    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))
```

实现softmax，基本就是翻译hw0里的softmax_loss函数，这里用了比较巧的计算方法， 那就是将减号前面和减号后面的分开计算，然后统一相减后求平均值

![image-20231024171353873](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231024171353873.png)

```c++
def softmax_loss(Z, y_one_hot):
    # BEGIN YOUR SOLUTION
    e_z = ndl.exp(Z)
    e_s = ndl.summation(e_z, (1,))
    e_l = ndl.log(e_s)
    e_s = ndl.summation(e_l)
    // 通过矩阵相乘，将one_hot向量变成对应的值
    y_s = ndl.summation(y_one_hot * Z)
    return (e_s - y_s) / Z.shape[0]
    # END YOUR SOLUTION
```

# 两层神经网络的SGD

1. 第一步，增加relu函数

前向计算比较简单

```c++
    def compute(self, a):
        return array_api.maximum(0, a)
```

反向传播的时候，原来小于0的部分梯度为0，不用更新，所以只需要更新原来大于0的部分，这部分函数是$relu(x)=x$，所以导数还是1

```c++
    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.realize_cached_data().copy()
        data[data > 0] = 1
        return multiply(out_grad, Tensor(data))
```

2. 第二步，完成sgd

基本的流程和hw0基本一致，但是有几个注意点

1. 这里的X和y是numpy类型，使用时要先转为Tensor
2. 反向传播是从最后一个结点开始的，这里就是loss
3. 更改参数直接操作W1和W2里面cached_data就可以完成

```c++
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):    
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
```












