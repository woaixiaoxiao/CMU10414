# 熟悉 NDArray 

NDArray类本质上就是对真正的数据进行了抽象，它主要由以下几部分组成

1. handle，其实代表的就是数据，而这个数据是以一个一维数组的形式存在
2. shape，虽然数据本质上是一维存在，但是我们在使用时，肯定是要用到多维数组的，这个shape就指定了这个多维数组
3. stride，stride可以指定每一维包含多少数据，可以方便地改变我们对数据的shape的解释
4. offest，表示当前NDArray的数据在handle中的起始位置
5. device，表示是在哪种设备上

# Part 1: Python array operations

## reshape

只需要修改shape和stride，其中stride可以通过`compact_strides`得到

```c++
        if self.size != prod(new_shape):
            raise ValueError()
        if self.is_compact() is False:
            return ValueError()
        return NDArray.make(
            new_shape,
            self.compact_strides(new_shape),
            self.device,
            self._handle,
            self._offset
            )
```

## permute

`permute(self, new_axes)`，通过指定axes来打乱原先的维度，要做的也只是修改shape和stride

```c++
        new_shape = tuple(np.array(self.shape)[list(new_axes)])
        new_stride = tuple(np.array(self.strides)[list(new_axes)])
        return NDArray.make(
            new_shape,
            new_stride,
            self.device,
            self._handle,
            self._offset
            )
```

## broadcast_to

`broadcast_to(self, new_shape)`，将一些长度为1的维度广播成指定的长度，要改变shape和stride，其中stride就是将广播的维度的stride都设置为0即可

```c++
        assert (len(new_shape) == len(self.shape))
        new_stride = np.array(self.strides)
        for i in range(len(self.shape)):
            if self.shape[i] != new_shape[i]:
                if self.shape[i] != 1:
                    raise AssertionError
                else:
                    new_stride[i] = 0
        return NDArray.make(
            tuple(new_shape), tuple(
                new_stride), self.device, self._handle, self._offset
            )
```

## getitem

`__getitem__(self, idxs)`，这是对取值符号`[]`的重载，框架保证idxs是一个元组，并且里面的每个元素都是切片的形式，并且都是从前往后遍历的常规切片，格式为$[start,stop,step]$

这里shape，stride，offset都要修改

shape：计算出切片之后，每个维度还剩下的长度

stride：在原来的stride基础上，乘上切片指定的step

offset：计算切片的起始位置（每个维度都要）

```c++
        # BEGIN YOUR SOLUTION
        # 计算新的shape
        new_shape = []
        for sl in idxs:
            start, end, step = sl.start, sl.stop, sl.step
            new_shape.append(math.ceil((end - start) / step))
        # 计算新的offset
        offset = []
        for st, sl in zip(self.strides, idxs):
            offset.append(st * sl.start)
        new_offset = sum(np.array(offset))
        # 计算新的stride
        new_strides = []
        for st, sl in zip(self.strides, idxs):
            new_strides.append(st * sl.step)
        return NDArray.make(
            tuple(new_shape),
            tuple(new_strides),
            self.device,
            self._handle,
            new_offset
            )
        # END YOUR SOLUTION
```

# Part 2: CPU Backend - Compact and setitem

这个部分要完成三个函数，但是基本一样

## compact

```c++
void Compact(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,std::vector<int32_t> strides, size_t offset)
```

a是一个不紧凑的数组，要将其按照shape，strides和offset的格式紧凑到out中，本来就是一个多重循环就搞定了，但是因为这里shape的维度不定，不知道要写几重循环，所以需要模拟循环多重的过程

## EwiseSetitem

```c++
void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset)
```

这里面a是一个紧凑数组，out不是紧凑的，后面的三个参数是用来形容out的

这里是将一个紧凑的变成一个不紧凑的，可以用来实现切片复制，比如`a[,3:4,]=b`

## ScalarSetitem

```c++
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset)
```

将值设置为固定值

## 实现

要注意的是，这里面a和out的需要操作的数量一定是相同的

即不紧凑的那个的shape和stride相乘之和，一定等于紧凑的那个的cnt

```c++
void RealOp(const AlignedArray *a, AlignedArray *out, std::vector<int32_t> shape,
            std::vector<int32_t> strides, size_t offset, OpMode mode, int val) {
    int for_len = shape.size();             // for循环的层数
    std::vector<uint32_t> loop(for_len, 0); // 记录当前各层的循环次数，0代表最外层，for_len-1代表最里面
    int cnt = 0;                            // out的指针
    // 开始循环
    while (true) {
        // 计算a的索引，并给out赋值
        int idx = offset;
        for (int i = 0; i < for_len; i++) {
            idx += loop[i] * strides[i];
        }
        switch (mode) {
        case in_mode:
            out->ptr[cnt] = a->ptr[idx];
            cnt += 1;
            break;
        case out_mode:
            out->ptr[idx] = a->ptr[cnt];
            cnt += 1;
            break;
        case set_mode:
            out->ptr[idx] = val;
            break;
        }
        // 更新loop，先更新最内层，然后根据有没有超出shape判断是否要更新上一层的loop
        int modify_idx = for_len - 1;
        loop[modify_idx] += 1;
        while (loop[modify_idx] == shape[modify_idx]) {
            if (modify_idx == 0) {
                return;
            }
            loop[modify_idx] = 0;
            modify_idx -= 1;
            loop[modify_idx] += 1;
        }
    }
}
```

# Part 3: CPU Backend - Elementwise and scalar operations

实现各种操作，通过std提供的数学函数来实现都比较简单

这里可以看一下Python文件和C文件是怎么联动的

通过`m.def("ewise_mul", EwiseMul);`就可以将python中的`ewise_mul`对应到c语言的`EwiseMul`

而在Python文件中，先对操作符进行重载，然后将相关的ewise或者scalar函数当做参数传入`ewise_or_scalar`函数。这个函数进一步根据other的类型选择执行哪个

```c++
self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
            )
```

如果other是ndarray，就会执行

```c++
ewise_func(self.compact()._handle,
           other.compact()._handle, out._handle)
```

而这个函数也对应到了这样的c语言参数列表

```c++
void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
```

现在还不太清楚的是为什么传入的都是handle，怎么out的就是指针，其他的就是引用

# Part 4: CPU Backend - Reductions

这里要完成对特定维度的sum或者max函数

在numpy中，可以通过直接axes实现，但是在c数组中，就有点麻烦了

lab先在Python文件中对这个操作进行了预处理

1. 首先，只允许指定一个轴
2. 通过改变shape和stride，将我们需要操作的一个reduce块变成连续的
    1. 首先，将要操作的那个维度放到shape的最后，可以通过`permute`来完成维度的移动。
    2. 构造出reduce操作之后out的形状，就是将操作的那个轴的长度变成1

因此，在c文件中，只需要将a数组分段求和或者求最大值，然后依次存入out数组就行了

# Part 5: CPU Backend - Matrix multiplication

第一部分的`Matmul`就是常规的二维矩阵乘法，记得要在累加之前先清零

第二部分的`MatmulTiled`是以`Tile`为变成的正方形进行操作

1. 首先需要完成`AlignedDot`，这个函数是对两个正方形矩形做乘法

    ```c++
        for (size_t i = 0; i < TILE; i++) 
            for (size_t j = 0; j < TILE; j++) 
                for (size_t k = 0; k < TILE; k++) 
                    out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
    ```

    需要注意的是，这里面的out后续还会加上其他的正方形相乘之后，因此不能在操作前清零

2. 在上面这个函数的帮助下，求出两个四维矩阵相乘的结果

    只需要遍历out每一个`Tile×Tile`的正方形，它是由很多个a和b的正方形相乘得到。在``AlignedDot``的抽象下可以将正方形看成一个普通的整数

    `MatmulTiled`需要在开始操作之前对out初始化为0

之所以要用Tile这种乘法，好像是为了利用cpu的向量化的操作，这一块目前还没啥了解

1. `__restrict__`表示这一块区域只有当前指针可以访问到，因此不会和别人有重复的区域。这个应该是可以增加cpu的并行性能
2. `__builtin_assume_aligned`表示这个指针在内存中已经对齐了。这个应该是增加内存访问的性能

# Part 6: CUDA Backend - Compact and setitem

要完成的效果和cpu的基本一样，但是因为cuda的特点，我们不需要手动模拟n重for循环，直接用线程块的id来表示当前处于for循环的哪个位置

因此，通过gid和shape，就可以得出来loop数组，然后再在便宜的基础上加上loop数组和stride数组的点积，就可以得到非compact数组的坐标。这里要注意，对于紧凑的数组，stride可以通过shape计算出来，但是对于非紧凑的数组，stride和shape没有必然的关系，因此在这里计算loop数组的时候，需要手动倒序遍历shape数组

```c++
__device__ size_t GetIndex(size_t cnt, CudaVec shape, CudaVec strides, size_t offset) {
    int pre_len = 1, cur_len;
    size_t loop[MAX_VEC_SIZE];
    for (int i = shape.size - 1; i >= 0; i--) {
        cur_len = pre_len * shape.data[i];
        loop[i] = (cnt % cur_len) / pre_len;
        pre_len = cur_len;
    }
    size_t idx = offset;
    for (int i = 0; i < strides.size; i++) {
        idx += loop[i] * strides.data[i];
    }
    return idx;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset, Modes mode, int val = -1) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        int idx = GetIndex(gid, shape, strides, offset);
        switch (mode) {
        case mode_in:
            out[gid] = a[idx];
            break;
        case mode_out:
            out[idx] = a[gid];
            break;
        case mode_set:
            out[idx] = val;
            break;
        }
    }
}
```

这两个函数就是最重要的，要实现的三个函数调用这个函数即可。要注意传入的size，这个size应该是紧凑的数组的长度，因此在Compact函数中要传入out.size，而在EwiseSetitem中，应该传入a.size。而传给CudaDims的size同理，也是这样的。

# Part 7: CUDA Backend - Elementwise and scalar **operations**

复制粘贴， 替换符号

感觉正确的做法应该是重载操作符，但是我不会

# Part 8: CUDA Backend - Reductions

意思和cpu的一样，很简单了

# Part 9: CUDA Backend - Matrix multiplication

这里需要实现cuda上的矩阵乘法，不优化的还是挺好写的

首先要自己构造grid和block，为了防止M和P过大，这里选择让grid的长和宽为`BASE_THREAD_NUM`，block的长和宽为M和P除以这个值的上取整

```c++
    dim3 grid(BASE_THREAD_NUM, BASE_THREAD_NUM);
    dim3 block(Ceil(M), Ceil(P));
```

在核函数中，用当前线程的x和y坐标代表out的坐标，然后用一个for循环就可以完成点积和累加了



























