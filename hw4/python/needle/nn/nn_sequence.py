"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from needle.ops.ops_mathematic import Tanh
from .nn_basic import Parameter, Module, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return 1 / (1 + ops.exp(-1))
        # END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-1 / hidden_size**0.5,
                high=1 / hidden_size**0.5,
                device=device,
                dtype=dtype,
                requires_grad=True
                )
            )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-1 / hidden_size**0.5,
                high=1 / hidden_size**0.5,
                device=device,
                dtype=dtype,
                requires_grad=True
                )
            )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-1 / hidden_size**0.5,
                    high=1 / hidden_size**0.5,
                    device=device,
                    dtype=dtype,
                    requires_grad=True
                    )
                )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-1 / hidden_size**0.5,
                    high=1 / hidden_size**0.5,
                    device=device,
                    dtype=dtype,
                    requires_grad=True
                    )
                )
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        # BEGIN YOUR SOLUTION

        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        out = X @ self.W_ih
        if self.bias:
            out += self.bias_ih.reshape((1, -1)).broadcast_to(out.shape)
        if h:
            out += h @ self.W_hh
        if self.bias:
            out += self.bias_hh.reshape((1, -1)).broadcast_to(out.shape)
        return self.nonlinearity(out)
        # END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        rnn_cells = [RNNCell(input_size, hidden_size, bias,
                             nonlinearity, device, dtype)]
        for i in range(num_layers - 1):
            rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias,
                        nonlinearity, device, dtype)
                )
        self.rnn_cells = rnn_cells
        # END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        # 对输入的x和h进行预处理，将纯tensor，变成一个列表中的多个tensor，便于访问
        xs = ops.split(X, 0)
        if h0 is None:
            hs = [None for i in range(self.num_layers)]
        else:
            hs = ops.split(h0, 0)
        # 遍历每个时刻
        out = []
        for t, x in enumerate(xs):
            # 记录这个时刻的所有隐层
            hidden_layers = []
            # 遍历每一层
            for l, model in enumerate(self.rnn_cells):
                x = model(x, hs[l])
                hidden_layers.append(x)
            out.append(x)
            hs[l] = hidden_layers
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        return out, hs
        # END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        il, hl = input_size, hidden_size
        self.W_ih = Parameter(
            init.rand(il, 4 * hl, low=-1 / hl**0.5, high=1 / hl**0.5,
                      device=device, dtype=dtype, requires_grad=True)
            )
        self.W_hh = Parameter(
            # NOTE: hl, 4*hl
            init.rand(hl, 4 * hl, low=-1 / hl**0.5, high=1 / hl**0.5,
                      device=device, dtype=dtype, requires_grad=True)
            )
        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hl, low=-1 / hl**0.5, high=1 / hl**0.5,
                          device=device, dtype=dtype, requires_grad=True)
                )
            self.bias_hh = Parameter(
                init.rand(4 * hl, low=-1 / hl**0.5, high=1 / hl**0.5,
                          device=device, dtype=dtype, requires_grad=True)
                )
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        # BEGIN YOUR SOLUTION
        # 预处理输入
        if h:
            h0, c0 = h
        else:
            h0, c0 = None, None
        # 开始计算输出
        out = X @ self.W_ih
        if self.bias:
            out += self.bias_ih.reshape((1, self.hidden_size)
                                        ).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, self.hidden_size)
                                        ).broadcast_to(out.shape)
        if h0:
            out += h0 @ self.W_hh
        # 将输出划分为ifgo,(batch,4*hl)
        out = ops.split(out, 1)  # (4*hl,batch)
        i, f, g, o = [self.getitem(out, i) for i in range(4)]
        # 计算两个输出
        c = i * g
        if c0:
            c += f * c0
        h = o * self.tanh(c)
        return h, c
        # END YOUR SOLUTION

    def getitem(self, out, pos):
        ans = []
        for i in range(pos * self.hidden_size, pos * (self.hidden_size + 1)):
            ans.append(out[i])
        # (hl,batch) -> (batch,hl)
        ans = ops.stack(tuple(ans), 1)
        if pos == 2:
            return self.tanh(ans)
        else:
            return self.sigmoid(ans)


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        # BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(num_layers):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        # 预处理输入的tensor
        xs = ops.split(X, 0)
        if h is None:
            hs, cs = None, None
        else:
            hs, cs = h
            hs = ops.split(hs, 0)
            cs = ops.split(cs, 0)
        # 开始计算
        out = []
        # 遍历时刻
        for i, x in enumerate(xs):
            h_ = []
            c_ = []
            # 遍历层
            for l, model in enumerate(self.lstm_cells):
                x, ci = model(x, (hs[l], cs[l]))
                h_.append(x)
                c_.append(ci)
            # 更新结果
            out.append(x)
            hs[i] = h_
            cs[i] = c_
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        cs = ops.stack(cs, 0)
        return (out, (hs, cs))
        # END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        # BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                num_embeddings, embedding_dim, device=device, dtype=dtype
                )
            )
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # BEGIN YOUR SOLUTION
        se, bs = x.shape
        # (seq_len, bs, num_embeddings)
        one_hot = init.one_hot(
            self.num_embeddings, x, device=x.device, dtype=x.dtype
            )
        one_hot = one_hot.reshape((-1, self.num_embeddings))
        out = one_hot @ self.weight
        out.reshape((se, bs, self.embedding_dim))
        return out
        # END YOUR SOLUTION
