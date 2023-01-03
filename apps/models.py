import sys
from typing import Optional
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size, 
        stride=1,
        bias=True, 
        device=None,
        dtype="float32",
    ) -> None:
        super().__init__()
        self.device = device

        self.conv = nn.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.batch_norm = nn.BatchNorm2d(
            out_channels,
            device=device,
            dtype=dtype,
        )

        self.relu = nn.ReLU()

    def forward(self, X: ndl.Tensor) -> ndl.Tensor:
        res = self.conv(X)
        res = self.batch_norm(res)
        return self.relu(res)


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.convs = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype), 
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype), 
                ),
            ),
            ConvBN(32, 64, 3, 2, device=device, dtype=dtype), 
            ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype), 
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ),
            ),
        )

        self.flatten = nn.Flatten()

        self.classification_layer = nn.Sequential(
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, X: ndl.Tensor) -> ndl.Tensor:
        res = self.convs(X)
        res = self.flatten(res)
        return self.classification_layer(res)


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        self.device = device

        self.embedding_layer = nn.Embedding(
            output_size,
            embedding_size,
            device=device,
            dtype=dtype,
        )
        
        if seq_model == 'rnn':
            seq2seq_cls = nn.RNN
        elif seq_model == 'lstm':
            seq2seq_cls = nn.LSTM
        else:
            raise NotImplementedError()

        self.seq2seq = seq2seq_cls(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

        self.linear_layer = nn.Linear(
            hidden_size,
            output_size,
            device=device,
            dtype=dtype,
        )


    def forward(self, X: ndl.Tensor, h: Optional[ndl.Tensor] = None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        embeddings = self.embedding_layer(X)
        hidden, last = self.seq2seq(embeddings, h)
        seq_len, bs, hidden_size = hidden.shape
        return self.linear_layer(hidden.reshape((seq_len * bs, hidden_size))), last


####### DEQ ######

class TanhLinearDEQLayer(nn.Module):
    def __init__(self, out_features: int, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.linear = nn.Linear(out_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, Z: ndl.Tensor, X: ndl.Tensor) -> ndl.Tensor:
        return ndl.ops.tanh(self.linear(Z) + X)    


class TanhLinearDEQ(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_size: int, 
        n_classes: int,
        device=None,
        dtype="float32", 
        use_deq: bool = False,
        depth: int = 50,
        solver: ndl.solver.BaseSolver = ndl.solver.ForwardIteration(),
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.flatten = nn.Flatten()
        self.input_linear = nn.Linear(in_features, hidden_size, device=device, dtype=dtype)

        self.use_deq = use_deq
        if self.use_deq:
            self.deq = nn.DEQ(TanhLinearDEQLayer(hidden_size, device=device, dtype=dtype), solver)
        else:
            self.layers = [TanhLinearDEQLayer(hidden_size, device=device, dtype=dtype) for _ in range(depth)]

        self.classification_layer = nn.Linear(hidden_size, n_classes, device=device)

    def forward(self, X: ndl.Tensor) -> ndl.Tensor:
        X = self.input_linear(self.flatten(X))

        if self.use_deq:
            Z = self.deq(X)
        else:
            Z = ndl.init.zeros_like(X)
            for layer in self.layers:
                Z = layer(Z, X)
        
        return self.classification_layer(Z)



class ResNetDEQLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        kernel_size: int,
        device=None, 
        dtype="float32",
):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.conv1 = nn.Conv(
            in_channels, 
            hidden_size, 
            kernel_size, 
            bias=False, 
            device=device, 
            dtype=dtype,
        )
        self.conv2 = nn.Conv(
            hidden_size, 
            in_channels, 
            kernel_size,
            bias=False,
            device=device, 
            dtype=dtype,
        )
        self.norm1 = nn.BatchNorm2d(
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.norm2 = nn.BatchNorm2d(
            in_channels,
            device=device,
            dtype=dtype,
        )
        self.norm3 = nn.BatchNorm2d(
            in_channels,
            device=device,
            dtype=dtype,
        )

        self.relu = nn.ReLU()
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, Z: ndl.Tensor, X: ndl.Tensor) -> ndl.Tensor:
        Y = self.norm1(self.relu(self.conv1(Z)))
        return self.norm3(self.relu(Z + self.norm2(X + self.conv2(Y))))


class ResNetDEQ(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        kernel_size: int,
        n_classes: int,
        device=None, 
        dtype="float32",
        use_deq: bool = False,
        depth: int = 50,
        solver: ndl.solver.BaseSolver = ndl.solver.ForwardIteration(),
    ):
        self.device = device
        self.dtype = dtype
        self.use_deq = use_deq
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, device=device, dtype=dtype)
        self.norm1 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        if self.use_deq:
            self.deq = nn.DEQ(
                ResNetDEQLayer(out_channels, hidden_size, kernel_size, device=device, dtype=dtype),
                solver,
            )
        else:
            self.layers = [
                ResNetDEQLayer(out_channels, hidden_size, kernel_size, device=device, dtype=dtype)
                for _ in range(depth)
            ]
        self.norm2 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.classification_layer = nn.Linear(out_channels * 32 * 32, n_classes, device=device)
        self.flatten = nn.Flatten()

    def forward(self, X: ndl.Tensor) -> ndl.Tensor:
        X = self.norm1(self.conv(X))

        if self.use_deq:
            Z = self.deq(X)
        else:
            Z = ndl.init.zeros_like(X)
            for layer in self.layers:
                Z = layer(Z, X)     
       
        return self.classification_layer(self.flatten(self.norm2(Z)))



if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)