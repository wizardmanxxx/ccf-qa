import torch.nn as nn
import torch

if __name__ == "__main__":
    # rnn = nn.LSTM(10, 20, 2)
    # input = torch.randn(5, 3, 10)
    # h0 = torch.randn(2, 3, 20)
    # c0 = torch.randn(2, 3, 20)
    # output, (hn, cn) = rnn(input, (h0, c0))
    # print(output)
    #
    # a = torch.ones([3])
    # b = torch.sum(a)
    # print(b)
    # a = torch.rand([2, 3, 4])
    # print(a)
    # print(a[:-1, :, :])

    # a = torch.rand(2, 3)
    # b = torch.zeros(1, 3)
    # print(a)
    # print(b)
    # print(torch.cat((a, b), 0))
    # a = torch.randn(4, 5)
    # print(a)
    # print(a[range(2), :])
    # a = torch.rand([1])
    # print(a)
    # print(a.dim())
    a = torch.Tensor([1, 2, 3, 4, 5])
    print(a[0:2])
