import torch
from torch.autograd import Function
from torch.autograd import Variable

import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, input):
        return LinearF()(input, self.weight)


class LinearF(Function):

    def forward(self, input, weight):
        self.save_for_backward(input, weight)
        output = torch.mm(input, weight.t())
        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors

        grad_input = grad_weight = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(grad_output, weight)
        if self.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input)
        return grad_input, grad_weight


if __name__ == "__main__":
    module = Linear(in_features=5, out_features=3)
    x = Variable(torch.randn(2, 5))
    print('x:')
    print(x)
    y = module(x)
    print('weight:')
    print(module.weight)
    z = y.sum()
    z.backward()
    print('weight.grad:')
    print(module.weight.grad)
