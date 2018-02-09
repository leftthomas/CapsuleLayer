import torch
from torch.autograd import Function
from torch.autograd import Variable


class ReLUF(Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, output_grad):
        input = self.saved_tensors[0]

        input_grad = output_grad.clone()
        input_grad[input < 0] = 0
        return input_grad


if __name__ == "__main__":
    a = torch.randn(2, 3)
    va = Variable(a, requires_grad=True)
    vb = ReLUF()(va)
    print(va.data, vb.data)

    vb.backward(torch.ones(va.size()))
    print(vb.grad.data, va.grad.data)
