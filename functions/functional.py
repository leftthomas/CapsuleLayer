from torch.autograd import Function

from _ext import my_lib


class CapsuleConv2d(Function):
    def forward(self, input):
        output = input.new()
        if not input.is_cuda:
            my_lib.my_lib_add_forward(input, output)
        else:
            my_lib.my_lib_add_forward_cuda(input, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            my_lib.my_lib_add_backward(grad_output, grad_input)
        else:
            my_lib.my_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input


class CapsuleLinear(Function):
    def forward(self, input):
        output = input.new()
        if not input.is_cuda:
            my_lib.my_lib_add_forward(input, output)
        else:
            my_lib.my_lib_add_forward_cuda(input, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            my_lib.my_lib_add_backward(grad_output, grad_input)
        else:
            my_lib.my_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input
