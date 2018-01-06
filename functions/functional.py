from torch.autograd import Function

from _ext import capsule_lib


class CapsuleConv2d(Function):
    def forward(self, input, weight, stride, padding, num_iterations):
        output = input.new()
        if not input.is_cuda:
            capsule_lib.conv2d_forward(input, weight, stride, padding, num_iterations, output)
        else:
            capsule_lib.conv2d_forward_cuda(input, weight, stride, padding, num_iterations, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            capsule_lib.conv2d_backward(grad_output, grad_input)
        else:
            capsule_lib.conv2d_backward_cuda(grad_output, grad_input)
        return grad_input


class CapsuleLinear(Function):
    def forward(self, input, weight, num_iterations):
        output = input.new()
        if not input.is_cuda:
            capsule_lib.linear_forward(input, weight, num_iterations, output)
        else:
            capsule_lib.linear_forward_cuda(input, weight, num_iterations, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            capsule_lib.linear_backward(grad_output, grad_input)
        else:
            capsule_lib.linear_backward_cuda(grad_output, grad_input)
        return grad_input
