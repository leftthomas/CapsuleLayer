from torch.autograd import Function

from .._ext import capsule_layer_lib


class CapsuleLayerFunction(Function):
    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            capsule_layer_lib.capsule_layer_lib_add_forward(input1, input2, output)
        else:
            capsule_layer_lib.capsule_layer_lib_add_forward_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            capsule_layer_lib.capsule_layer_lib_add_backward(grad_output, grad_input)
        else:
            capsule_layer_lib.capsule_layer_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input
