import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from capsule_layer.capsule_cpu import capsule_conv2d_cpu, capsule_linear_cpu
from capsule_layer.utils import load_kernel, Dtype, Stream, CUDA_NUM_THREADS, GET_BLOCKS, capsule_linear_kernels, \
    capsule_conv2d_kernels


class CapsuleConv2d(Function):

    def __init__(self, stride, padding, num_iterations):
        super(CapsuleConv2d, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.num_iterations = num_iterations

    def forward(self, input, weight):
        if input.dim() != 4:
            raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))
        if not (input.is_cuda and weight.is_cuda):
            raise ValueError("Expected input tensor and weight tensor should be in cuda, got cpu tensor instead.")

        kernel_size = (weight.size(2), weight.size(3))
        in_length = weight.size(4)
        out_length = weight.size(-1)
        batch_size, in_channels, in_height, in_width = input.size()
        out_height = 1 + (in_height + 2 * self.padding[0] - kernel_size[0]) // self.stride[0]
        out_width = 1 + (in_width + 2 * self.padding[1] - kernel_size[1]) // self.stride[1]
        out_channels = weight.size(0) * out_length

        with torch.cuda.device_of(input):
            output = input.new(batch_size, out_channels, out_height, out_width)
            n = output.numel()
            f = load_kernel('capsule_conv2d_forward', capsule_conv2d_kernels, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, in_channels=in_channels,
                            in_height=in_height, in_width=in_width,
                            out_height=out_height, out_width=out_width,
                            in_length=in_length, out_length=out_length,
                            kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                            stride_h=self.stride[0], stride_w=self.stride[1],
                            pad_h=self.padding[0], pad_w=self.padding[1])
            f(args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    def backward(self, grad_output):
        return grad_output


class CapsuleLinear(Function):

    def __init__(self, num_iterations):
        super(CapsuleLinear, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, input, weight):
        if input.dim() != 3:
            raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))
        if not (input.is_cuda and weight.is_cuda):
            raise ValueError("Expected input tensor and weight tensor should be in cuda, got cpu tensor instead.")

        batch_size, in_capsules, in_length = input.size()
        out_capsules, out_length = weight.size(0), weight.size(-1)
        with torch.cuda.device_of(input):
            output = input.new(batch_size, out_capsules, out_length)
            n = output.numel()
            f = load_kernel('capsule_linear_forward', capsule_linear_kernels, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, in_capsules=in_capsules, in_length=in_length, out_capsules=out_capsules,
                            out_length=out_length, num_iterations=self.num_iterations)
            f(args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    def backward(self, grad_output):
        return grad_output


def capsule_cov2d(input, weight, stride=1, padding=0, num_iterations=3):
    if input.size(1) != weight.size(1) * weight.size(4):
        raise ValueError("Expected input tensor has the same in_channels as weight, got {} in_channels in input tensor,"
                         " {} in_channels in weight.".format(input.size(1), weight.size(1) * weight.size(4)))
    if input.is_cuda:
        out = CapsuleConv2d(stride, padding, num_iterations)(input, weight)
    else:
        out = capsule_conv2d_cpu(input, weight, stride, padding, num_iterations)
    return out


def capsule_linear(input, weight, num_iterations=3):
    if input.size(1) != weight.size(1):
        raise ValueError("Expected input tensor has the same in_capsules as weight, got {} "
                         "in_capsules in input tensor, {} in_capsules in weight.".format(input.size(1), weight.size(1)))
    if input.size(2) != weight.size(2):
        raise ValueError("Expected input tensor has the same in_length as weight, got in_length {} "
                         "in input tensor, in_length {} in weight.".format(input.size(2), weight.size(2)))
    if input.is_cuda:
        out = CapsuleLinear(num_iterations)(input, weight)
    else:
        out = capsule_linear_cpu(input, weight, num_iterations)
    return out
