import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import capsule_layer as CL


class CapsuleConv2d(nn.Module):
    r"""Applies a 2D capsule convolution over an input signal composed of several input
    planes.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the capsule convolution
        kernel_size (int or tuple): Size of the capsule convolving kernel
        in_length (int): length of each input sample's each capsule
        out_length (int): length of each output sample's each capsule
        stride (int or tuple, optional): Stride of the capsule convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        routing_type (str, optional):  routing algorithm type -- options: ['sum', 'dynamic', 'EM']
        kwargs (dict, optional): other args:
          - num_iterations (int, optional): number of routing iterations -- default value is 3, it works for dynamic
           and EM routing algorithms

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels // out_length, in_channels // in_length, kernel_size[0], kernel_size[1],
                        in_length, out_length)

    ------------------------------------------------------------------------------------------------
    !!!!!!!!!     PAY ATTENTION: MAKE SURE CapsuleConv2d's OUTPUT CAPSULE's LENGTH EQUALS
                               THE NEXT CapsuleConv2d's INPUT CAPSULE's LENGTH            !!!!!!!!!!
    ------------------------------------------------------------------------------------------------
    Examples::

        >>> from capsule_layer import CapsuleConv2d
        >>> from torch.autograd import Variable
        >>> # With square kernels and equal stride
        >>> m = CapsuleConv2d(16, 33, 3, 4, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m1 = CapsuleConv2d(16, 33, (3, 5), 4, 3, stride=(2, 1), padding=(4, 2))
        >>> input = Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([20, 33, 24, 49])
        >>> output = m1(input)
        >>> print(output.size())
        torch.Size([20, 33, 28, 100])
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_length, out_length, stride=1, padding=0,
                 routing_type='sum', **kwargs):
        super(CapsuleConv2d, self).__init__()
        if in_channels % in_length != 0:
            raise ValueError('in_channels must be divisible by in_length')
        if out_channels % out_length != 0:
            raise ValueError('out_channels must be divisible by out_length')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_length = in_length
        self.out_length = out_length
        self.stride = stride
        self.padding = padding
        self.routing_type = routing_type
        self.kwargs = kwargs
        self.weight = Parameter(
            torch.randn(out_channels // out_length, in_channels // in_length, *kernel_size, in_length, out_length))

    def forward(self, input):
        return CL.capsule_cov2d(input, self.weight, self.stride, self.padding, self.routing_type, **self.kwargs)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', in_length={in_length}, out_length={out_length}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CapsuleLinear(nn.Module):
    r"""Applies a fully connection capsules to the incoming data

     Args:
         in_capsules (int): number of each input sample's capsules
         out_capsules (int): number of each output sample's capsules
         in_length (int): length of each input sample's each capsule
         out_length (int): length of each output sample's each capsule
         routing_type (str, optional):  routing algorithm type -- options: ['sum', 'dynamic', 'EM']
         kwargs (dict, optional): other args:
           - num_iterations (int, optional): number of routing iterations -- default value is 3, it works for dynamic
            and EM routing algorithms

     Shape:
         - Input: :math:`(N, in_capsules, in_length)`
         - Output: :math:`(N, out_capsules, out_length)`

     Attributes:
         weight (Tensor): the learnable weights of the module of shape
             (out_capsules, out_length, in_capsules, in_length)

     Examples::
         >>> from capsule_layer import CapsuleLinear
         >>> from torch.autograd import Variable
         >>> m = CapsuleLinear(20, 30, 8, 16, routing_type = 'dynamic', num_iterations=1)
         >>> input = Variable(torch.randn(5, 20, 8))
         >>> output = m(input)
         >>> print(output.size())
         torch.Size([5, 30, 16])
     """

    def __init__(self, in_capsules, out_capsules, in_length, out_length, routing_type='sum', **kwargs):
        super(CapsuleLinear, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_type = routing_type
        self.kwargs = kwargs
        self.weight = Parameter(torch.randn(out_capsules, out_length, in_capsules, in_length))

    def forward(self, input):
        return CL.capsule_linear(input, self.weight, self.routing_type, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'
