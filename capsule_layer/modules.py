import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import capsule_layer as CL


class CapsuleConv2d(nn.Module):
    r"""Applies a 2D capsule convolution over an input signal composed of several input
    planes.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the capsule convolution
        kernel_size (int or tuple): size of the capsule convolving kernel
        in_length (int): length of each input capsule
        out_length (int): length of each output capsule
        stride (int or tuple, optional): stride of the capsule convolution
        padding (int or tuple, optional): zero-padding added to both sides of the input
        dilation (int or tuple, optional): spacing between kernel elements
        share_weight (bool, optional): if True, share weight between input capsules
        routing_type (str, optional): routing algorithm type
           -- options: ['dynamic', 'k_means']
        num_iterations (int, optional): number of routing iterations
        squash (bool, optional): squash output capsules or not, it works for all routing
        kwargs (dict, optional): other args:
           - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
               -- options: ['dot', 'cosine', 'tonimoto', 'pearson']

    Shape:
        - Input: (Tensor): (N, C_{in}, H_{in}, W_{in})
        - Output: (Tensor): (N, C_{out}, H_{out}, W_{out}) where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel_size[0] -1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel_size[1] -1) - 1) / stride[1] + 1)`

    Attributes:
         if share_weight:
            - weight (Tensor): the learnable weights of the module of shape
              (out_channels // out_length, out_length, in_length, kernel_size[0], kernel_size[1])
        else:
            -  weight (Tensor): the learnable weights of the module of shape
              (out_channels // out_length, in_channels // in_length, out_length, in_length, kernel_size[0], kernel_size[1])

    ------------------------------------------------------------------------------------------------
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        MAKE SURE THE CapsuleConv2d's OUTPUT CAPSULE's LENGTH EQUALS
                               THE NEXT CapsuleConv2d's INPUT CAPSULE's LENGTH
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ------------------------------------------------------------------------------------------------
    Examples::
        >>> import torch
        >>> from capsule_layer import CapsuleConv2d
        >>> # With square kernels and equal stride
        >>> m = CapsuleConv2d(16, 24, 3, 4, 6, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m1 = CapsuleConv2d(3, 16, (3, 5), 3, 4, stride=(2, 1), padding=(4, 2))
        >>> input = torch.rand(20, 16, 20, 50)
        >>> output, prob = m(input)
        >>> print(output.size())
        torch.Size([20, 24, 9, 24])
        >>> input = torch.rand(10, 3, 14, 25)
        >>> output, prob = m1(input)
        >>> print(output.size())
        torch.Size([10, 16, 10, 25])
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_length, out_length, stride=1, padding=0, dilation=1,
                 share_weight=True, routing_type='k_means', num_iterations=3, squash=False, **kwargs):
        super(CapsuleConv2d, self).__init__()
        if in_channels % in_length != 0:
            raise ValueError('Expected in_channels must be divisible by in_length.')
        if out_channels % out_length != 0:
            raise ValueError('Expected out_channels must be divisible by out_length.')
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_length = in_length
        self.out_length = out_length
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.share_weight = share_weight
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.squash = squash
        self.kwargs = kwargs

        if self.share_weight:
            self.weight = Parameter(torch.Tensor(out_channels // out_length, out_length, in_length, *kernel_size))
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels // out_length, in_channels // in_length, out_length, in_length, *kernel_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return CL.capsule_cov2d(input, self.weight, self.stride, self.padding, self.dilation, self.share_weight,
                                self.routing_type, self.num_iterations, self.squash, **self.kwargs)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', in_length={in_length}, out_length={out_length}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CapsuleLinear(nn.Module):
    r"""Applies a linear combination to the incoming capsules

     Args:
         out_capsules (int): number of output capsules
         in_length (int): length of each input capsule
         out_length (int): length of each output capsule
         in_capsules (int, optional): number of input capsules
         share_weight (bool, optional): if True, share weight between input capsules
         routing_type (str, optional): routing algorithm type
            -- options: ['dynamic', 'k_means']
         num_iterations (int, optional): number of routing iterations
         squash (bool, optional): squash output capsules or not, it works for all routing
         kwargs (dict, optional): other args:
            - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
                -- options: ['dot', 'cosine', 'tonimoto', 'pearson']

     Shape:
         - Input: (Tensor): (N, in_capsules, in_length)
         - Output: (Tensor): (N, out_capsules, out_length)

     Attributes:
         if share_weight:
            - weight (Tensor): the learnable weights of the module of shape
              (out_capsules, out_length, in_length)
        else:
            -  weight (Tensor): the learnable weights of the module of shape
              (out_capsules, in_capsules, out_length, in_length)

     Examples::
         >>> import torch
         >>> from capsule_layer import CapsuleLinear
         >>> m = CapsuleLinear(3, 4, 5, 6, share_weight=False, routing_type='dynamic', num_iterations=50)
         >>> input = torch.rand(2, 6, 4)
         >>> output, prob = m(input)
         >>> print(output.size())
         torch.Size([2, 3, 5])
         >>> print(prob.size())
         torch.Size([2, 3, 6])
     """

    def __init__(self, out_capsules, in_length, out_length, in_capsules=None, share_weight=True,
                 routing_type='k_means', num_iterations=3, squash=False, **kwargs):
        super(CapsuleLinear, self).__init__()
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

        self.out_capsules = out_capsules
        self.in_capsules = in_capsules
        self.share_weight = share_weight
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.squash = squash
        self.kwargs = kwargs

        if self.share_weight:
            if in_capsules is not None:
                raise ValueError('Expected in_capsules must be None.')
            else:
                self.weight = Parameter(torch.Tensor(out_capsules, out_length, in_length))
        else:
            if in_capsules is None:
                raise ValueError('Expected in_capsules must be int.')
            else:
                self.weight = Parameter(torch.Tensor(out_capsules, in_capsules, out_length, in_length))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return CL.capsule_linear(input, self.weight, self.share_weight, self.routing_type, self.num_iterations,
                                 self.squash, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'
