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
        routing_type (str, optional): routing algorithm type
           -- options: ['dynamic', 'k_means']
        num_iterations (int, optional): number of routing iterations
        dropout (float, optional): if non-zero, introduces a dropout layer on the inputs
        bias (bool, optional):  if True, adds a learnable bias to the output
        kwargs (dict, optional): other args:
           - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
               -- options: ['dot', 'cosine', 'tonimoto', 'pearson']
           - squash (bool, optional): squash output capsules or not, it works for all routing
           - return_prob (bool, optional): return output capsules' prob or not, it works for all routing
           - softmax_dim (int, optional): specify the softmax dim between capsules, it works for all routing

    Shape:
        - Input: (Tensor): (N, C_{in}, H_{in}, W_{in})
        - Output: (Tensor): (N, C_{out}, H_{out}, W_{out}) where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel_size[0] -1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel_size[1] -1) - 1) / stride[1] + 1)`

    Attributes:
        - weight (Tensor): the learnable weights of the module of shape
           (out_channels // out_length, out_length, in_length, kernel_size[0], kernel_size[1])
        - bias (Tensor): the learnable bias of the module of shape
           (out_channels // out_length, out_length)

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
        >>> input = torch.randn(20, 16, 20, 50)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([20, 24, 9, 24])
        >>> input = torch.randn(10, 3, 14, 25)
        >>> output = m1(input)
        >>> print(output.size())
        torch.Size([10, 16, 10, 25])
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_length, out_length, stride=1, padding=0, dilation=1,
                 routing_type='k_means', num_iterations=3, dropout=0, bias=True, **kwargs):
        super(CapsuleConv2d, self).__init__()
        if in_channels % in_length != 0:
            raise ValueError('Expected in_channels must be divisible by in_length.')
        if out_channels % out_length != 0:
            raise ValueError('Expected out_channels must be divisible by out_length.')
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}'.format(num_iterations))
        if dropout < 0 or dropout > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(dropout))
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
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.dropout = dropout
        self.kwargs = kwargs
        self.weight = Parameter(torch.Tensor(out_channels // out_length, out_length, in_length, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels // out_length, out_length))
            nn.init.xavier_uniform_(self.bias)
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return CL.capsule_cov2d(input, self.weight, self.stride, self.padding, self.dilation, self.routing_type,
                                self.num_iterations, self.dropout, self.bias, self.training, **self.kwargs)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', in_length={in_length}, out_length={out_length}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CapsuleConvTranspose2d(nn.Module):
    r"""Applies a 2D capsule transposed convolution over an input signal composed of several input
    planes.

    This module can be seen as the gradient of capsule Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation`, :attr:`output_padding` can
    either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the capsule transposed convolution
        kernel_size (int or tuple): size of the capsule convolving kernel
        in_length (int): length of each input capsule
        out_length (int): length of each output capsule
        stride (int or tuple, optional): stride of the capsule convolution
        padding (int or tuple, optional): zero-padding added to both sides of the input
        output_padding (int or tuple, optional): additional size added to one side of each dimension in the output shape
        dilation (int or tuple, optional): spacing between kernel elements
        routing_type (str, optional): routing algorithm type
           -- options: ['dynamic', 'k_means']
        num_iterations (int, optional): number of routing iterations
        dropout (float, optional): if non-zero, introduces a dropout layer on the inputs
        bias (bool, optional):  if True, adds a learnable bias to the output
        kwargs (dict, optional): other args:
           - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
               -- options: ['dot', 'cosine', 'tonimoto', 'pearson']
           - squash (bool, optional): squash output capsules or not, it works for all routing
           - return_prob (bool, optional): return output capsules' prob or not, it works for all routing
           - softmax_dim (int, optional): specify the softmax dim between capsules, it works for all routing

    Shape:
        - Input: (Tensor): (N, C_{in}, H_{in}, W_{in})
        - Output: (Tensor): (N, C_{out}, H_{out}, W_{out}) where
          :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] -1) + 1
            + output_padding[0]`
          :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] -1) + 1
            + output_padding[1]`

    Attributes:
        - weight (Tensor): the learnable weights of the module of shape
           (in_length, out_channels // out_length, out_length, kernel_size[0], kernel_size[1])
        - bias (Tensor): the learnable bias of the module of shape
           (out_channels // out_length, out_length)

    ------------------------------------------------------------------------------------------------
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        MAKE SURE THE CapsuleConvTranspose2d's OUTPUT CAPSULE's LENGTH EQUALS
                               THE NEXT CapsuleConvTranspose2d's INPUT CAPSULE's LENGTH
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ------------------------------------------------------------------------------------------------
    Examples::
        >>> import torch
        >>> from capsule_layer import CapsuleConvTranspose2d
        >>> # With square kernels and equal stride
        >>> m = CapsuleConvTranspose2d(16, 33, 3, 4, 3, stride=2, dilation=3, output_padding=2, padding=1)
        >>> # non-square kernels and unequal stride and with padding
        >>> m1 = CapsuleConvTranspose2d(16, 33, (3, 5), 4, 3, stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> output.size()
        torch.Size([20, 33, 105, 205])
        >>> input = torch.randn(20, 16, 25, 50)
        >>> output = m1(input)
        >>> output.size()
        torch.Size([20, 33, 43, 50])
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = CapsuleConv2d(16, 16, 3, 4, 2, stride=2, padding=1)
        >>> upsample = CapsuleConvTranspose2d(16, 16, 3, 2, 4, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_length, out_length, stride=1, padding=0,
                 output_padding=0, dilation=1, routing_type='k_means', num_iterations=3, dropout=0, bias=True,
                 **kwargs):
        super(CapsuleConvTranspose2d, self).__init__()
        if in_channels % in_length != 0:
            raise ValueError('Expected in_channels must be divisible by in_length.')
        if out_channels % out_length != 0:
            raise ValueError('Expected out_channels must be divisible by out_length.')
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}'.format(num_iterations))
        if dropout < 0 or dropout > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(dropout))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_length = in_length
        self.out_length = out_length
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.dropout = dropout
        self.kwargs = kwargs
        self.weight = Parameter(torch.Tensor(in_length, out_channels // out_length, out_length, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels // out_length, out_length))
            nn.init.xavier_uniform_(self.bias)
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError('output_size must have {} or {} elements (got {})'.format(k, k + 2, len(output_size)))

        def dim_size(d):
            return (input.size(d + 2) - 1) * self.stride[d] - 2 * self.padding[d] + self.dilation[d] * (
                    self.kernel_size[d] - 1) + 1

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError(
                    'requested an output size of {}, but valid sizes range from {} to {} (for an input of {})'.format(
                        output_size, min_sizes, max_sizes, input.size()[-2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])

    def forward(self, input, output_size=None):
        self.output_padding = self._output_padding(input, output_size)
        return CL.capsule_conv_transpose2d(input, self.weight, self.stride, self.padding, self.output_padding,
                                           self.dilation, self.routing_type, self.num_iterations, self.dropout,
                                           self.bias, self.training, **self.kwargs)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', in_length={in_length}, out_length={out_length}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
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
         dropout (float, optional): if non-zero, introduces a dropout layer on the inputs
         bias (bool, optional):  if True, adds a learnable bias to the output
         kwargs (dict, optional): other args:
            - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
                -- options: ['dot', 'cosine', 'tonimoto', 'pearson']
            - squash (bool, optional): squash output capsules or not, it works for all routing
            - return_prob (bool, optional): return output capsules' prob or not, it works for all routing
            - softmax_dim (int, optional): specify the softmax dim between capsules, it works for all routing

     Shape:
         - Input: (Tensor): (N, in_capsules, in_length)
         - Output: (Tensor): (N, out_capsules, out_length)

     Attributes:
         if share_weight:
            - weight (Tensor): the learnable weights of the module of shape
              (out_capsules, out_length, in_length)
            - bias (Tensor): the learnable bias of the module of shape
              (out_capsules, out_length)
        else:
            -  weight (Tensor): the learnable weights of the module of shape
              (out_capsules, in_capsules, out_length, in_length)
            - bias (Tensor): the learnable bias of the module of shape
              (out_capsules, out_length)

     Examples::
         >>> import torch
         >>> from capsule_layer import CapsuleLinear
         >>> m = CapsuleLinear(30, 8, 16, 20, share_weight=False, routing_type = 'dynamic', num_iterations=5)
         >>> input = torch.randn(5, 20, 8)
         >>> output = m(input)
         >>> print(output.size())
         torch.Size([5, 30, 16])
     """

    def __init__(self, out_capsules, in_length, out_length, in_capsules=None, share_weight=True,
                 routing_type='k_means', num_iterations=3, dropout=0, bias=True, **kwargs):
        super(CapsuleLinear, self).__init__()
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}'.format(num_iterations))
        if dropout < 0 or dropout > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(dropout))

        self.out_capsules = out_capsules
        self.in_capsules = in_capsules
        self.share_weight = share_weight
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.dropout = dropout
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
        if bias:
            self.bias = Parameter(torch.Tensor(out_capsules, out_length))
            nn.init.xavier_uniform_(self.bias)
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return CL.capsule_linear(input, self.weight, self.share_weight, self.routing_type, self.num_iterations,
                                 self.dropout, self.bias, self.training, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'
