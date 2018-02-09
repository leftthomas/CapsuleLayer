import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair


def capsule_conv2d_cpu(input, weight, stride, padding, with_routing, num_iterations):
    if input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))
    if input.is_cuda or weight.is_cuda:
        raise ValueError("Expected input tensor and weight tensor should be in cpu, got gpu tensor instead.")
    kernel_size = (weight.size(2), weight.size(3))
    in_length = weight.size(4)
    stride = _pair(stride)
    padding = _pair(padding)
    num_iterations = num_iterations
    N, C_in, H_in, W_in = input.size()
    H_out = 1 + (H_in + 2 * padding[0] - kernel_size[0]) // stride[0]
    W_out = 1 + (W_in + 2 * padding[1] - kernel_size[1]) // stride[1]

    # it could be optimized, because it require many memory,
    # and the matrix multiplication also could be optimized to speed up
    input = F.pad(input, (padding[1], padding[1], padding[0], padding[0]))
    # [batch_size, num_plane, num_height_kernel, num_width_kernel, length_capsule, kernel_size[0], kernel_size[1]]
    input_windows = input.unfold(1, in_length, in_length). \
        unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1).transpose(-1, -2)

    weight = weight.view(*weight.size()[:2], -1, *weight.size()[-2:])
    # [batch_size, num_out_plane, num_in_plane, num_height_kernel, num_width_kernel, kernel_size[0]*kernel_size[1], 1,
    # length_capsule]
    priors = input_windows[:, None, :, :, :, :, None, :] @ weight[None, :, :, None, None, :, :, :]

    # [batch_size, num_out_plane, num_height_kernel, num_width_kernel, length_capsule]
    if with_routing:
        out = route_conv2d(priors, num_iterations)
    else:
        out = priors.sum(dim=-3, keepdim=True).sum(dim=2, keepdim=True).squeeze(dim=-2).squeeze(dim=-2).squeeze(
            dim=2)
    out = out.view(*out.size()[:2], -1, out.size(-1)).transpose(-1, -2)
    out = out.contiguous().view(out.size(0), -1, H_out, W_out)
    return out


def capsule_linear_cpu(input, weight, with_routing, num_iterations):
    if input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))
    if input.is_cuda or weight.is_cuda:
        raise ValueError("Expected input tensor and weight tensor should be in cpu, got gpu tensor instead.")
    weight = weight.transpose(1, 2)
    priors = (weight[:, None, :, :, :] @ input[None, :, :, :, None]).squeeze(dim=-1)
    if with_routing:
        out = route_linear(priors, num_iterations)
    else:
        out = priors.sum(dim=2, keepdim=True).squeeze(dim=-2).transpose(0, 1)
    return out


def route_conv2d(input, num_iterations):
    logits = Variable(torch.zeros(*input.size())).type_as(input)
    outputs = None
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=-3)
        outputs = squash((probs * input).sum(dim=-3, keepdim=True).sum(dim=2, keepdim=True))
        if r != num_iterations - 1:
            delta_logits = (input * outputs).sum(dim=-1, keepdim=True)
            logits = logits + delta_logits
    return outputs.squeeze(dim=-2).squeeze(dim=-2).squeeze(dim=2)


def route_linear(input, num_iterations):
    logits = Variable(torch.zeros(*input.size())).type_as(input)
    outputs = None
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=2)
        outputs = squash((probs * input).sum(dim=2, keepdim=True))
        if r != num_iterations - 1:
            delta_logits = (input * outputs).sum(dim=-1, keepdim=True)
            logits = logits + delta_logits
    return outputs.squeeze(dim=-2).transpose(0, 1)


def squash(tensor, dim=-1):
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * tensor
