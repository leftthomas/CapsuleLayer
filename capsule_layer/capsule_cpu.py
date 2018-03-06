import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn.modules.utils import _pair


def capsule_conv2d_cpu(input, weight, stride, padding, routing_type, **kwargs):
    if input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if input.is_cuda:
        raise ValueError('Expected input tensor should be in cpu, got gpu tensor instead.')
    if weight.is_cuda:
        raise ValueError('Expected weight tensor should be in cpu, got gpu tensor instead.')
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    kernel_size = (weight.size(2), weight.size(3))
    in_length = weight.size(4)
    stride = _pair(stride)
    padding = _pair(padding)
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
    if routing_type == 'sum':
        out = priors.sum(dim=-3, keepdim=True).sum(dim=2, keepdim=True).squeeze(dim=-2).squeeze(dim=-2).squeeze(dim=2)
    elif routing_type == 'dynamic':
        out = dynamic_route_conv2d(priors, **kwargs)
    else:
        # TODO
        raise NotImplementedError('{} routing algorithm is not implemented on cpu.'.format(routing_type))
    out = out.view(*out.size()[:2], -1, out.size(-1)).transpose(-1, -2)
    out = out.contiguous().view(out.size(0), -1, H_out, W_out)
    return out


def capsule_linear_cpu(input, weight, routing_type, **kwargs):
    if input.dim() != 3:
        raise ValueError('Expected 3D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if input.is_cuda:
        raise ValueError('Expected input tensor should be in cpu, got gpu tensor instead.')
    if weight.is_cuda:
        raise ValueError('Expected weight tensor should be in cpu, got gpu tensor instead.')
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    weight = weight.transpose(1, 2)
    priors = (weight[:, None, :, :, :] @ input[None, :, :, :, None]).squeeze(dim=-1)
    if routing_type == 'sum':
        out = priors.sum(dim=2, keepdim=True).squeeze(dim=-2).transpose(0, 1)
    elif routing_type == 'dynamic':
        out = dynamic_route_linear(priors, **kwargs)
    else:
        # TODO
        raise NotImplementedError('{} routing algorithm is not implemented on cpu.'.format(routing_type))
    return out


def dynamic_route_conv2d(input, num_iterations=3):
    logits = torch.zeros_like(input)
    outputs = None
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=-3)
        outputs = squash((probs * input).sum(dim=-3, keepdim=True).sum(dim=2, keepdim=True))
        if r != num_iterations - 1:
            logits = (input * outputs).sum(dim=-1, keepdim=True)
    return outputs.squeeze(dim=-2).squeeze(dim=-2).squeeze(dim=2)


def dynamic_route_linear(input, num_iterations=3):
    logits = torch.zeros_like(input)
    outputs = None
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=2)
        outputs = squash((probs * input).sum(dim=2, keepdim=True))
        if r != num_iterations - 1:
            logits = (input * outputs).sum(dim=-1, keepdim=True)
    return outputs.squeeze(dim=-2).transpose(0, 1)


def means_route_conv2d(input, num_iterations=3):
    outputs = input.mean(dim=-2, keepdim=True).mean(dim=-3, keepdim=True)
    for r in range(num_iterations):
        norm = outputs.norm(p=2, dim=-1, keepdim=True)
        outputs = outputs / norm
        logits = (input * outputs).sum(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-2)
        outputs = (probs * input).sum(dim=-2, keepdim=True).sum(dim=-3, keepdim=True)
    return squash(outputs).squeeze(dim=-2).squeeze(dim=-2).transpose(0, 1)


def means_route_linear(input, num_iterations=3):
    outputs = input.mean(dim=2, keepdim=True)
    for r in range(num_iterations):
        norm = outputs.norm(p=2, dim=-1, keepdim=True)
        outputs = outputs / norm
        logits = (input * outputs).sum(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=2)
        outputs = (probs * input).sum(dim=2, keepdim=True)
    return squash(outputs).squeeze(dim=-2).transpose(0, 1)


def em_route_conv2d(input, lambda_, a_, V, num_iterations=3):
    # routing coefficient
    batch_size = input.size(0)
    Cww = w * w * C
    Bkk = K * K * B
    R = Variable(torch.ones([batch_size, Bkk, Cww])) / Cww
    beta_v = nn.Parameter(torch.randn(1, Cww, 1))
    beta_a = nn.Parameter(torch.randn(1, Cww))
    for r in range(num_iterations):
        # M-step
        R = (R * a_)[..., None]
        sum_R = R.sum(dim=1)
        mu = ((R * V).sum(dim=1) / sum_R)[:, None, :, :]
        sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R

        # E-step
        if r != num_iterations - 1:
            mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
            normal = Normal(mu, sigma_square[:, None, :, :] ** (1 / 2))
            p = torch.exp(normal.log_prob(V_))
            ap = a__ * p.sum(-1)
            R = Variable(ap / ap.sum(dim=-1)[..., None])
        else:
            const = (beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
            a = torch.sigmoid(lambda_ * (beta_a.repeat(batch_size, 1) - const.sum(dim=2)))

    return a, mu


def em_route_linear(input, lambda_, a_, V, num_iterations=3):
    # routing coefficient
    batch_size = input.size(0)
    Cww = w * w * C
    Bkk = K * K * B
    R = Variable(torch.ones([batch_size, Bkk, Cww])) / Cww
    beta_v = nn.Parameter(torch.randn(1, Cww, 1))
    beta_a = nn.Parameter(torch.randn(1, Cww))
    for r in range(num_iterations):
        # M-step
        R = (R * a_)[..., None]
        sum_R = R.sum(dim=1)
        mu = ((R * V).sum(dim=1) / sum_R)[:, None, :, :]
        sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R

        # E-step
        if r != num_iterations - 1:
            mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
            normal = Normal(mu, sigma_square[:, None, :, :] ** (1 / 2))
            p = torch.exp(normal.log_prob(V_))
            ap = a__ * p.sum(-1)
            R = Variable(ap / ap.sum(dim=-1)[..., None])
        else:
            const = (beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
            a = torch.sigmoid(lambda_ * (beta_a.repeat(batch_size, 1) - const.sum(dim=2)))

    return a, mu


def squash(tensor, dim=-1):
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * tensor
