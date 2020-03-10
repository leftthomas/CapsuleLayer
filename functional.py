import math

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def capsule_cov2d(input, weight, stride=1, padding=0, dilation=1, share_weight=True, routing_type='k_means',
                  num_iterations=3, squash=True, **kwargs):
    if input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if share_weight and (weight.dim() != 5):
        raise ValueError('Expected 5D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if (not share_weight) and (weight.dim() != 6):
        raise ValueError('Expected 6D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if input.type() != weight.type():
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'input tensor, {} in weight tensor instead.'.format(input.type(), weight.type()))
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if input.size(1) % weight.size(-3) != 0:
        raise ValueError('Expected in_channels must be divisible by in_length.')
    if not share_weight and input.size(1) != (weight.size(1) * weight.size(3)):
        raise ValueError('Expected input tensor has the same in_channels as weight tensor, got {} in_channels '
                         'in input tensor, {} in_channels in weight tensor.'.format(input.size(1),
                                                                                    weight.size(1) * weight.size(3)))
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

    kernel_size = (weight.size(-2), weight.size(-1))
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    out_h = math.floor((input.size(2) + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((input.size(3) + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    inp = F.unfold(input, weight.size()[-2:], dilation, padding, stride)
    # [batch_size, in_group, in_length, kernel_size[0], kernel_size[1], out_height, out_width]
    inp = inp.view(input.size(0), input.size(1) // weight.size(-3), *weight.size()[-3:], out_h, out_w)
    # [batch_size, out_height, out_width, in_group, kernel_size[0], kernel_size[1], in_length]
    inp = inp.permute(0, 5, 6, 1, 3, 4, 2).contiguous()

    if share_weight:
        weight = weight.permute(0, 3, 4, 1, 2).contiguous()
        # [batch_size, out_height, out_width, in_group, out_group, kernel_size[0], kernel_size[1], out_length]
        priors = torch.matmul(weight.view(1, 1, 1, 1, *weight.size()), inp.unsqueeze(dim=4).unsqueeze(dim=-1)) \
            .squeeze(dim=-1)
    else:
        weight = weight.permute(1, 0, 4, 5, 2, 3).contiguous()
        priors = torch.matmul(weight.view(1, 1, 1, *weight.size()), inp.unsqueeze(dim=4).unsqueeze(dim=-1)) \
            .squeeze(dim=-1)

    # [batch_size, out_height, out_width, in_group, out_group, kernel_size[0]*kernel_size[1], out_length]
    priors = priors.view(*priors.size()[:5], -1, priors.size(-1))

    if routing_type == 'dynamic':
        # [batch_size, out_height, out_width, in_group, out_group, out_length]
        # [batch_size, out_height, out_width, in_group, out_group, kernel_size[0]*kernel_size[1]]
        out, probs = dynamic_routing(priors, num_iterations)
    elif routing_type == 'k_means':
        out, probs = k_means_routing(priors, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))

    out = out.sum(dim=-3)
    out = _squash(out) if squash is True else out
    # [batch_size, out_height, out_width, out_channels]
    out = out.view(*out.size()[:3], -1)
    # [batch_size, out_channels, out_height, out_width]
    out = out.permute(0, 3, 1, 2).contiguous()
    # [batch_size, out_height, out_width, in_group, out_group, kernel_size[0], kernel_size[1]]
    probs = probs.view(*probs.size()[:5], kernel_size[0], kernel_size[1])

    return out, probs


def capsule_linear(input, weight, share_weight=True, routing_type='k_means', num_iterations=3, squash=True, **kwargs):
    if input.dim() != 3:
        raise ValueError('Expected 3D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if share_weight and (weight.dim() != 3):
        raise ValueError('Expected 3D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if (not share_weight) and (weight.dim() != 4):
        raise ValueError('Expected 4D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if input.type() != weight.type():
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'input tensor, {} in weight tensor instead.'.format(input.type(), weight.type()))
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if (not share_weight) and (input.size(1) != weight.size(1)):
        raise ValueError('Expected input tensor has the same in_capsules as weight tensor, got {} in_capsules '
                         'in input tensor, {} in_capsules in weight tensor.'.format(input.size(1), weight.size(1)))
    if input.size(-1) != weight.size(-1):
        raise ValueError('Expected input tensor has the same in_length as weight tensor, got in_length {} '
                         'in input tensor, in_length {} in weight tensor.'.format(input.size(-1), weight.size(-1)))
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

    if share_weight:
        # [batch_size, out_capsules, in_capsules, out_length]
        priors = torch.matmul(weight.unsqueeze(dim=1).unsqueeze(dim=0), input.unsqueeze(dim=1).unsqueeze(dim=-1)) \
            .squeeze(dim=-1)
    else:
        priors = torch.matmul(weight.unsqueeze(dim=0), input.unsqueeze(dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)

    if routing_type == 'dynamic':
        # [batch_size, out_capsules, out_length], [batch_size, out_capsules, in_capsules]
        out, probs = dynamic_routing(priors, num_iterations)
    elif routing_type == 'k_means':
        out, probs = k_means_routing(priors, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))

    out = _squash(out) if squash is True else out
    return out, probs


def dynamic_routing(input, num_iterations=3):
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    logits = torch.zeros_like(input)
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=-3)
        output = (probs * input).sum(dim=-2, keepdim=True)
        if r != num_iterations - 1:
            output = _squash(output)
            logits = logits + (input * output).sum(dim=-1, keepdim=True)
    return output.squeeze(dim=-2), probs.mean(dim=-1)


def k_means_routing(input, num_iterations=3, similarity='dot'):
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    output = input.sum(dim=-2, keepdim=True) / input.size(-3)
    for r in range(num_iterations):
        if similarity == 'dot':
            logits = (input * F.normalize(output, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        elif similarity == 'cosine':
            logits = F.cosine_similarity(input, output, dim=-1).unsqueeze(dim=-1)
        elif similarity == 'tonimoto':
            logits = tonimoto_similarity(input, output)
        elif similarity == 'pearson':
            logits = pearson_similarity(input, output)
        else:
            raise NotImplementedError(
                '{} similarity is not implemented on k-means routing algorithm.'.format(similarity))
        probs = F.softmax(logits, dim=-3)
        output = (probs * input).sum(dim=-2, keepdim=True)
    return output.squeeze(dim=-2), probs.squeeze(dim=-1)


def tonimoto_similarity(x1, x2, dim=-1, eps=1e-8):
    x1_norm = x1.norm(p=2, dim=dim, keepdim=True)
    x2_norm = x2.norm(p=2, dim=dim, keepdim=True)
    dot_value = (x1 * x2).sum(dim=dim, keepdim=True)
    return dot_value / (x1_norm ** 2 + x2_norm ** 2 - dot_value).clamp(min=eps)


def pearson_similarity(x1, x2, dim=-1, eps=1e-8):
    centered_x1 = x1 - x1.mean(dim=dim, keepdim=True)
    centered_x2 = x2 - x2.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(centered_x1, centered_x2, dim=dim, eps=eps).unsqueeze(dim=dim)


def _squash(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input
