import math

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def capsule_cov2d(input, weight, stride=1, padding=0, dilation=1, share_weight=True, routing_type='k_means',
                  num_iterations=3, bias=None, share_type='bottom', out_channels=None, **kwargs):
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
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    if share_type not in ['bottom', 'top']:
        raise NotImplementedError('{} share type is not implemented.'.format(share_type))
    if share_weight and share_type == 'top' and input.size(1) != (weight.size(0) * weight.size(2)):
        raise ValueError('Expected input tensor has the same in_channels as weight tensor, got {} in_channels '
                         'in input tensor, {} in_channels in weight tensor.'.format(input.size(1),
                                                                                    weight.size(0) * weight.size(2)))
    if not share_weight and input.size(1) != (weight.size(1) * weight.size(3)):
        raise ValueError('Expected input tensor has the same in_channels as weight tensor, got {} in_channels '
                         'in input tensor, {} in_channels in weight tensor.'.format(input.size(1),
                                                                                    weight.size(1) * weight.size(3)))
    if share_weight and share_type == 'top' and out_channels is None:
        raise ValueError('Expected out_channels must be int.')
    if bias is not None:
        if bias.dim() != 2:
            raise ValueError('Expected 2D tensor as bias, got {}D tensor instead.'.format(bias.dim()))
        if share_weight and share_type == 'top' and (bias.size(0) * bias.size(1)) != out_channels:
            raise ValueError('Expected bias tensor has the same out_channels as out_channels, got {} out_channels '
                             'in bias tensor, {} out_channels in out_channels.'.format(bias.size(0) * bias.size(1),
                                                                                       out_channels))
        if share_weight and share_type == 'bottom' and (bias.size(0) * bias.size(1)) != (
                weight.size(0) * weight.size(1)):
            raise ValueError('Expected bias tensor has the same out_channels as weight tensor, got {} out_channels '
                             'in bias tensor, {} out_channels in weight tensor.'.format(bias.size(0) * bias.size(1),
                                                                                        weight.size(0) * weight.size(
                                                                                            1)))
        if not share_weight and (bias.size(0) * bias.size(1)) != (weight.size(0) * weight.size(2)):
            raise ValueError('Expected bias tensor has the same out_channels as weight tensor, got {} out_channels '
                             'in bias tensor, {} out_channels in weight tensor.'.format(bias.size(0) * bias.size(1),
                                                                                        weight.size(0) * weight.size(
                                                                                            2)))
        if bias.size(-1) != weight.size(-4):
            raise ValueError('Expected bias tensor has the same out_length as weight tensor, got {} out_length '
                             'in bias tensor, {} out_length in weight tensor.'.format(bias.size(-1), weight.size(-4)))

    kernel_size = (weight.size(-2), weight.size(-1))
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    inp = F.unfold(input, weight.size()[-2:], dilation, padding, stride)
    inp = inp.view(input.size(0), input.size(1) // weight.size(-3), *weight.size()[-3:], -1)
    # [batch_size, in_capsules, block, kernel_size[0], kernel_size[1], in_length]
    inp = inp.permute(0, 1, 5, 3, 4, 2).contiguous()

    if share_weight and share_type == 'bottom':
        weight = weight.permute(0, 3, 4, 1, 2).contiguous()
        # [batch_size, out_capsules, in_capsules, block, kernel_size[0], kernel_size[1], out_length]
        priors = (weight[None, :, None, None, :, :, :, :] @ inp[:, None, :, :, :, :, :, None]).squeeze(dim=-1)
    elif share_weight and share_type == 'top':
        weight = weight.permute(0, 3, 4, 1, 2).contiguous()
        priors = (weight[None, None, :, None, :, :, :, :] @ inp[:, None, :, :, :, :, :, None]).squeeze(dim=-1)
        priors = priors.expand(priors.size(0), out_channels // priors.size(-1), *priors.size()[2:])
    else:
        weight = weight.permute(0, 1, 4, 5, 2, 3).contiguous()
        priors = (weight[None, :, :, None, :, :, :, :] @ inp[:, None, :, :, :, :, :, None]).squeeze(dim=-1)

    priors = priors.permute(0, 3, 1, 2, 4, 5, 6).contiguous()
    priors = priors.view(*priors.size()[:3], -1, priors.size(-1))
    out_h = math.floor((input.size(2) + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((input.size(3) + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    priors = priors.view(priors.size(0), out_h, out_w, *priors.size()[-3:])
    if bias is not None:
        bias = bias.view(1, 1, 1, 1, *bias.size())
        bias = bias.permute(0, 1, 2, 4, 3, 5).contiguous()

    if routing_type == 'dynamic':
        if kwargs.__contains__('return_prob') and kwargs['return_prob']:
            out, probs = dynamic_routing(priors, bias, num_iterations, **kwargs)
        else:
            # [batch_size, out_height, out_width, out_capsules, out_length]
            out = dynamic_routing(priors, bias, num_iterations, **kwargs)
    elif routing_type == 'k_means':
        if kwargs.__contains__('return_prob') and kwargs['return_prob']:
            out, probs = k_means_routing(priors, bias, num_iterations, **kwargs)
        else:
            out = k_means_routing(priors, bias, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))

    # [batch_size, out_height, out_width, out_channels]
    out = out.view(*out.size()[:3], -1)
    # [batch_size, out_channels, out_height, out_width]
    out = out.permute(0, 3, 1, 2).contiguous()
    if kwargs.__contains__('return_prob') and kwargs['return_prob']:
        return out, probs
    else:
        return out


def capsule_conv_transpose2d(input, weight, stride=1, padding=0, output_padding=0, dilation=1, routing_type='k_means',
                             num_iterations=3, bias=None, **kwargs):
    if input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if weight.dim() != 5:
        raise ValueError('Expected 5D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if input.type() != weight.type():
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'input tensor, {} in weight tensor instead.'.format(input.type(), weight.type()))
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if input.size(1) % weight.size(0) != 0:
        raise ValueError('Expected in_channels must be divisible by in_length.')
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

    batch_size = input.size(0)
    input = input.view(input.size(0), input.size(1) // weight.size(0), weight.size(0), *input.size()[-2:])
    input = input.view(-1, *input.size()[2:])
    priors = F.conv_transpose2d(input, weight.view(weight.size(0), -1, *weight.size()[-2:]), stride=stride,
                                padding=padding, output_padding=output_padding, dilation=dilation)
    # [batch_size, in_capsule_types, out_capsule_types, out_length, out_height, out_width]
    priors = priors.view(batch_size, priors.size(0) // batch_size, priors.size(1) // weight.size(2), -1,
                         *priors.size()[-2:])
    # [batch_size, out_height, out_width, out_capsule_types, in_capsule_types, out_length]
    priors = priors.permute(0, 4, 5, 2, 1, 3).contiguous()
    if bias is not None:
        bias = bias.view(1, 1, 1, *bias.size(), 1)
        bias = bias.permute(0, 1, 2, 3, 5, 4).contiguous()

    if routing_type == 'dynamic':
        # [batch_size, out_height, out_width, out_capsule_types, out_length]
        out = dynamic_routing(priors, bias, num_iterations, **kwargs)
    elif routing_type == 'k_means':
        out = k_means_routing(priors, bias, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))
    out = out.permute(0, 3, 4, 1, 2)
    # [batch_size, out_channels, out_height, out_width]
    out = out.contiguous().view(out.size(0), -1, *out.size()[-2:])
    return out


def capsule_linear(input, weight, share_weight=True, routing_type='k_means', num_iterations=3, bias=None,
                   share_type='bottom', out_capsules=None, **kwargs):
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
    if share_type not in ['bottom', 'top']:
        raise NotImplementedError('{} share type is not implemented.'.format(share_type))
    if share_weight and share_type == 'top' and input.size(1) != weight.size(0):
        raise ValueError('Expected input tensor has the same in_capsules as weight tensor, got {} in_capsules '
                         'in input tensor, {} in_capsules in weight tensor.'.format(input.size(1), weight.size(0)))
    if share_weight and share_type == 'top' and out_capsules is None:
        raise ValueError('Expected out_capsules must be int.')
    if bias is not None:
        if bias.dim() != 2:
            raise ValueError('Expected 2D tensor as bias, got {}D tensor instead.'.format(bias.dim()))
        if share_weight and share_type == 'top':
            if bias.size(0) != out_capsules:
                raise ValueError('Expected bias tensor has the same out_capsules as out_capsules, got {} out_capsules '
                                 'in bias tensor, {} out_capsules in out_capsules.'.format(bias.size(0), out_capsules))
        else:
            if bias.size(0) != weight.size(0):
                raise ValueError('Expected bias tensor has the same out_capsules as weight tensor, got {} out_capsules '
                                 'in bias tensor, {} out_capsules in weight tensor.'.format(bias.size(0),
                                                                                            weight.size(0)))
        if bias.size(-1) != weight.size(-2):
            raise ValueError('Expected bias tensor has the same out_length as weight tensor, got {} out_length '
                             'in bias tensor, {} out_length in weight tensor.'.format(bias.size(-1), weight.size(-2)))

    if share_weight and share_type == 'bottom':
        # [batch_size, out_capsules, in_capsules, out_length]
        priors = (weight[None, :, None, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    elif share_weight and share_type == 'top':
        priors = (weight[None, None, :, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
        priors = priors.expand(priors.size(0), out_capsules, *priors.size()[2:])
    else:
        priors = (weight[None, :, :, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    if bias is not None:
        bias = bias.view(1, *bias.size(), 1)
        bias = bias.permute(0, 1, 3, 2).contiguous()

    if routing_type == 'dynamic':
        # [batch_size, out_capsules, out_length]
        out = dynamic_routing(priors, bias, num_iterations, **kwargs)
    elif routing_type == 'k_means':
        out = k_means_routing(priors, bias, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))
    return out


def dynamic_routing(input, bias=None, num_iterations=3, squash=True, return_prob=False, softmax_dim=-3):
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    logits = torch.zeros_like(input)
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=softmax_dim)
        output = (probs * input).sum(dim=-2, keepdim=True)
        if bias is not None:
            output = output + bias
        if r != num_iterations - 1:
            output = _squash(output)
            logits = logits + (input * output).sum(dim=-1, keepdim=True)
    if squash:
        if return_prob:
            return _squash(output).squeeze(dim=-2), probs.mean(dim=-1)
        else:
            return _squash(output).squeeze(dim=-2)
    else:
        if return_prob:
            return output.squeeze(dim=-2), probs.mean(dim=-1)
        else:
            return output.squeeze(dim=-2)


def k_means_routing(input, bias=None, num_iterations=3, similarity='dot', squash=True, return_prob=False,
                    softmax_dim=-3):
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
        probs = F.softmax(logits, dim=softmax_dim)
        output = (probs * input).sum(dim=-2, keepdim=True)
        if bias is not None:
            output = output + bias
    if squash:
        if return_prob:
            return _squash(output).squeeze(dim=-2), probs.squeeze(dim=-1)
        else:
            return _squash(output).squeeze(dim=-2)
    else:
        if return_prob:
            return output.squeeze(dim=-2), probs.squeeze(dim=-1)
        else:
            return output.squeeze(dim=-2)


def tonimoto_similarity(x1, x2, dim=-1, eps=1e-8):
    x1_norm = x1.norm(p=2, dim=dim, keepdim=True)
    x2_norm = x2.norm(p=2, dim=dim, keepdim=True)
    dot_value = (x1 * x2).sum(dim=dim, keepdim=True)
    return dot_value / (x1_norm ** 2 + x2_norm ** 2 - dot_value).clamp(min=eps)


def pearson_similarity(x1, x2, dim=-1, eps=1e-8):
    centered_x1 = x1 - x1.mean(dim=dim, keepdim=True)
    centered_x2 = x2 - x2.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(centered_x1, centered_x2, dim=dim, eps=eps).unsqueeze(dim=-1)


def _squash(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input
