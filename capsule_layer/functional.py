import torch
import torch.nn.functional as F


def capsule_cov2d(input, weight, out_length, stride=1, padding=0):
    if input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if input.data.type() != weight.data.type():
        raise ValueError('Expected input and weight tensor should be the same type, got different type instead.')
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if (weight.size(1) != 1) and (input.size(1) / weight.size(1) != weight.size(0) / out_length):
        raise ValueError('Expected input and output tensor should be the same group, got different group instead.')
    if input.size(1) == 1:
        out = F.conv2d(input, weight, stride=stride, padding=padding, groups=weight.size(0) / out_length)
    else:
        out = F.conv2d(input, weight, stride=stride, padding=padding, groups=weight.size(0) / out_length)
    return out


def capsule_linear(input, weight, routing_type='sum', **kwargs):
    if input.dim() != 3:
        raise ValueError('Expected 3D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if input.data.type() != weight.data.type():
        raise ValueError('Expected input and weight tensor should be the same type, got different type instead.')
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if input.size(1) != weight.size(1):
        raise ValueError('Expected input tensor has the same in_capsules as weight, got {} '
                         'in_capsules in input tensor, {} in_capsules in weight.'.format(input.size(1), weight.size(1)))
    if input.size(-1) != weight.size(-1):
        raise ValueError('Expected input tensor has the same in_length as weight, got in_length {} '
                         'in input tensor, in_length {} in weight.'.format(input.size(-1), weight.size(-1)))
    # [batch_size, out_capsules, in_capsules, out_length]
    priors = (weight[None, :, :, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    if routing_type == 'sum':
        out = priors.sum(dim=-2)
    elif routing_type == 'dynamic':
        out = dynamic_route_linear(priors, **kwargs)
    elif routing_type == 'means':
        out = means_route_linear(priors, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))
    return out


def dynamic_route_linear(input, num_iterations=3):
    if num_iterations < 1:
        raise ValueError('Expected num_iterations should greater than 1 or '
                         'equal to 1, got num_iterations {} instead.'.format(num_iterations))
    else:
        logits = torch.zeros_like(input)
        for r in range(num_iterations):
            probs = F.softmax(logits, dim=-2)
            output = squash((probs * input).sum(dim=-2, keepdim=True))
            if r != num_iterations - 1:
                logits = (input * output).sum(dim=-1, keepdim=True)
        return output.squeeze(dim=-2)


def means_route_linear(input, num_iterations=3):
    if num_iterations < 0:
        raise ValueError('Expected num_iterations should greater than 0 or '
                         'equal to 0, got num_iterations {} instead.'.format(num_iterations))
    else:
        output = input.mean(dim=-2, keepdim=True)
        for r in range(num_iterations):
            output = F.normalize(output, p=2, dim=-1)
            logits = (input * output).sum(dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-2)
            output = (probs * input).sum(dim=-2, keepdim=True)
        return squash(output).squeeze(dim=-2)


def squash(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input
