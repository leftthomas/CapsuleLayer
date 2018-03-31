import torch
import torch.nn.functional as F


def capsule_cov2d(input, weight, stride=1, padding=0, routing_type='dynamic', num_iterations=3, **kwargs):
    if input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if weight.dim() != 6:
        raise ValueError('Expected 6D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if input.data.type() != weight.data.type():
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'input tensor, {} in weight tensor instead.'.format(input.data.type(), weight.data.type()))
    if not input.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if input.size(1) % weight.size(-1) != 0:
        raise ValueError('Expected in_channels must be divisible by in_length.')
    if input.size(1) != (weight.size(1) * weight.size(-1)):
        raise ValueError('Expected input tensor has the same in_channels as weight tensor, got in_channels {} in input '
                         'tensor, in_channels {} in weight tensor.'.format(input.size(-1),
                                                                           weight.size(1) * weight.size(-1)))
    # TODO
    # two method
    # 1. softmax between lower layer capsules, sum the prob of capsule_i to 1
    # 2. softmax between higher layer capsules, sum the prob of capsule_j to 1
    raise NotImplementedError('CapsuleConv2d is not implemented.')


def capsule_linear(input, weight, share_weight=True, routing_type='dynamic', num_iterations=3, **kwargs):
    if input.dim() != 3:
        raise ValueError('Expected 3D tensor as input, got {}D tensor instead.'.format(input.dim()))
    if share_weight and (weight.dim() != 3):
        raise ValueError('Expected 3D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if (not share_weight) and (weight.dim() != 4):
        raise ValueError('Expected 4D tensor as weight, got {}D tensor instead.'.format(weight.dim()))
    if input.data.type() != weight.data.type():
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'input tensor, {} in weight tensor instead.'.format(input.data.type(), weight.data.type()))
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
    if share_weight:
        # [batch_size, out_capsules, in_capsules, out_length]
        priors = (weight[None, :, None, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    else:
        priors = (weight[None, :, :, :, :] @ input[:, None, :, :, None]).squeeze(dim=-1)
    if routing_type == 'dynamic':
        # [batch_size, out_capsules, out_length]
        out = dynamic_routing(priors, num_iterations, **kwargs)
    elif routing_type == 'k_means':
        out = k_means_routing(priors, num_iterations, **kwargs)
    elif routing_type == 'db_scan':
        out = db_scan_routing(priors, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))
    return out


def dynamic_routing(input, num_iterations=3, cum=False, squash=True):
    logits = torch.zeros_like(input)
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=1)
        output = (probs * input).sum(dim=-2, keepdim=True)
        if r != num_iterations - 1:
            output = flaser(output)
            if cum:
                logits = logits + (input * output).sum(dim=-1, keepdim=True)
            else:
                logits = (input * output).sum(dim=-1, keepdim=True)
    if squash:
        return flaser(output).squeeze(dim=-2)
    else:
        return output.squeeze(dim=-2)


def k_means_routing(input, num_iterations=3, similarity='cosine', squash=True):
    output = input.mean(dim=-2, keepdim=True)
    for r in range(num_iterations):
        if similarity == 'cosine':
            logits = F.cosine_similarity(input, output, dim=-1).unsqueeze(dim=-1)
        elif similarity == 'tonimoto':
            logits = tonimoto_similarity(input, output)
        elif similarity == 'pearson':
            logits = pearson_similarity(input, output)
        else:
            raise NotImplementedError(
                '{} similarity is not implemented on k-means routing algorithm.'.format(similarity))
        probs = F.softmax(logits, dim=1)
        output = (probs * input).sum(dim=-2, keepdim=True)
    if squash:
        return flaser(output).squeeze(dim=-2)
    else:
        return output.squeeze(dim=-2)


def db_scan_routing(input, num_iterations=3, distance='euclidean', squash=False):
    # TODO
    raise NotImplementedError('DB SCAN routing algorithm is not implemented.')


def tonimoto_similarity(x1, x2, dim=-1, eps=1e-8):
    x1_norm = x1.norm(p=2, dim=dim, keepdim=True)
    x2_norm = x2.norm(p=2, dim=dim, keepdim=True)
    dot_value = (x1 * x2).sum(dim=dim, keepdim=True)
    return dot_value / (x1_norm ** 2 + x2_norm ** 2 - dot_value).clamp(min=eps)


def pearson_similarity(x1, x2, dim=-1, eps=1e-8):
    centered_x1 = x1 - x1.mean(dim=dim, keepdim=True)
    centered_x2 = x2 - x2.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(centered_x1, centered_x2, dim=dim, eps=eps).unsqueeze(dim=-1)


def flaser(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (0.5 + norm ** 2)
    return scale * input

