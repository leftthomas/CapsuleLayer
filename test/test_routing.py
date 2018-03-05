import torch
import torch.nn.functional as F
from torch.autograd import Variable


def dynamic_route_linear(input, num_iterations=3):
    logits = torch.zeros_like(input)
    outputs = None
    for r in range(num_iterations):
        probs = F.softmax(logits, dim=1)
        outputs = squash((probs * input).sum(dim=-1, keepdim=True))
        if r != num_iterations - 1:
            delta_logits = (input * outputs).sum(dim=-1, keepdim=True)
            logits = logits + delta_logits
    print(probs)
    return outputs.squeeze(dim=-2)


def squash(tensor, dim=-1):
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * tensor


if __name__ == '__main__':
    # out_capsules, in_capsules, out_length
    a = Variable(torch.randn(2, 5, 6))
    dynamic_route_linear(a, num_iterations=100)
