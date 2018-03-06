import torch
import torch.nn.functional as F
from torch.autograd import Variable


def means_route_linear(input, num_iterations=3):
    outputs = input.mean(dim=-2, keepdim=True)
    for r in range(num_iterations):
        norm = outputs.norm(p=2, dim=-1, keepdim=True)
        outputs = outputs / norm
        logits = (input * outputs).sum(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-2)
        outputs = (probs * input).sum(dim=-2, keepdim=True)
    print(probs)
    return squash(outputs).squeeze(dim=-2)


def squash(tensor, dim=-1):
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * tensor


if __name__ == '__main__':
    # out_capsules, in_capsules, out_length
    a = Variable(torch.randn(2, 5, 6))
    b = a
    means_route_linear(a, num_iterations=20)
    means_route_linear(b, num_iterations=100)
