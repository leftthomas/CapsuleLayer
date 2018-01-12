import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.capsule import CapsuleConv2d, CapsuleLinear


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = CapsuleConv2d(in_channels=1, out_channels=32, kernel_size=3, in_length=1, out_length=8,
                                      stride=1,
                                      padding=1)
        self.classifier = CapsuleLinear(in_capsules=3 * 3 * 32 // 8, out_capsules=10, in_length=8, out_length=16)

    def forward(self, input):
        out = self.features(input)
        out = self.classifier(out)
        return out


model = Model()
x = torch.arange(1, 26).view(1, 1, 5, 5)
input = Variable(x)
print(input)
print(model(input))
print(input + input)

if torch.cuda.is_available():
    input = input.cuda()
    print(input)
    print(model(input))
    print(input + input)
