from torch.nn.modules.module import Module

from ..functions.add import CapsuleLayerFunction


class CapsuleLayerModule(Module):
    def forward(self, input1, input2):
        return CapsuleLayerFunction()(input1, input2)
