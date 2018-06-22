import pytest
import torch.nn as nn

from capsule_layer import CapsuleConv2d, CapsuleLinear
from capsule_layer.optim import MultiStepRI, MultiStepDropout

test_ri_data = [(milestones, addition) for milestones in [[1, 2], [3, 5], [7, 12], [5, 8], [4, 7]] for addition
                in [[1, 5], [2, 4], 1, 2, 5]]

test_dropout_data = [(milestones, addition) for milestones in [[1, 2], [3, 5], [7, 12], [5, 8], [4, 7]] for addition
                     in [[0.1, 0.5], [0.2, 0.4], 0.1, 0.2, 0.3]]


@pytest.mark.parametrize('milestones, addition', test_ri_data)
def test_ri_optim(milestones, addition):
    models = [nn.Conv2d(1, 3, 3), CapsuleLinear(10, 8, 16, num_iterations=1), CapsuleConv2d(8, 16, 3, 4, 8),
              nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), CapsuleLinear(10, 8, 16)), nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, 5, 3), nn.ReLU(), CapsuleLinear(10, 8, 16, num_iterations=2)),
             nn.Sequential(CapsuleLinear(10, 8, 16), CapsuleConv2d(8, 16, 3, 4, 8)), CapsuleLinear(10, 8, 16)])]
    for model in models:
        schedule = MultiStepRI(model, milestones, addition, verbose=True)
        for epoch in range(10):
            schedule.step()


@pytest.mark.parametrize('milestones, addition', test_dropout_data)
def test_dropout_optim(milestones, addition):
    models = [nn.Conv2d(1, 3, 3), CapsuleLinear(10, 8, 16, dropout=0.1), CapsuleConv2d(8, 16, 3, 4, 8),
              nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), CapsuleLinear(10, 8, 16)), nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, 5, 3), nn.ReLU(), CapsuleLinear(10, 8, 16, dropout=0.2)),
             nn.Sequential(CapsuleLinear(10, 8, 16), CapsuleConv2d(8, 16, 3, 4, 8)), CapsuleLinear(10, 8, 16)])]
    for model in models:
        schedule = MultiStepDropout(model, milestones, addition, verbose=True)
        for epoch in range(10):
            schedule.step()
