import pytest
import torch
from pytest import approx
from torch.autograd import Variable

from capsule_layer.functional import dynamic_route_linear, means_route_linear, cosine_route_linear, \
    tonimoto_route_linear, pearson_route_linear

test_data = [(batch_size, out_capsules, in_capsules, out_length, routing_func, num_iterations) for batch_size
             in [1, 2] for out_capsules in [1, 5, 10] for in_capsules in [1, 4, 20] for out_length in [1, 8, 16] for
             routing_func in [dynamic_route_linear, means_route_linear, cosine_route_linear, tonimoto_route_linear,
                              pearson_route_linear] for num_iterations in [1, 4, 50]]


@pytest.mark.parametrize('batch_size, out_capsules, in_capsules, out_length, routing_func, num_iterations', test_data)
def test_routing(batch_size, in_capsules, out_capsules, out_length, routing_func, num_iterations):
    x = Variable(torch.randn(batch_size, out_capsules, in_capsules, out_length).double())
    y_cpu = routing_func(x, num_iterations)
    y_cuda = routing_func(x.cuda(), num_iterations)
    assert y_cuda.cpu().data.view(-1).tolist() == approx(y_cpu.data.view(-1).tolist())
