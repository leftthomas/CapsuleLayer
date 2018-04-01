import pytest
import torch
from pytest import approx
from torch.autograd import Variable

from capsule_layer.functional import k_means_routing, db_scan_routing

kwargs_data = {'k_means': [{'similarity': similarity, 'squash': squash} for similarity in
                           ['cosine', 'tonimoto', 'pearson'] for squash in [True, False]],
               'db_scan': [{'distance': distance, 'squash': squash} for distance in ['euclidean'] for squash in
                           [True, False]]}
routing_funcs = {'k_means': k_means_routing, 'db_scan': db_scan_routing}

test_data = [(batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations) for batch_size
             in [1, 2] for out_capsules in [1, 5, 10] for in_capsules in [1, 4, 20] for out_length in [1, 8, 16] for
             routing_type in ['k_means', 'db_scan'] for kwargs in kwargs_data[routing_type]
             for num_iterations in [1, 4, 50]]


@pytest.mark.parametrize('batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations',
                         test_data)
def test_routing(batch_size, in_capsules, out_capsules, out_length, routing_type, kwargs, num_iterations):
    x = torch.randn(batch_size, out_capsules, in_capsules, out_length).double()
    y_cpu = routing_funcs[routing_type](Variable(x), num_iterations, **kwargs)
    y_cuda = routing_funcs[routing_type](Variable(x.cuda()), num_iterations, **kwargs)
    assert y_cuda.cpu().data.view(-1).tolist() == approx(y_cpu.data.view(-1).tolist())
