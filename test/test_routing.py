import pytest
import torch
from pytest import approx
from torch.autograd import Variable

from capsule_layer.functional import dynamic_routing, k_means_routing, db_scan_routing

kwargs_data = {'dynamic': [True, False], 'k_means': ['cosine', 'standardized_cosine', 'tonimoto', 'pearson'],
               'db_scan': ['euclidean']}
routing_funcs = {'dynamic': dynamic_routing, 'k_means': k_means_routing, 'db_scan': db_scan_routing}

test_data = [(batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations) for batch_size
             in [1, 2] for out_capsules in [1, 5, 10] for in_capsules in [1, 4, 20] for out_length in [1, 8, 16] for
             routing_type in ['dynamic', 'k_means', 'db_scan'] for kwargs in kwargs_data[routing_type]
             for num_iterations in [1, 4, 50]]


@pytest.mark.parametrize('batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations',
                         test_data)
def test_routing(batch_size, in_capsules, out_capsules, out_length, routing_type, kwargs, num_iterations):
    x = torch.randn(batch_size, out_capsules, in_capsules, out_length).double()
    y_cpu = routing_funcs[routing_type](Variable(x), num_iterations, kwargs)
    y_cuda = routing_funcs[routing_type](Variable(x.cuda()), num_iterations, kwargs)
    assert y_cuda.cpu().data.view(-1).tolist() == approx(y_cpu.data.view(-1).tolist())
