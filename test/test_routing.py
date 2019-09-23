import pytest
import torch

from capsule_layer.functional import dynamic_routing, k_means_routing

kwargs_data = {'dynamic': [],
               'k_means': [{'similarity': similarity} for similarity in ['dot', 'cosine', 'tonimoto', 'pearson']]}
routing_funcs = {'dynamic': dynamic_routing, 'k_means': k_means_routing}

test_data = [(batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations) for batch_size
             in [1, 2] for out_capsules in [1, 5, 10] for in_capsules in [1, 4, 20] for out_length in [1, 8, 16] for
             routing_type in ['dynamic', 'k_means'] for kwargs in kwargs_data[routing_type]
             for num_iterations in [1, 4, 7]]


@pytest.mark.parametrize('batch_size, out_capsules, in_capsules, out_length, routing_type, kwargs, num_iterations',
                         test_data)
def test_routing(batch_size, in_capsules, out_capsules, out_length, routing_type, kwargs, num_iterations):
    x = torch.randn(batch_size, out_capsules, in_capsules, out_length, dtype=torch.double)
    y_cpu, prob_cpu = routing_funcs[routing_type](x, num_iterations, **kwargs)
    y_cuda, prob_cuda = routing_funcs[routing_type](x.to('cuda'), num_iterations, **kwargs)
    assert torch.allclose(y_cuda.cpu(), y_cpu)
    assert torch.allclose(prob_cuda.cpu(), prob_cpu)
