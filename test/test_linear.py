from functools import partial

import pytest
import torch
from pytest import approx
from torch.autograd import Variable, gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleLinear

kwargs_data = {'dynamic': [{'cum': cum, 'squash': squash} for cum in [True, False] for squash in [True, False]],
               'k_means': [{'similarity': similarity, 'squash': squash} for similarity in
                           ['cosine', 'tonimoto', 'pearson'] for squash in [True, False]],
               'db_scan': [{'distance': distance, 'squash': squash} for distance in ['euclidean'] for squash in
                           [True, False]]}
test_data = [
    (batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, kwargs, share_weight, num_iterations)
    for batch_size in [1, 2] for in_capsules in [1, 5, 10] for out_capsules in [1, 4] for in_length in
    [1, 2, 3] for out_length in [1, 2, 3] for routing_type in ['dynamic', 'k_means', 'db_scan'] for kwargs in
    kwargs_data[routing_type] for share_weight in [True, False] for num_iterations in [1, 3, 4]]


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, kwargs, share_weight, num_iterations', test_data)
def test_function(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, kwargs, share_weight,
                  num_iterations):
    x_cpu = Variable(torch.randn(batch_size, in_capsules, in_length).double(), requires_grad=True)
    if share_weight:
        w_cpu = Variable(torch.randn(out_capsules, out_length, in_length).double(), requires_grad=True)
    else:
        w_cpu = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_linear(x_gpu, w_gpu, share_weight, routing_type, num_iterations, **kwargs)
    y_ref = CL.capsule_linear(x_cpu, w_cpu, share_weight, routing_type, num_iterations, **kwargs)
    assert y_fast.cpu().data.view(-1).tolist() == approx(y_ref.data.view(-1).tolist())

    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    assert gradcheck(
        partial(CL.capsule_linear, share_weight=share_weight, routing_type=routing_type, num_iterations=num_iterations,
                **kwargs), (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    assert gradcheck(
        partial(CL.capsule_linear, share_weight=share_weight, routing_type=routing_type, num_iterations=num_iterations,
                **kwargs), (x_cpu, w_cpu))

    assert gx_fast.cpu().view(-1).tolist() == approx(gx_ref.view(-1).tolist())
    assert gw_fast.cpu().view(-1).tolist() == approx(gw_ref.view(-1).tolist())


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, kwargs, share_weight, num_iterations', test_data)
def test_module(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, kwargs, share_weight,
                num_iterations):
    if share_weight:
        num_in_capsules = None
    else:
        num_in_capsules = in_capsules
    module = CapsuleLinear(out_capsules, in_length, out_length, num_in_capsules, share_weight, routing_type,
                           num_iterations, **kwargs)
    x = torch.randn(batch_size, in_capsules, in_length)
    y_cpu = module(Variable(x))
    y_cuda = module.cuda()(Variable(x.cuda()))
    assert y_cuda.cpu().data.view(-1).tolist() == approx(y_cpu.data.view(-1).tolist(), abs=1e-5)


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, kwargs, share_weight, num_iterations', test_data)
def test_multigpu(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, kwargs, share_weight,
                  num_iterations):
    a0 = Variable(torch.randn(batch_size, in_capsules, in_length).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(batch_size, in_capsules, in_length).cuda(1), requires_grad=True)
    if share_weight:
        w0 = Variable(torch.randn(out_capsules, out_length, in_length).cuda(0), requires_grad=True)
        w1 = Variable(torch.randn(out_capsules, out_length, in_length).cuda(1), requires_grad=True)
    else:
        w0 = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).cuda(0), requires_grad=True)
        w1 = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).cuda(1), requires_grad=True)
    y0 = CL.capsule_linear(a0, w0, share_weight, routing_type, num_iterations, **kwargs)
    go = torch.randn(y0.size()).cuda()
    y0.backward(go)
    y1 = CL.capsule_linear(a1, w1, share_weight, routing_type, num_iterations, **kwargs)
    y1.backward(go.cuda(1))
