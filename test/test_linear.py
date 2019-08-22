from functools import partial

import pytest
import torch
from torch.autograd import gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleLinear

test_data = [
    (batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, share_weight, num_iterations, squash)
    for batch_size in [1, 2] for in_capsules in [1, 5, 10] for out_capsules in [1, 4] for in_length in
    [1, 2, 3] for out_length in [1, 2, 3] for routing_type in ['dynamic', 'k_means'] for share_weight in [True, False]
    for num_iterations in [1, 3, 4] for squash in [True, False]]


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, share_weight, num_iterations, squash', test_data)
def test_function(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, share_weight,
                  num_iterations, squash):
    x_cpu = torch.randn(batch_size, in_capsules, in_length, dtype=torch.double, requires_grad=True)
    if share_weight:
        w_cpu = torch.randn(out_capsules, out_length, in_length, dtype=torch.double, requires_grad=True)
    else:
        w_cpu = torch.randn(out_capsules, in_capsules, out_length, in_length, dtype=torch.double, requires_grad=True)
    x_gpu = x_cpu.detach().to('cuda').requires_grad_()
    w_gpu = w_cpu.detach().to('cuda').requires_grad_()
    y_fast = CL.capsule_linear(x_gpu, w_gpu, share_weight, routing_type, num_iterations, squash=squash)
    y_ref = CL.capsule_linear(x_cpu, w_cpu, share_weight, routing_type, num_iterations, squash=squash)
    assert torch.allclose(y_fast.cpu(), y_ref)

    go_cpu = torch.randn(y_ref.size(), dtype=torch.double)
    go_gpu = go_cpu.detach().to('cuda')
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.clone()
    gw_fast = w_gpu.grad.clone()
    assert gradcheck(
        partial(CL.capsule_linear, share_weight=share_weight, routing_type=routing_type, num_iterations=num_iterations,
                squash=squash), (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.clone()
    gw_ref = w_cpu.grad.clone()
    assert gradcheck(
        partial(CL.capsule_linear, share_weight=share_weight, routing_type=routing_type, num_iterations=num_iterations,
                squash=squash), (x_cpu, w_cpu))

    assert torch.allclose(gx_fast.cpu(), gx_ref)
    assert torch.allclose(gw_fast.cpu(), gw_ref)


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, share_weight, num_iterations, squash', test_data)
def test_module(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, share_weight,
                num_iterations, squash):
    if share_weight:
        num_in_capsules = None
    else:
        num_in_capsules = in_capsules
    module = CapsuleLinear(out_capsules, in_length, out_length, num_in_capsules, share_weight, routing_type,
                           num_iterations, squash=squash)
    x = torch.randn(batch_size, in_capsules, in_length)
    y_cpu = module(x)
    y_cuda = module.to('cuda')(x.to('cuda'))
    assert torch.allclose(y_cuda.cpu(), y_cpu)


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, '
                         'routing_type, share_weight, num_iterations, squash', test_data)
def test_multigpu(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, share_weight,
                  num_iterations, squash):
    a0 = torch.randn(batch_size, in_capsules, in_length, device='cuda:0', requires_grad=True)
    a1 = torch.randn(batch_size, in_capsules, in_length, device='cuda:1', requires_grad=True)
    if share_weight:
        w0 = torch.randn(out_capsules, out_length, in_length, device='cuda:0', requires_grad=True)
        w1 = torch.randn(out_capsules, out_length, in_length, device='cuda:1', requires_grad=True)
    else:
        w0 = torch.randn(out_capsules, in_capsules, out_length, in_length, device='cuda:0', requires_grad=True)
        w1 = torch.randn(out_capsules, in_capsules, out_length, in_length, device='cuda:1', requires_grad=True)
    y0 = CL.capsule_linear(a0, w0, share_weight, routing_type, num_iterations, squash=squash)
    go = torch.randn(y0.size(), device='cuda:0')
    y0.backward(go)
    y1 = CL.capsule_linear(a1, w1, share_weight, routing_type, num_iterations, squash=squash)
    y1.backward(go.detach().to('cuda:1'))
