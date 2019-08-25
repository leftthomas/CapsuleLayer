from functools import partial

import pytest
import torch
from torch.autograd import gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleConv2d

test_data = [(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length, out_length,
              stride, padding, dilation, routing_type, num_iterations, squash) for batch_size in [1, 3] for height in
             [5, 12] for width in [5, 12] for in_length in [1, 3] for out_length in [1, 3] for kernel_size_h in [1, 3]
             for kernel_size_w in [1, 3] for in_channels in [1 * in_length, 3 * in_length] for out_channels in
             [3 * out_length] for stride in [1, 2] for padding in [0, 1] for dilation in [1, 2] for routing_type in
             ['dynamic', 'k_means'] for num_iterations in [1, 3, 4] for squash in [True, False]]


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, dilation, routing_type, num_iterations, squash',
                         test_data)
def test_function(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, dilation, routing_type, num_iterations, squash):
    x_cpu = torch.randn(batch_size, in_channels, height, width, dtype=torch.double, requires_grad=True)
    w_cpu = torch.randn(out_channels // out_length, out_length, in_length, kernel_size_h, kernel_size_w,
                        dtype=torch.double, requires_grad=True)
    x_gpu = x_cpu.detach().to('cuda').requires_grad_()
    w_gpu = w_cpu.detach().to('cuda').requires_grad_()
    y_fast, prob_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride, padding, dilation, routing_type=routing_type,
                                         num_iterations=num_iterations, squash=squash)
    y_ref, prob_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride, padding, dilation, routing_type=routing_type,
                                       num_iterations=num_iterations, squash=squash)
    assert torch.allclose(y_fast.cpu(), y_ref)
    assert torch.allclose(prob_fast.cpu(), prob_ref)

    go_cpu = torch.randn(y_ref.size(), dtype=torch.double)
    go_gpu = go_cpu.detach().to('cuda')
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.clone()
    gw_fast = w_gpu.grad.clone()
    assert gradcheck(
        partial(CL.capsule_cov2d, stride=stride, padding=padding, dilation=dilation, routing_type=routing_type,
                num_iterations=num_iterations, squash=squash), (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.clone()
    gw_ref = w_cpu.grad.clone()
    assert gradcheck(
        partial(CL.capsule_cov2d, stride=stride, padding=padding, dilation=dilation, routing_type=routing_type,
                num_iterations=num_iterations, squash=squash), (x_cpu, w_cpu))

    assert torch.allclose(gx_fast.cpu(), gx_ref)
    assert torch.allclose(gw_fast.cpu(), gw_ref)


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, dilation, routing_type, num_iterations, squash',
                         test_data)
def test_module(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                out_length, stride, padding, dilation, routing_type, num_iterations, squash):
    module = CapsuleConv2d(in_channels, out_channels, (kernel_size_h, kernel_size_w), in_length, out_length, stride,
                           padding, dilation, routing_type=routing_type, num_iterations=num_iterations, squash=squash)
    x = torch.randn(batch_size, in_channels, height, width)
    y_cpu, prob_cpu = module(x)
    y_cuda, prob_cuda = module.to('cuda')(x.to('cuda'))
    assert torch.allclose(y_cuda.cpu(), y_cpu)
    assert torch.allclose(prob_cuda.cpu(), prob_cpu)


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, dilation, routing_type, num_iterations, squash',
                         test_data)
def test_multigpu(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, dilation, routing_type, num_iterations, squash):
    a0 = torch.randn(batch_size, in_channels, height, width, device='cuda:0', requires_grad=True)
    a1 = torch.randn(batch_size, in_channels, height, width, device='cuda:1', requires_grad=True)
    w0 = torch.randn(out_channels // out_length, out_length, in_length, kernel_size_h, kernel_size_w, device='cuda:0',
                     requires_grad=True)
    w1 = torch.randn(out_channels // out_length, out_length, in_length, kernel_size_h, kernel_size_w, device='cuda:1',
                     requires_grad=True)
    y0, prob0 = CL.capsule_cov2d(a0, w0, stride, padding, dilation, routing_type=routing_type,
                                 num_iterations=num_iterations, squash=squash)
    go = torch.randn(y0.size(), device='cuda:0')
    y0.backward(go)
    y1, prob1 = CL.capsule_cov2d(a1, w1, stride, padding, dilation, routing_type=routing_type,
                                 num_iterations=num_iterations, squash=squash)
    y1.backward(go.detach().to('cuda:1'))
