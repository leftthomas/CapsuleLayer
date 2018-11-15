from functools import partial

import pytest
import torch
from pytest import approx
from torch.autograd import gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleConv2d

kwargs_data = {
    'dynamic': [{'squash': squash, 'softmax_dim': softmax_dim} for squash in [True, False] for softmax_dim in [1, 2]],
    'k_means': [{'similarity': similarity, 'squash': squash, 'softmax_dim': softmax_dim} for similarity in
                ['dot', 'cosine', 'tonimoto', 'pearson'] for squash in [True, False] for softmax_dim in [1, 2]]
}
test_data = [(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length, out_length,
              stride, padding, routing_type, kwargs, num_iterations) for batch_size in [1, 3] for height in [5, 12] for
             width in
             [5, 12] for in_length in [1, 3] for out_length in [1, 3] for kernel_size_h in [1, 3] for kernel_size_w in
             [1, 3] for in_channels in [1 * in_length, 3 * in_length] for out_channels in [3 * out_length] for stride in
             [1, 2] for padding in [0, 1] for routing_type in ['dynamic', 'k_means']
             for kwargs in kwargs_data[routing_type] for num_iterations in [1, 3, 4]]


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, kwargs, num_iterations', test_data)
def test_function(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, routing_type, kwargs, num_iterations):
    x_cpu = torch.randn(batch_size, in_channels, height, width, dtype=torch.double, requires_grad=True)
    w_cpu = torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                        in_length, dtype=torch.double, requires_grad=True)
    x_gpu = x_cpu.detach().to('cuda').requires_grad_()
    w_gpu = w_cpu.detach().to('cuda').requires_grad_()
    y_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride, padding, routing_type, num_iterations, **kwargs)
    y_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride, padding, routing_type, num_iterations, **kwargs)
    assert y_fast.view(-1).tolist() == approx(y_ref.view(-1).tolist())

    go_cpu = torch.randn(y_ref.size(), dtype=torch.double)
    go_gpu = go_cpu.detach().to('cuda')
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.clone()
    gw_fast = w_gpu.grad.clone()
    assert gradcheck(partial(CL.capsule_cov2d, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations, **kwargs), (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.clone()
    gw_ref = w_cpu.grad.clone()
    assert gradcheck(partial(CL.capsule_cov2d, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations, **kwargs), (x_cpu, w_cpu))

    assert gx_fast.view(-1).tolist() == approx(gx_ref.view(-1).tolist())
    assert gw_fast.view(-1).tolist() == approx(gw_ref.view(-1).tolist())


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, kwargs, num_iterations', test_data)
def test_module(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                out_length, stride, padding, routing_type, kwargs, num_iterations):
    module = CapsuleConv2d(in_channels, out_channels, (kernel_size_h, kernel_size_w), in_length, out_length, stride,
                           padding, routing_type, num_iterations, **kwargs)
    x = torch.randn(batch_size, in_channels, height, width)
    y_cpu = module(x)
    y_cuda = module.to('cuda')(x.to('cuda'))
    assert y_cuda.view(-1).tolist() == approx(y_cpu.view(-1).tolist(), abs=1e-5)


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, kwargs, num_iterations', test_data)
def test_multigpu(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, routing_type, kwargs, num_iterations):
    a0 = torch.randn(batch_size, in_channels, height, width, device='cuda:0', requires_grad=True)
    a1 = torch.randn(batch_size, in_channels, height, width, device='cuda:1', requires_grad=True)
    w0 = torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                     in_length, device='cuda:0', requires_grad=True)
    w1 = torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                     in_length, device='cuda:1', requires_grad=True)
    y0 = CL.capsule_cov2d(a0, w0, stride, padding, routing_type, num_iterations, **kwargs)
    go = torch.randn(y0.size(), device='cuda:0')
    y0.backward(go)
    y1 = CL.capsule_cov2d(a1, w1, stride, padding, routing_type, num_iterations, **kwargs)
    y1.backward(go.detach().to('cuda:1'))
