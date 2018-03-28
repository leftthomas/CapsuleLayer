from functools import partial

import pytest
import torch
from pytest import approx
from torch.autograd import Variable, gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleConv2d

test_data = [(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length, out_length,
              stride, padding, routing_type, num_iterations) for batch_size in [1, 3] for height in [5, 12] for width in
             [5, 12] for in_length in [1, 3] for out_length in [1, 3] for kernel_size_h in [1, 3] for kernel_size_w in
             [1, 3] for in_channels in [1 * in_length, 3 * in_length] for out_channels in [3 * out_length] for stride in
             [1, 2] for padding in [0, 1] for routing_type in
             ['dynamic', 'contract', 'means', 'cosine', 'tonimoto', 'pearson'] for num_iterations in [1, 3, 4]]


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, num_iterations', test_data)
def test_function(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, routing_type, num_iterations):
    x_cpu = Variable(torch.randn(batch_size, in_channels, height, width).double(), requires_grad=True)
    w_cpu = Variable(
        torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                    in_length).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride=stride, padding=padding, routing_type=routing_type,
                              num_iterations=num_iterations)
    y_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations)
    assert y_fast.cpu().data.view(-1).tolist() == approx(y_ref.data.view(-1).tolist())

    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_cov2d, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations), (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_cov2d, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations), (x_cpu, w_cpu))

    assert gx_fast.cpu().view(-1).tolist() == approx(gx_ref.view(-1).tolist())
    assert gw_fast.cpu().view(-1).tolist() == approx(gw_ref.view(-1).tolist())


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, num_iterations', test_data)
def test_module(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                out_length, stride, padding, routing_type, num_iterations):
    module = CapsuleConv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=(kernel_size_h, kernel_size_w), in_length=in_length, out_length=out_length,
                           stride=stride, padding=padding, routing_type=routing_type, num_iterations=num_iterations)
    x = torch.randn(batch_size, in_channels, height, width)
    y_cpu = module(Variable(x))
    y_cuda = module.cuda()(Variable(x.cuda()))
    assert y_cuda.cpu().data.view(-1).tolist() == approx(y_cpu.data.view(-1).tolist(), abs=1e-5)


@pytest.mark.parametrize('batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, '
                         'in_length, out_length, stride, padding, routing_type, num_iterations', test_data)
def test_multigpu(batch_size, height, width, in_channels, out_channels, kernel_size_h, kernel_size_w, in_length,
                  out_length, stride, padding, routing_type, num_iterations):
    a0 = Variable(torch.randn(batch_size, in_channels, height, width).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(batch_size, in_channels, height, width).cuda(1), requires_grad=True)
    w0 = Variable(
        torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                    in_length).cuda(0), requires_grad=True)
    w1 = Variable(
        torch.randn(out_channels // out_length, in_channels // in_length, kernel_size_h, kernel_size_w, out_length,
                    in_length).cuda(1), requires_grad=True)
    y0 = CL.capsule_cov2d(a0, w0, stride=stride, padding=padding, routing_type=routing_type,
                          num_iterations=num_iterations)
    go = torch.randn(y0.size()).cuda()
    y0.backward(go)
    y1 = CL.capsule_cov2d(a1, w1, stride=stride, padding=padding, routing_type=routing_type,
                          num_iterations=num_iterations)
    y1.backward(go.cuda(1))
