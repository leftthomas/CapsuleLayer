import sys
from functools import partial

import pytest
import torch
from torch.autograd import Variable, gradcheck

sys.path.append("..")
import capsule_layer as CL
from capsule_layer import CapsuleConv2d

test_data = \
    [(3, 3, 1, 0, 'sum', 1), (3, 3, 2, 0, 'sum', 1), (3, 3, 1, 1, 'sum', 1), (3, 3, 1, 2, 'sum', 1),
     (3, 3, 2, 1, 'sum', 1), (3, 3, 2, 2, 'sum', 1), (3, 3, 1, 0, 'sum', 2), (3, 3, 2, 0, 'sum', 2),
     (3, 3, 1, 1, 'sum', 2), (3, 3, 1, 2, 'sum', 2), (3, 3, 2, 1, 'sum', 2), (3, 3, 2, 2, 'sum', 2),
     (3, 3, 1, 0, 'sum', 3), (3, 3, 2, 0, 'sum', 3), (3, 3, 1, 1, 'sum', 3), (3, 3, 1, 2, 'sum', 3),
     (3, 3, 2, 1, 'sum', 3), (3, 3, 2, 2, 'sum', 3), (3, 3, 1, 0, 'sum', 4), (3, 3, 2, 0, 'sum', 4),
     (3, 3, 1, 1, 'sum', 4), (3, 3, 1, 2, 'sum', 4), (3, 3, 2, 1, 'sum', 4), (3, 3, 2, 2, 'sum', 4),
     (3, 2, 1, 0, 'sum', 1), (3, 2, 2, 0, 'sum', 1), (3, 2, 1, 1, 'sum', 1), (3, 2, 1, 2, 'sum', 1),
     (3, 2, 2, 1, 'sum', 1), (3, 2, 2, 2, 'sum', 1), (3, 2, 1, 0, 'sum', 2), (3, 2, 2, 0, 'sum', 2),
     (3, 2, 1, 1, 'sum', 2), (3, 2, 1, 2, 'sum', 2), (3, 2, 2, 1, 'sum', 2), (3, 2, 2, 2, 'sum', 2),
     (3, 2, 1, 0, 'sum', 3), (3, 2, 2, 0, 'sum', 3), (3, 2, 1, 1, 'sum', 3), (3, 2, 1, 2, 'sum', 3),
     (3, 2, 2, 1, 'sum', 3), (3, 2, 2, 2, 'sum', 3), (3, 2, 1, 0, 'sum', 4), (3, 2, 2, 0, 'sum', 4),
     (3, 2, 1, 1, 'sum', 4), (3, 2, 1, 2, 'sum', 4), (3, 2, 2, 1, 'sum', 4), (3, 2, 2, 2, 'sum', 4),
     (2, 3, 1, 0, 'sum', 1), (2, 3, 2, 0, 'sum', 1), (2, 3, 1, 1, 'sum', 1), (2, 3, 1, 2, 'sum', 1),
     (2, 3, 2, 1, 'sum', 1), (2, 3, 2, 2, 'sum', 1), (2, 3, 1, 0, 'sum', 2), (2, 3, 2, 0, 'sum', 2),
     (2, 3, 1, 1, 'sum', 2), (2, 3, 1, 2, 'sum', 2), (2, 3, 2, 1, 'sum', 2), (2, 3, 2, 2, 'sum', 2),
     (2, 3, 1, 0, 'sum', 3), (2, 3, 2, 0, 'sum', 3), (2, 3, 1, 1, 'sum', 3), (2, 3, 1, 2, 'sum', 3),
     (2, 3, 2, 1, 'sum', 3), (2, 3, 2, 2, 'sum', 3), (2, 3, 1, 0, 'sum', 4), (2, 3, 2, 0, 'sum', 4),
     (2, 3, 1, 1, 'sum', 4), (2, 3, 1, 2, 'sum', 4), (2, 3, 2, 1, 'sum', 4), (2, 3, 2, 2, 'sum', 4),
     (3, 3, 1, 0, 'dynamic', 1), (3, 3, 2, 0, 'dynamic', 1), (3, 3, 1, 1, 'dynamic', 1), (3, 3, 1, 2, 'dynamic', 1),
     (3, 3, 2, 1, 'dynamic', 1), (3, 3, 2, 2, 'dynamic', 1), (3, 3, 1, 0, 'dynamic', 2), (3, 3, 2, 0, 'dynamic', 2),
     (3, 3, 1, 1, 'dynamic', 2), (3, 3, 1, 2, 'dynamic', 2), (3, 3, 2, 1, 'dynamic', 2), (3, 3, 2, 2, 'dynamic', 2),
     (3, 3, 1, 0, 'dynamic', 3), (3, 3, 2, 0, 'dynamic', 3), (3, 3, 1, 1, 'dynamic', 3), (3, 3, 1, 2, 'dynamic', 3),
     (3, 3, 2, 1, 'dynamic', 3), (3, 3, 2, 2, 'dynamic', 3), (3, 3, 1, 0, 'dynamic', 4), (3, 3, 2, 0, 'dynamic', 4),
     (3, 3, 1, 1, 'dynamic', 4), (3, 3, 1, 2, 'dynamic', 4), (3, 3, 2, 1, 'dynamic', 4), (3, 3, 2, 2, 'dynamic', 4),
     (3, 2, 1, 0, 'dynamic', 1), (3, 2, 2, 0, 'dynamic', 1), (3, 2, 1, 1, 'dynamic', 1), (3, 2, 1, 2, 'dynamic', 1),
     (3, 2, 2, 1, 'dynamic', 1), (3, 2, 2, 2, 'dynamic', 1), (3, 2, 1, 0, 'dynamic', 2), (3, 2, 2, 0, 'dynamic', 2),
     (3, 2, 1, 1, 'dynamic', 2), (3, 2, 1, 2, 'dynamic', 2), (3, 2, 2, 1, 'dynamic', 2), (3, 2, 2, 2, 'dynamic', 2),
     (3, 2, 1, 0, 'dynamic', 3), (3, 2, 2, 0, 'dynamic', 3), (3, 2, 1, 1, 'dynamic', 3), (3, 2, 1, 2, 'dynamic', 3),
     (3, 2, 2, 1, 'dynamic', 3), (3, 2, 2, 2, 'dynamic', 3), (3, 2, 1, 0, 'dynamic', 4), (3, 2, 2, 0, 'dynamic', 4),
     (3, 2, 1, 1, 'dynamic', 4), (3, 2, 1, 2, 'dynamic', 4), (3, 2, 2, 1, 'dynamic', 4), (3, 2, 2, 2, 'dynamic', 4),
     (2, 3, 1, 0, 'dynamic', 1), (2, 3, 2, 0, 'dynamic', 1), (2, 3, 1, 1, 'dynamic', 1), (2, 3, 1, 2, 'dynamic', 1),
     (2, 3, 2, 1, 'dynamic', 1), (2, 3, 2, 2, 'dynamic', 1), (2, 3, 1, 0, 'dynamic', 2), (2, 3, 2, 0, 'dynamic', 2),
     (2, 3, 1, 1, 'dynamic', 2), (2, 3, 1, 2, 'dynamic', 2), (2, 3, 2, 1, 'dynamic', 2), (2, 3, 2, 2, 'dynamic', 2),
     (2, 3, 1, 0, 'dynamic', 3), (2, 3, 2, 0, 'dynamic', 3), (2, 3, 1, 1, 'dynamic', 3), (2, 3, 1, 2, 'dynamic', 3),
     (2, 3, 2, 1, 'dynamic', 3), (2, 3, 2, 2, 'dynamic', 3), (2, 3, 1, 0, 'dynamic', 4), (2, 3, 2, 0, 'dynamic', 4),
     (2, 3, 1, 1, 'dynamic', 4), (2, 3, 1, 2, 'dynamic', 4), (2, 3, 2, 1, 'dynamic', 4), (2, 3, 2, 2, 'dynamic', 4),
     (3, 3, 1, 0, 'EM', 1), (3, 3, 2, 0, 'EM', 1), (3, 3, 1, 1, 'EM', 1), (3, 3, 1, 2, 'EM', 1),
     (3, 3, 2, 1, 'EM', 1), (3, 3, 2, 2, 'EM', 1), (3, 3, 1, 0, 'EM', 2), (3, 3, 2, 0, 'EM', 2),
     (3, 3, 1, 1, 'EM', 2), (3, 3, 1, 2, 'EM', 2), (3, 3, 2, 1, 'EM', 2), (3, 3, 2, 2, 'EM', 2),
     (3, 3, 1, 0, 'EM', 3), (3, 3, 2, 0, 'EM', 3), (3, 3, 1, 1, 'EM', 3), (3, 3, 1, 2, 'EM', 3),
     (3, 3, 2, 1, 'EM', 3), (3, 3, 2, 2, 'EM', 3), (3, 3, 1, 0, 'EM', 4), (3, 3, 2, 0, 'EM', 4),
     (3, 3, 1, 1, 'EM', 4), (3, 3, 1, 2, 'EM', 4), (3, 3, 2, 1, 'EM', 4), (3, 3, 2, 2, 'EM', 4),
     (3, 2, 1, 0, 'EM', 1), (3, 2, 2, 0, 'EM', 1), (3, 2, 1, 1, 'EM', 1), (3, 2, 1, 2, 'EM', 1),
     (3, 2, 2, 1, 'EM', 1), (3, 2, 2, 2, 'EM', 1), (3, 2, 1, 0, 'EM', 2), (3, 2, 2, 0, 'EM', 2),
     (3, 2, 1, 1, 'EM', 2), (3, 2, 1, 2, 'EM', 2), (3, 2, 2, 1, 'EM', 2), (3, 2, 2, 2, 'EM', 2),
     (3, 2, 1, 0, 'EM', 3), (3, 2, 2, 0, 'EM', 3), (3, 2, 1, 1, 'EM', 3), (3, 2, 1, 2, 'EM', 3),
     (3, 2, 2, 1, 'EM', 3), (3, 2, 2, 2, 'EM', 3), (3, 2, 1, 0, 'EM', 4), (3, 2, 2, 0, 'EM', 4),
     (3, 2, 1, 1, 'EM', 4), (3, 2, 1, 2, 'EM', 4), (3, 2, 2, 1, 'EM', 4), (3, 2, 2, 2, 'EM', 4),
     (2, 3, 1, 0, 'EM', 1), (2, 3, 2, 0, 'EM', 1), (2, 3, 1, 1, 'EM', 1), (2, 3, 1, 2, 'EM', 1),
     (2, 3, 2, 1, 'EM', 1), (2, 3, 2, 2, 'EM', 1), (2, 3, 1, 0, 'EM', 2), (2, 3, 2, 0, 'EM', 2),
     (2, 3, 1, 1, 'EM', 2), (2, 3, 1, 2, 'EM', 2), (2, 3, 2, 1, 'EM', 2), (2, 3, 2, 2, 'EM', 2),
     (2, 3, 1, 0, 'EM', 3), (2, 3, 2, 0, 'EM', 3), (2, 3, 1, 1, 'EM', 3), (2, 3, 1, 2, 'EM', 3),
     (2, 3, 2, 1, 'EM', 3), (2, 3, 2, 2, 'EM', 3), (2, 3, 1, 0, 'EM', 4), (2, 3, 2, 0, 'EM', 4),
     (2, 3, 1, 1, 'EM', 4), (2, 3, 1, 2, 'EM', 4), (2, 3, 2, 1, 'EM', 4), (2, 3, 2, 2, 'EM', 4)]


@pytest.mark.parametrize('kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations', test_data)
def test_function(kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations):
    x_cpu = Variable(torch.randn(6, 3, 7, 11).double(), requires_grad=True)
    w_cpu = Variable(torch.randn(5, 3, kernel_size_h, kernel_size_w, 1, 4).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride=stride, padding=padding, routing_type=routing_type,
                              num_iterations=num_iterations)
    y_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride=stride, padding=padding, routing_type=routing_type,
                             num_iterations=num_iterations)
    assert (y_fast.cpu() - y_ref).data.abs().max() < 1e-9

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

    assert (gx_fast.cpu() - gx_ref).data.abs().max() < 1e-9
    assert (gw_fast.cpu() - gw_ref).data.abs().max() < 1e-9


@pytest.mark.parametrize('kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations', test_data)
def test_module(kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations):
    module = CapsuleConv2d(in_channels=4, out_channels=8, kernel_size=(kernel_size_h, kernel_size_w), in_length=2,
                           out_length=4, stride=stride, padding=padding, routing_type=routing_type,
                           num_iterations=num_iterations)
    x = Variable(torch.randn(5, 4, 6, 11))
    y_cpu = module(x)
    y_cuda = module.cuda()(x.cuda())
    assert (y_cuda.cpu() - y_cpu).data.abs().max() < 1e-6


@pytest.mark.parametrize('kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations', test_data)
def test_multigpu(kernel_size_h, kernel_size_w, stride, padding, routing_type, num_iterations):
    a0 = Variable(torch.randn(6, 3, 12, 15).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(6, 3, 12, 15).cuda(1), requires_grad=True)
    w0 = Variable(torch.randn(4, 3, kernel_size_h, kernel_size_w, 1, 8).double().cuda(0), requires_grad=True)
    w1 = Variable(torch.randn(4, 3, kernel_size_h, kernel_size_w, 1, 8).double().cuda(1), requires_grad=True)
    y0 = CL.capsule_cov2d(a0, w0, stride=stride, padding=padding, routing_type=routing_type,
                          num_iterations=num_iterations)
    go = torch.randn(y0.size()).double().cuda()
    y0.backward(go)
    y1 = CL.capsule_cov2d(a1, w1, stride=stride, padding=padding, routing_type=routing_type,
                          num_iterations=num_iterations)
    y1.backward(go.cuda(1))
