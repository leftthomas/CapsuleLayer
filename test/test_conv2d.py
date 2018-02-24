import sys
from functools import partial

import pytest
import torch
from torch.autograd import Variable, gradcheck

sys.path.append("..")
import capsule_layer as CL
from capsule_layer import CapsuleConv2d

test_data = [('sum', 1), ('dynamic', 2), ('EM', 3)]


@pytest.mark.parametrize('routing_type, num_iterations', test_data)
def test_function(routing_type, num_iterations):
    x_cpu = Variable(torch.randn(6, 3, 28, 32).double(), requires_grad=True)
    w_cpu = Variable(torch.randn(64, 3, 3, 2, 1, 8).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride=1, padding=1, routing_type=routing_type,
                              num_iterations=num_iterations)
    y_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride=1, padding=1, routing_type=routing_type,
                             num_iterations=num_iterations)
    assert (y_fast.cpu() - y_ref).data.abs().max() < 1e-9

    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_cov2d, padding=1, routing_type=routing_type, num_iterations=num_iterations),
                     (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_cov2d, padding=1, routing_type=routing_type, num_iterations=num_iterations),
                     (x_cpu, w_cpu))

    assert (gx_fast.cpu() - gx_ref).data.abs().max() < 1e-9
    assert (gw_fast.cpu() - gw_ref).data.abs().max() < 1e-9


@pytest.mark.parametrize('routing_type, num_iterations', test_data)
def test_module(routing_type, num_iterations):
    module = CapsuleConv2d(in_channels=3, out_channels=16, kernel_size=3, in_length=1, out_length=8,
                           padding=1, routing_type=routing_type, num_iterations=num_iterations)
    x = Variable(torch.randn(10, 3, 56, 64))
    y_cpu = module(x)
    y_cuda = module.cuda()(x.cuda())
    assert (y_cpu - y_cuda.cpu()).data.abs().max() < 1e-6


@pytest.mark.parametrize('routing_type, num_iterations', test_data)
def test_multigpu(routing_type, num_iterations):
    a0 = Variable(torch.randn(6, 3, 28, 32).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(6, 3, 28, 32).cuda(1), requires_grad=True)
    w0 = Variable(torch.randn(64, 3, 3, 2, 1, 8).double().cuda(0), requires_grad=True)
    w1 = Variable(torch.randn(64, 3, 3, 2, 1, 8).double().cuda(1), requires_grad=True)
    y0 = CL.capsule_cov2d(a0, w0, padding=1, routing_type=routing_type, num_iterations=num_iterations)
    go = torch.randn(y0.size()).double().cuda()
    y0.backward(go)
    y1 = CL.capsule_cov2d(a1, w1, padding=1, routing_type=routing_type, num_iterations=num_iterations)
    y1.backward(go.cuda(1))
