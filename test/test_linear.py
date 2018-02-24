import sys
from functools import partial

import pytest
import torch
from torch.autograd import Variable, gradcheck

sys.path.append("..")
import capsule_layer as CL
from capsule_layer import CapsuleLinear

test_datas = [('sum', None)]


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_function(routing_type, num_iterations):
    x_cpu = Variable(torch.randn(64, 5, 8).double(), requires_grad=True)
    w_cpu = Variable(torch.randn(10, 16, 5, 8).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_linear(x_gpu, w_gpu, routing_type=routing_type, num_iterations=num_iterations)
    y_ref = CL.capsule_linear(x_cpu, w_cpu, routing_type=routing_type, num_iterations=num_iterations)
    assert (y_fast.cpu() - y_ref).data.abs().max() < 1e-9

    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    # assert gradcheck(partial(CL.capsule_linear, routing_type=routing_type, num_iterations=num_iterations),
    #                  (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    # assert gradcheck(partial(CL.capsule_linear, routing_type=routing_type, num_iterations=num_iterations),
    #                  (x_cpu, w_cpu))

    assert (gx_fast.cpu() - gx_ref).abs().max() < 1e-9
    assert (gw_fast.cpu() - gw_ref).abs().max() < 1e-9


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_module(routing_type, num_iterations):
    module = CapsuleLinear(in_capsules=32, out_capsules=10, in_length=8, out_length=16, routing_type=routing_type,
                           num_iterations=num_iterations)
    x = Variable(torch.randn(128, 32, 8))
    y_cpu = module(x)
    y_cuda = module.cuda()(Variable(x.data.cuda(), requires_grad=True))
    assert (y_cpu - y_cuda.cpu()).data.abs().max() < 1e-4


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_multigpu(routing_type, num_iterations):
    a0 = Variable(torch.randn(6, 7, 8).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(6, 7, 8).cuda(1), requires_grad=True)
    w0 = Variable(torch.randn(5, 4, 7, 8).double().cuda(0), requires_grad=True)
    w1 = Variable(torch.randn(5, 4, 7, 8).double().cuda(1), requires_grad=True)
    y0 = CL.capsule_linear(a0, w0, routing_type=routing_type, num_iterations=num_iterations)
    go = torch.randn(y0.size()).double().cuda()
    y0.backward(go)
    y1 = CL.capsule_linear(a1, w1, routing_type=routing_type, num_iterations=num_iterations)
    y1.backward(go.cuda(1))
