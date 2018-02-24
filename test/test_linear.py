import sys
import time
import pytest
from functools import partial

import torch
from torch.autograd import Variable, gradcheck

sys.path.append("..")
import capsule_layer as CL
from capsule_layer import CapsuleLinear

test_datas = [('sum', None), ('dynamic', 1), ('dynamic', 3), ('EM', 2), ('EM', 4)]


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_function(routing_type, num_iterations):
    print('--------test capsule linear forward function of {} routing--------'.format(routing_type))
    x_cpu = Variable(torch.randn(64, 512, 8).double(), requires_grad=True)
    w_cpu = Variable(torch.randn(10, 16, 512, 8).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    start = time.clock()
    y_fast = CL.capsule_linear(x_gpu, w_gpu)
    print('gpu mode cost ' + str(time.clock() - start) + 's')
    start = time.clock()
    y_ref = CL.capsule_linear(x_cpu, w_cpu, routing_type=routing_type, num_iterations=num_iterations)
    print('cpu mode cost ' + str(time.clock() - start) + 's')
    assert (y_fast.cpu() - y_ref).data.abs().max() < 1e-9

    print('--------test capsule linear backward function of {} routing--------'.format(routing_type))
    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    start = time.clock()
    y_fast.backward(go_gpu)
    print('gpu mode cost ' + str(time.clock() - start) + 's')
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    # self.assertTrue(gradcheck(partial(CL.capsule_linear), (x_gpu, w_gpu)))

    start = time.clock()
    y_ref.backward(go_cpu)
    print('cpu mode cost ' + str(time.clock() - start) + 's')
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    # self.assertTrue(gradcheck(partial(CL.capsule_linear), (x_cpu, w_cpu)))

    assert (gx_fast.cpu() - gx_ref).abs().max() < 1e-9
    assert (gw_fast.cpu() - gw_ref).abs().max() < 1e-9


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_module(routing_type, num_iterations):
    print('--------test capsule linear module of {} routing--------'.format(routing_type))
    module = CapsuleLinear(in_capsules=32, out_capsules=10, in_length=8, out_length=16, routing_type=routing_type,
                           num_iterations=num_iterations)
    x = Variable(torch.randn(16, 32, 8))
    y_cpu = module(x)
    y_cuda = module.cuda()(x.cuda())
    assert (y_cpu - y_cuda.cpu()).data.abs().max() < 1e-6


@pytest.mark.parametrize('routing_type, num_iterations', test_datas)
def test_multigpu(routing_type, num_iterations):
    print('--------test capsule linear on multigpu of {} routing--------'.format(routing_type))
    a0 = Variable(torch.randn(6, 7, 8).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(6, 7, 8).cuda(1), requires_grad=True)
    w0 = Variable(torch.randn(5, 4, 7, 8).double().cuda(0), requires_grad=True)
    w1 = Variable(torch.randn(5, 4, 7, 8).double().cuda(1), requires_grad=True)
    y0 = CL.capsule_linear(a0, w0, routing_type=routing_type, num_iterations=num_iterations)
    go = torch.randn(y0.size()).double().cuda()
    y0.backward(go)
    y1 = CL.capsule_linear(a1, w1)
    y1.backward(go.cuda(1))
