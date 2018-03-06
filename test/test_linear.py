from functools import partial

import pytest
import torch
from pytest import approx
from torch.autograd import Variable, gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleLinear

test_data = [(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations) for batch_size
             in [1, 2, 3] for in_capsules in [1, 2, 5] for out_capsules in [1, 2, 5] for in_length in [1, 2, 3] for
             out_length in [1, 2, 3] for routing_type in ['sum', 'dynamic', 'means'] for num_iterations in [1, 2, 3, 4]]


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations',
                         test_data)
def test_function(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations):
    x_cpu = Variable(torch.randn(batch_size, in_capsules, in_length).double(), requires_grad=True)
    w_cpu = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).double(), requires_grad=True)
    x_gpu = Variable(x_cpu.data.cuda(), requires_grad=True)
    w_gpu = Variable(w_cpu.data.cuda(), requires_grad=True)
    y_fast = CL.capsule_linear(x_gpu, w_gpu, routing_type=routing_type, num_iterations=num_iterations)
    y_ref = CL.capsule_linear(x_cpu, w_cpu, routing_type=routing_type, num_iterations=num_iterations)
    assert y_fast.cpu().data.numpy() == approx(y_ref.data.numpy())

    go_cpu = torch.randn(y_ref.size()).double()
    go_gpu = go_cpu.cuda()
    y_fast.backward(go_gpu)
    gx_fast = x_gpu.grad.data.clone()
    gw_fast = w_gpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_linear, routing_type=routing_type, num_iterations=num_iterations),
                     (x_gpu, w_gpu))

    y_ref.backward(go_cpu)
    gx_ref = x_cpu.grad.data.clone()
    gw_ref = w_cpu.grad.data.clone()
    assert gradcheck(partial(CL.capsule_linear, routing_type=routing_type, num_iterations=num_iterations),
                     (x_cpu, w_cpu))

    assert gx_fast.cpu().numpy() == approx(gx_ref.numpy())
    assert gw_fast.cpu().numpy() == approx(gw_ref.numpy())


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations',
                         test_data)
def test_module(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations):
    module = CapsuleLinear(in_capsules=in_capsules, out_capsules=out_capsules, in_length=in_length,
                           out_length=out_length, routing_type=routing_type, num_iterations=num_iterations)
    x = Variable(torch.randn(batch_size, in_capsules, in_length))
    y_cpu = module(x)
    y_cuda = module.cuda()(x.cuda())
    assert y_cuda.cpu().data.numpy() == approx(y_cpu.data.numpy())


@pytest.mark.parametrize('batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations',
                         test_data)
def test_multigpu(batch_size, in_capsules, out_capsules, in_length, out_length, routing_type, num_iterations):
    a0 = Variable(torch.randn(batch_size, in_capsules, in_length).cuda(0), requires_grad=True)
    a1 = Variable(torch.randn(batch_size, in_capsules, in_length).cuda(1), requires_grad=True)
    w0 = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).double().cuda(0), requires_grad=True)
    w1 = Variable(torch.randn(out_capsules, in_capsules, out_length, in_length).double().cuda(1), requires_grad=True)
    y0 = CL.capsule_linear(a0, w0, routing_type=routing_type, num_iterations=num_iterations)
    go = torch.randn(y0.size()).double().cuda()
    y0.backward(go)
    y1 = CL.capsule_linear(a1, w1, routing_type=routing_type, num_iterations=num_iterations)
    y1.backward(go.cuda(1))
