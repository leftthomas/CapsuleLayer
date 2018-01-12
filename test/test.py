import unittest
from functools import partial

import torch
from torch.autograd import Variable, gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleConv2d, CapsuleLinear


class TestCapsuleLayer(unittest.TestCase):

    def test_capsule_conv2d(self):
        x_gpu = Variable(torch.randn(6, 3, 28, 32).double().cuda(), requires_grad=True)
        w_gpu = Variable(torch.randn(64, 3, 3, 2, 1, 8).double().cuda(), requires_grad=True)
        x_cpu = x_gpu.cpu()
        w_cpu = w_gpu.cpu()
        y_fast = CL.capsule_cov2d(x_gpu, w_gpu, stride=1, padding=1, num_iterations=3)
        y_ref = CL.capsule_cov2d(x_cpu, w_cpu, stride=1, padding=1, num_iterations=3)
        go_gpu = torch.randn(y_fast.size()).double().cuda()
        go_cpu = go_gpu.cpu()

        self.assertLess((y_fast.cpu() - y_ref).data.abs().max(), 1e-9)

        x_gpu.requires_grad = True
        w_gpu.requires_grad = True
        y_fast.backward(go_gpu)
        gx_fast = x_gpu.grad.data.clone()
        gw_fast = w_gpu.grad.data.clone()

        self.assertTrue(gradcheck(partial(CL.capsule_cov2d, padding=1), (x_gpu, w_gpu,)))

        x_cpu.requires_grad = True
        w_cpu.requires_grad = True
        y_ref.backward(go_cpu)
        gx_ref = x_cpu.grad.data.clone()
        gw_ref = w_cpu.grad.data.clone()

        self.assertTrue(gradcheck(partial(CL.capsule_cov2d, padding=1), (x_cpu, w_cpu,)))

        self.assertLess((gx_fast.cpu() - gx_ref).data.abs().max(), 1e-9)
        self.assertLess((gw_fast.cpu() - gw_ref).data.abs().max(), 1e-9)

    def test_capsule_linear(self):
        x_gpu = Variable(torch.randn(6, 7, 8).double().cuda(), requires_grad=True)
        w_gpu = Variable(torch.randn(5, 7, 8, 4).double().cuda(), requires_grad=True)
        x_cpu = x_gpu.cpu()
        w_cpu = w_gpu.cpu()
        y_fast = CL.capsule_linear(x_gpu, w_gpu, num_iterations=3)
        y_ref = CL.capsule_linear(x_cpu, w_cpu, num_iterations=3)
        go_gpu = torch.randn(y_fast.size()).double().cuda()
        go_cpu = go_gpu.cpu()

        self.assertLess((y_fast.cpu() - y_ref).data.abs().max(), 1e-9)

        x_gpu.requires_grad = True
        w_gpu.requires_grad = True
        y_fast.backward(go_gpu)
        gx_fast = x_gpu.grad.data.clone()
        gw_fast = w_gpu.grad.data.clone()

        self.assertTrue(gradcheck(partial(CL.capsule_linear, padding=1), (x_gpu, w_gpu,)))

        x_cpu.requires_grad = True
        w_cpu.requires_grad = True
        y_ref.backward(go_cpu)
        gx_ref = x_cpu.grad.data.clone()
        gw_ref = w_cpu.grad.data.clone()

        self.assertTrue(gradcheck(partial(CL.capsule_linear, padding=1), (x_cpu, w_cpu,)))

        self.assertLess((gx_fast.cpu() - gx_ref).data.abs().max(), 1e-9)
        self.assertLess((gw_fast.cpu() - gw_ref).data.abs().max(), 1e-9)

    def test_capsule_conv2d_multigpu(self):
        n = 6
        a0 = Variable(torch.randn(1, n, 5, 5).cuda(0), requires_grad=True)
        a1 = Variable(torch.randn(1, n, 5, 5).cuda(1), requires_grad=True)
        w0 = Variable(torch.randn(n, 1, 3, 3).double().cuda(0), requires_grad=True)
        w1 = Variable(torch.randn(n, 1, 3, 3).double().cuda(1), requires_grad=True)
        y0 = P.conv2d_depthwise(a0, w0, padding=1)
        go = torch.randn(y0.size()).double().cuda()
        y0.backward(go)
        y1 = P.conv2d_depthwise(a1, w1, padding=1)
        y1.backward(go.cuda(1))

    def test_capsule_linear_multigpu(self):
        n = 6
        a0 = Variable(torch.randn(1, n, 5, 5).cuda(0), requires_grad=True)
        a1 = Variable(torch.randn(1, n, 5, 5).cuda(1), requires_grad=True)
        w0 = Variable(torch.randn(n, 1, 3, 3).double().cuda(0), requires_grad=True)
        w1 = Variable(torch.randn(n, 1, 3, 3).double().cuda(1), requires_grad=True)
        y0 = P.conv2d_depthwise(a0, w0, padding=1)
        go = torch.randn(y0.size()).double().cuda()
        y0.backward(go)
        y1 = P.conv2d_depthwise(a1, w1, padding=1)
        y1.backward(go.cuda(1))

    def test_modules(self):
        module = CapsuleConv2d(in_channels=3, out_channels=16, kernel_size=3, in_length=1, out_length=8, padding=1)
        x = Variable(torch.randn(1, 8, 5, 5))
        y = module(x)
        y_cuda = module.cuda()(x.cuda())
        self.assertLess((y - y_cuda.cpu()).data.abs().max(), 1e-6)

        module = CapsuleLinear(in_capsules=32, out_capsules=10, in_length=8, out_length=16)
        x = Variable(torch.randn(1, 8, 5, 5))
        y = module(x)
        y_cuda = module.cuda()(x.cuda())
        self.assertLess((y - y_cuda.cpu()).data.abs().max(), 1e-6)
