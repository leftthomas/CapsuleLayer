import unittest
from functools import partial

import torch
from torch.autograd import Variable, gradcheck

import capsule_layer as CL
from capsule_layer import CapsuleConv2d, CapsuleLinear


class TestPYINN(unittest.TestCase):

    def test_capsule_conv2d(self):
        x = Variable(torch.randn(8, 1, 5, 5).cuda(), requires_grad=True)
        w = Variable(torch.randn(n, 1, 3, 3).cuda(), requires_grad=True)
        y_fast = CL.capsule_cov2d(x, w, padding=1)
        y_ref = F.conv2d(x, w, padding=1, groups=n)
        go = torch.randn(y_fast.size()).double().cuda()

        self.assertLess((y_fast - y_ref).data.abs().max(), 1e-9)

        x.requires_grad = True
        w.requires_grad = True
        y_fast.backward(go)
        gx_fast = x.grad.data.clone()
        gw_fast = w.grad.data.clone()

        x.grad.data.zero_()
        w.grad.data.zero_()
        y_ref.backward(go)
        gx_ref = x.grad.data.clone()
        gw_ref = w.grad.data.clone()

        self.assertTrue(gradcheck(partial(P.conv2d_depthwise, padding=1), (x, w,)))

    def test_capsule_linear(self):
        n = 6
        x = Variable(torch.randn(1, n, 5, 5).double().cuda(), requires_grad=True)
        w = Variable(torch.randn(n, 1, 3, 3).double().cuda(), requires_grad=True)
        y_fast = CL.capsule_linear(x, w, padding=1)
        y_ref = F.conv2d(x, w, padding=1, groups=n)
        go = torch.randn(y_fast.size()).double().cuda()

        self.assertLess((y_fast - y_ref).data.abs().max(), 1e-9)

        x.requires_grad = True
        w.requires_grad = True
        y_fast.backward(go)
        gx_fast = x.grad.data.clone()
        gw_fast = w.grad.data.clone()

        x.grad.data.zero_()
        w.grad.data.zero_()
        y_ref.backward(go)
        gx_ref = x.grad.data.clone()
        gw_ref = w.grad.data.clone()

        self.assertTrue(gradcheck(partial(P.conv2d_depthwise, padding=1), (x, w,)))

    def test_conv2d_depthwise_multigpu(self):
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
        module = CapsuleConv2d(channels=8, kernel_size=3)
        x = Variable(torch.randn(1, 8, 5, 5))
        y = module(x)
        y_cuda = module.cuda()(x.cuda())
        self.assertLess((y - y_cuda.cpu()).data.abs().max(), 1e-6)

        module = CapsuleLinear(channels=8, kernel_size=3)
        x = Variable(torch.randn(1, 8, 5, 5))
        y = module(x)
        y_cuda = module.cuda()(x.cuda())
        self.assertLess((y - y_cuda.cpu()).data.abs().max(), 1e-6)
