# Capsule Layer
CuPy fused PyTorch Capsule Layer, based on Hao Ren's paper [Convolutional Capsule Network for Image Classification](xxx).

The purpose of this package is to contain CUDA ops written in Python with CuPy, which is not a PyTorch dependency.

An alternative to CuPy would be <https://github.com/pytorch/extension-ffi>,
but it requires a lot of wrapping code like <https://github.com/sniklaus/pytorch-extension>,
so doesn't really work with quick prototyping. Another advantage of CuPy over C code is that dimensions of each op
are known at JIT-ing time, and compiled kernels potentially can be faster.

## Requirements
* PyTorch
```
conda install pytorch torchvision cuda90 -c pytorch
```
* setuptools
```
pip install setuptools
```
* fastrlock
```
pip install fastrlock
```
* pytest
```
pip install -U pytest
```
* CuPy
```
pip install cupy
```

## Installation
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Examples
* CapsuleConv2d
```python
import torch
from torch.autograd import Variable
from capsule_layer import capsule_cov2d
x = Variable(torch.randn(4,1,5,7))
w = Variable(torch.randn(4,1,3,3,1,4)) 
if torch.cuda.is_available():
    x = x.cuda()
    w = w.cuda()
# routing_type options: ['sum', 'dynamic', 'EM']
y = capsule_cov2d(x, w, stride=1, padding=1, routing_type='sum')
```
or with modules interface:
```python
import torch
from torch.autograd import Variable
from capsule_layer import CapsuleConv2d
x = Variable(torch.randn(4,1,5,7))
# routing_type options: ['sum', 'dynamic', 'EM']
module = CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=3, in_length=1, out_length=4, stride=1, padding=1, routing_type='sum')
if torch.cuda.is_available():
    x = x.cuda()
    module.cuda()
y = module(x)
```

* CapsuleLinear
```python
import torch
from torch.autograd import Variable
from capsule_layer import capsule_linear
x = Variable(torch.randn(8,128,8))
w = Variable(torch.randn(10,16,128,8)) 
if torch.cuda.is_available():
    x = x.cuda()
    w = w.cuda()
# routing_type options: ['sum', 'dynamic', 'EM']
y = capsule_linear(x, w, routing_type='sum')
```
or with modules interface:
```python
import torch
from torch.autograd import Variable
from capsule_layer import CapsuleLinear
x = Variable(torch.randn(8,128,8))
# routing_type options: ['sum', 'dynamic', 'EM']
module = CapsuleLinear(in_capsules=128, out_capsules=10, in_length=8, out_length=16, routing_type='dynamic', num_iterations=3)
if torch.cuda.is_available():
    x = x.cuda()
    module.cuda()
y = module(x)
```

## Note
The dynamic and matrix routing algorithms isn't implemented now, the cpu version could be speed up and optimized.

## Contribution
Any contributions to Capsule Layer are welcome!

## Copyright and License
Capsule Layer is provided under the [MIT License](LICENSE).

## Credits
Referenced CuPy fused PyTorch neural networks ops:
[PyINN by @szagoruyko](https://github.com/szagoruyko/pyinn).