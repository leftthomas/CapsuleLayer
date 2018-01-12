# Capsule Layer

CuPy implementations of fused PyTorch Capsule Layer.

The purpose of this package is to contain CUDA ops written in Python with CuPy, which is not a PyTorch dependency.

### Requirements
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
* CuPy
```
pip install cupy
```

### Installation
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

### Example
```
import torch
from torch.autograd import Variable
from capsule_layer.modules import CapsuleConv2d
x = Variable(torch.randn(1,4,5,5).cuda())

module = CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=3, 
                          in_length=1, out_length=4, stride=1, padding=1).cuda()
y = module(x)
```
