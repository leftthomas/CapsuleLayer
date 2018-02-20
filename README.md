# Capsule Layer
CuPy fused PyTorch Capsule Layer.

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

### Examples
* CapsuleConv2d
```python
import torch
from torch.autograd import Variable
from capsule_layer.modules import CapsuleConv2d
x = Variable(torch.randn(4,1,5,7).cuda())

module = CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=3, in_length=1, out_length=4, padding=1).cuda()
y = module(x)
```
* CapsuleLinear
```python
import torch
from torch.autograd import Variable
from capsule_layer.modules import CapsuleLinear
x = Variable(torch.randn(8,128,16).cuda())

module = CapsuleLinear(in_capsules=32, out_capsules=10, in_length=8, out_length=16).cuda()
y = module(x)
```

## Note
The dynamic routing algorithm or other routing algorithms isn't implemented now! 
If someone could implement it with single CUDA Kernel Function, please let me know.

## Credits
Referenced CuPy fused PyTorch neural networks ops:
[PyINN by @szagoruyko](https://github.com/szagoruyko/pyinn)