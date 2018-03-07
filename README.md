# Capsule Layer
PyTorch Capsule Layer.

## Requirements
* [Anaconda(Python 3.6 version)](https://www.anaconda.com/download/)
* PyTorch(version >= 0.3.1)
```
conda install pytorch torchvision cuda90 -c pytorch
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
x = Variable(torch.randn(4,8,28,50))
w = Variable(torch.randn(16,4,3,5)) 
if torch.cuda.is_available():
    x = x.cuda()
    w = w.cuda()
y = capsule_cov2d(x, w, in_length=4, out_length=8, stride=1, padding=1)
```
or with modules interface:
```python
import torch
from torch.autograd import Variable
from capsule_layer import CapsuleConv2d
x = Variable(torch.randn(4,8,28,50))
module = CapsuleConv2d(in_channels=8, out_channels=16, kernel_size=(3,5), in_length=4, out_length=8, stride=1, padding=1)
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
x = Variable(torch.randn(64,128,8))
w = Variable(torch.randn(10,128,16,8)) 
if torch.cuda.is_available():
    x = x.cuda()
    w = w.cuda()
# routing_type options: ['sum', 'dynamic', 'means', 'cosine', 'tonimoto', 'pearson']
y = capsule_linear(x, w, routing_type='sum')
```
or with modules interface:
```python
import torch
from torch.autograd import Variable
from capsule_layer import CapsuleLinear
x = Variable(torch.randn(64,128,8))
# routing_type options: ['sum', 'dynamic', 'means', 'cosine', 'tonimoto', 'pearson']
module = CapsuleLinear(in_capsules=128, out_capsules=10, in_length=8, out_length=16, routing_type='dynamic', num_iterations=3)
if torch.cuda.is_available():
    x = x.cuda()
    module.cuda()
y = module(x)
```

## Contribution
Any contributions to Capsule Layer are welcome!

## Copyright and License
Capsule Layer is provided under the [MIT License](LICENSE).