import os

import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['capsule_layer/src/capsule_layer_lib.c']
headers = ['capsule_layer/src/capsule_layer_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['capsule_layer/src/capsule_layer_lib_cuda.c']
    headers += ['capsule_layer/src/capsule_layer_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'capsule_layer._ext.capsule_layer_lib',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
