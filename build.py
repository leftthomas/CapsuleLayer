import os

import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/capsule_lib.c']
headers = ['src/capsule_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/capsule_lib_cuda.c']
    headers += ['src/capsule_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    '_ext.capsule_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
