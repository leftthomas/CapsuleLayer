from collections import namedtuple
from string import Template

import cupy
import torch

Stream = namedtuple('Stream', ['ptr'])
CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


capsule_linear_kernels = '''
extern "C"
__global__ void capsule_linear_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
}

extern "C"
__global__ void capsule_linear_backward(${Dtype} *grad_input, const unsigned char *mask, const ${Dtype} *grad_output,
                                int chw, int total)
{
}
'''

capsule_conv2d_kernels = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
}

extern "C"
__global__ void capsule_conv2d_backward(${Dtype} *grad_input, const unsigned char *mask, const ${Dtype} *grad_output,
                                int chw, int total)
{
}
'''
