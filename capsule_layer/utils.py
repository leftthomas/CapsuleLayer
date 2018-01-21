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
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    ${Dtype} sum_capsule[${out_length}] = {0};
    for (int ic = 0; ic < ${in_capsules}; ++ic){
      ${Dtype} capsule[${out_length}] = {0};
      for (int ol = 0; ol < ${out_length}; ++ol){
        ${Dtype} value = 0;
        for (int il = 0; il < ${in_length}; ++il){
          value += input_data[ic*${in_length}+il] * weight_data[ic*${in_length}+ol*${out_length}+il]
        }
        capsule[ol] = value
        sum_capsule[ol] += value
      }
      sum_capsule
    }
    output_data[index] = value;
  }
}
'''

capsule_conv2d_kernels = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{

}
'''
