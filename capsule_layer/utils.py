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
  int batch_size = ${nthreads} / (${out_capsules} * ${out_length});
  if (index < ${nthreads}){
    int batch = index / (${out_capsules} * ${out_length});
    int oc = (index / ${out_length}) % batch_size;
    int ol = index % ${out_length};
    for (int ic = 0; ic < ${in_capsules}; ++ic){
      for (int il = 0; il < ${in_length}; ++il){
        output_data[index] += input_data[batch*${in_capsules}*${in_length}+ic*${in_length}+il] * weight_data[oc*${out_length}*${in_capsules}*${in_length}+ol*${in_capsules}*${in_length}+ic*${in_length}+il];
      }
    }
  }
}

extern "C"
__global__ void capsule_linear_backward(${Dtype} *grad_input, const ${Dtype} *grad_output)
{
}
'''

capsule_conv2d_kernels = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{

}

extern "C"
__global__ void capsule_conv2d_backward(${Dtype} *grad_input, const ${Dtype} *grad_output)
{
}
'''
