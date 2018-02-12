from collections import namedtuple
from string import Template

import cupy
import torch

Stream = namedtuple('Stream', ['ptr'])
cuda_num_threads = 1024


def get_thread_blocks(n, k=cuda_num_threads):
    return (n + k - 1) // k


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


capsule_conv2d_kernels = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{

}

extern "C"
__global__ void capsule_conv2d_input_backward(${Dtype}* grad_input, const ${Dtype}* grad_output)
{

}

extern "C"
__global__ void capsule_conv2d_weight_backward(${Dtype}* grad_input, const ${Dtype}* grad_output)
{

}
'''

capsule_linear_kernels = '''
extern "C"
__global__ void capsule_linear_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_size = ${nthreads} / (${out_capsules} * ${out_length});
  if (index < ${nthreads}){
    int batch = index / (${out_capsules} * ${out_length});
    int oc = (index / ${out_length}) % ${out_capsules};
    int ol = index % ${out_length};
    for (int ic = 0; ic < ${in_capsules}; ++ic){
      for (int il = 0; il < ${in_length}; ++il){
        output_data[index] += input_data[batch*${in_capsules}*${in_length}+ic*${in_length}+il] * weight_data[oc*${out_length}*${in_capsules}*${in_length}+ol*${in_capsules}*${in_length}+ic*${in_length}+il];
      }
    }
  }
}

extern "C"
__global__ void capsule_linear_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{

}

extern "C"
__global__ void capsule_linear_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{

}
'''
