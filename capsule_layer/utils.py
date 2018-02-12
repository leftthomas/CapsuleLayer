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


capsule_conv2d_forward_kernel = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{

}
'''

capsule_conv2d_input_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{

}
'''

capsule_conv2d_weight_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{

}
'''

capsule_linear_forward_kernel = '''
extern "C"
__global__ void capsule_linear_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
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
'''

capsule_linear_input_backward_kernel = '''
extern "C"
__global__ void capsule_linear_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int batch = index / (${in_capsules} * ${in_length});
    int ic = (index / ${in_length}) % ${in_capsules};
    int il = index % ${in_length};
    for (int oc = 0; oc < ${out_capsules}; ++oc){
      for (int ol = 0; ol < ${out_length}; ++ol){
        grad_input[index] += grad_output[batch*${out_capsules}*${out_length}+oc*${out_length}+ol] * weight[oc*${out_length}*${in_capsules}*${in_length}+ol*${in_capsules}*${in_length}+ic*${in_length}+il];
      }
    }
  }
}
'''

capsule_linear_weight_backward_kernel = '''
extern "C"
__global__ void capsule_linear_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int ic = (index / ${in_length}) % ${in_capsules};
    int il = index % ${in_length};
    int oc = (index / (${out_length} * ${in_capsules} * ${in_length})) % ${out_capsules};
    int ol = (index / (${in_capsules} * ${in_length})) % ${out_length};
    for (int batch = 0; batch < ${batch_size}; ++batch){
      grad_weight[index] += grad_output[batch*${out_capsules}*${out_length}+oc*${out_length}+ol] * input[batch*${in_capsules}*${in_length}+ic*${in_length}+il];
    }
  }
}
'''
