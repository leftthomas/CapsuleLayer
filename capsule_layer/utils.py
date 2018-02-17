from collections import namedtuple
from string import Template

import cupy
import torch

Stream = namedtuple('Stream', ['ptr'])
num_threads = 1024


def get_thread_blocks(n, k=num_threads):
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
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    const int n = index / ${out_channels} / ${out_height} / ${out_width};
    const int c = (index / ${out_height} / ${out_width}) % ${out_channels};
    const int h = (index / ${out_width}) % ${out_height};
    const int w = index % ${out_width};
    const ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h};
        const int w_in = -${pad_w} + w * ${stride_w};
        if ((h_in >= 0) && (h_in < ${in_height}) && (w_in >= 0) && (w_in < ${in_width})) {
          for (int ic = 0; ic < ${in_capsules}; ++ic){
            for (int il = 0; il < ${in_length}; ++il){
              const int input_offset = batch * ${in_capsules} * ${in_length} + ic * ${in_length} + il;
              const int weight_offset = oc * ${out_length} * ${in_capsules} * ${in_length} + ol * ${in_capsules} 
                * ${in_length} + ic * ${in_length} + il;
              output_data[index] += input_data[input_offset] * weight_data[weight_offset];
            }
          }
          const int offset = ((n * ${in_channels} + c) * ${in_height} + h_in) * ${in_width} + w_in;
          value += (*weight) * input_data[offset];
        }
        ++weight;
      }
    }
    output_data[index] = value;
  }
}
'''

capsule_conv2d_input_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    
  }
}
'''

capsule_conv2d_weight_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    
  }
}
'''

capsule_linear_forward_kernel = '''
extern "C"
__global__ void capsule_linear_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    const int batch = index / ${out_capsules} / ${out_length};
    const int oc = (index / ${out_length}) % ${out_capsules};
    const int ol = index % ${out_length};
    for (int ic = 0; ic < ${in_capsules}; ++ic){
      for (int il = 0; il < ${in_length}; ++il){
        const int input_offset = batch * ${in_capsules} * ${in_length} + ic * ${in_length} + il;
        const int weight_offset = oc * ${out_length} * ${in_capsules} * ${in_length} + ol * ${in_capsules} 
          * ${in_length} + ic * ${in_length} + il;
        output_data[index] += input_data[input_offset] * weight_data[weight_offset];
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
    const int batch = index / ${in_capsules} / ${in_length};
    const int ic = (index / ${in_length}) % ${in_capsules};
    const int il = index % ${in_length};
    for (int oc = 0; oc < ${out_capsules}; ++oc){
      for (int ol = 0; ol < ${out_length}; ++ol){
        const int grad_offset = batch * ${out_capsules} * ${out_length} + oc * ${out_length} + ol;
        const int weight_offset = oc * ${out_length} * ${in_capsules} * ${in_length} + ol * ${in_capsules} 
          * ${in_length} + ic * ${in_length} + il;
        grad_input[index] += grad_output[grad_offset] * weight[weight_offset];
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
    const int ic = (index / ${in_length}) % ${in_capsules};
    const int il = index % ${in_length};
    const int oc = (index / ${out_length} / ${in_capsules} / ${in_length}) % ${out_capsules};
    const int ol = (index / ${in_capsules} / ${in_length}) % ${out_length};
    for (int batch = 0; batch < ${batch_size}; ++batch){
      const int grad_offset = batch * ${out_capsules} * ${out_length} + oc * ${out_length} + ol;
      const int input_offset = batch * ${in_capsules} * ${in_length} +ic * ${in_length} + il;
      grad_weight[index] += grad_output[grad_offset] * input[input_offset];
    }
  }
}
'''
