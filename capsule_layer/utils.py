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
  // which batch
  const int batch_index = index / (${in_capsules} * ${in_length});
  // which capsule in each batch
  const int capsule_index = (index / ${in_length}) % ${in_capsules};
  if (index < ${nthreads}){
    const int neural_index = index % ${out_length};
    // which weight matrix
    const ${Dtype}* weight = weight_data + capsule_index * ${in_length} * ${out_length} + neural_index * ${out_length};
    ${Dtype} value = 0;
    for (int il = 0; il < ${in_length}; ++il){
      const int offset = batch_index * (${in_capsules} * ${in_length}) + capsule_index * ${in_length} + il;
      value += (*weight) * input_data[offset];
      ++weight;
    }
    output_data[index] = value;
  }
}
'''

capsule_conv2d_kernels = '''
extern "C"
__global__ void capsule_conv2d_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}
'''
