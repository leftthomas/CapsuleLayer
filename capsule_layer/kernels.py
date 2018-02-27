capsule_conv2d_sum_forward_kernel = '''
extern "C"
__global__ void capsule_conv2d_sum_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int n = index / ${out_channels} / ${out_height} / ${out_width};
    int c = (index / ${out_height} / ${out_width}) % ${out_channels};
    int h = (index / ${out_width}) % ${out_height};
    int w = index % ${out_width};
    ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        int h_in = -${pad_h} + h * ${stride_h} + kh;
        int w_in = -${pad_w} + w * ${stride_w} + kw;
        if ((h_in >= 0) && (h_in < ${in_height}) && (w_in >= 0) && (w_in < ${in_width})) {
          int offset = ((n * ${in_channels} + c) * ${in_height} + h_in) * ${in_width} + w_in;
          value += (*weight) * input_data[offset];
        }
        ++weight;
      }
    }
    output_data[index] = value;
  }
}
'''

capsule_conv2d_sum_input_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_sum_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){

  }
}
'''

capsule_conv2d_sum_weight_backward_kernel = '''
extern "C"
__global__ void capsule_conv2d_sum_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){

  }
}
'''

capsule_linear_sum_forward_kernel = '''
extern "C"
__global__ void capsule_linear_sum_forward(const ${Dtype}* input_data, const ${Dtype}* weight_data, ${Dtype}* output_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int batch = index / ${out_capsules} / ${out_length};
    int oc = (index / ${out_length}) % ${out_capsules};
    int ol = index % ${out_length};
    ${Dtype} value = 0;
    for (int ic = 0; ic < ${in_capsules}; ++ic){
      for (int il = 0; il < ${in_length}; ++il){
        int input_offset = batch * ${in_capsules} * ${in_length} + ic * ${in_length} + il;
        int weight_offset = oc * ${out_length} * ${in_capsules} * ${in_length} + ol * ${in_capsules} 
          * ${in_length} + ic * ${in_length} + il;
        value += input_data[input_offset] * weight_data[weight_offset];
      }
    }
    output_data[index] = value;
  }
}
'''

capsule_linear_sum_input_backward_kernel = '''
extern "C"
__global__ void capsule_linear_sum_input_backward(const ${Dtype}* grad_output, const ${Dtype}* weight, ${Dtype}* grad_input)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int batch = index / ${in_capsules} / ${in_length};
    int ic = (index / ${in_length}) % ${in_capsules};
    int il = index % ${in_length};
    ${Dtype} value = 0;
    for (int oc = 0; oc < ${out_capsules}; ++oc){
      for (int ol = 0; ol < ${out_length}; ++ol){
        int grad_offset = batch * ${out_capsules} * ${out_length} + oc * ${out_length} + ol;
        int weight_offset = oc * ${out_length} * ${in_capsules} * ${in_length} + ol * ${in_capsules} 
          * ${in_length} + ic * ${in_length} + il;
        value += grad_output[grad_offset] * weight[weight_offset];
      }
    }
    grad_input[index] = value;
  }
}
'''

capsule_linear_sum_weight_backward_kernel = '''
extern "C"
__global__ void capsule_linear_sum_weight_backward(const ${Dtype}* grad_output, const ${Dtype}* input, ${Dtype}* grad_weight)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ${nthreads}){
    int ic = (index / ${in_length}) % ${in_capsules};
    int il = index % ${in_length};
    int oc = (index / ${out_length} / ${in_capsules} / ${in_length}) % ${out_capsules};
    int ol = (index / ${in_capsules} / ${in_length}) % ${out_length};
    ${Dtype} value = 0;
    for (int batch = 0; batch < ${batch_size}; ++batch){
      int grad_offset = batch * ${out_capsules} * ${out_length} + oc * ${out_length} + ol;
      int input_offset = batch * ${in_capsules} * ${in_length} +ic * ${in_length} + il;
      value += grad_output[grad_offset] * input[input_offset];
    }
    grad_weight[index] = value;
  }
}
'''
