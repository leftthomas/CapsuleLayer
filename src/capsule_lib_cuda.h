int conv2d_forward_cuda(THFloatTensor *input, THFloatTensor *weight, THFloatTensor *stride, THFloatTensor *padding, THFloatTensor *num_iterations, THFloatTensor *output);
int conv2d_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);

int linear_forward_cuda(THFloatTensor *input, THFloatTensor *weight, THFloatTensor *num_iterations, THFloatTensor *output);
int linear_backward_cuda(THFloatTensor *grad_output, THFloatTensor *grad_input);
