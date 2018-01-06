int conv2d_forward_cuda(THCudaTensor *input, THCudaTensor *weight, THCudaTensor *stride, THCudaTensor *padding, THCudaTensor *num_iterations, THCudaTensor *output);
int conv2d_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);

int linear_forward_cuda(THCudaTensor *input, THCudaTensor *weight, THCudaTensor *num_iterations, THCudaTensor *output);
int linear_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);
