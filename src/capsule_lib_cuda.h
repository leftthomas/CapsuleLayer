int conv2d_forward_cuda(THCudaTensor *input, THCudaTensor *output);
int conv2d_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);
