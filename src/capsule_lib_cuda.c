#include <THC/THC.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int conv2d_forward_cuda(THCudaTensor *input, THCudaTensor *output)
{
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_cadd(state, output, input, 1.0, input);
  return 1;
}

int conv2d_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{
  THCudaTensor_resizeAs(state, grad_input, grad_output);
  THCudaTensor_fill(state, grad_input, 1);
  return 1;
}
