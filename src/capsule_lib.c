#include <TH/TH.h>

int conv2d_forward(THFloatTensor *input, THFloatTensor *output)
{
  THFloatTensor_resizeAs(output, input);
  THFloatTensor_cadd(output, input, 1.0, input);
  return 1;
}

int conv2d_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_fill(grad_input, 1);
  return 1;
}
