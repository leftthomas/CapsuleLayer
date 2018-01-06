int conv2d_forward(THFloatTensor *input, THFloatTensor *weight, THFloatTensor *stride, THFloatTensor *padding, THFloatTensor *num_iterations, THFloatTensor *output);
int conv2d_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

int linear_forward(THFloatTensor *input, THFloatTensor *weight, THFloatTensor *num_iterations, THFloatTensor *output);
int linear_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
