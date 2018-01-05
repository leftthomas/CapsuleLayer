int capsule_layer_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
		       THCudaTensor *output);
int capsule_layer_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);
