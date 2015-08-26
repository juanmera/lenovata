#pragma once
#include "cuda_runtime.h"

void handle_cuda_error(cudaError_t, const char *, int);

#define HandleCudaError(err) handle_cuda_error(err, __FILE__, __LINE__)
