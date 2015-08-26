#include <stdio.h>
#include "errors.cuh"

void handle_cuda_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n",
            cudaGetErrorString(err), file, line);
        if (err == cudaErrorLaunchOutOfResources)
            fprintf(stderr, "Try decreasing THREADS in the corresponding cuda*h file. See doc/README-CUDA\n");
        exit(EXIT_FAILURE);
    }
}
