/*
* This software is Copyright (c) 2011-2012 Lukas Odzioba <ukasz at openwall dot net>
* and it is hereby released to the general public under the following terms:
* Redistribution and use in source and binary forms, with or without modification, are permitted.
*/
#ifndef _CUDA_COMMON_CU
#define _CUDA_COMMON_CU

#include <stdio.h>
#include "common.cuh"

void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s in %s at line %d\n",
		    cudaGetErrorString(err), file, line);
		if (err == cudaErrorLaunchOutOfResources)
			fprintf(stderr, "Try decreasing THREADS in the corresponding cuda*h file. See doc/README-CUDA\n");
		exit(EXIT_FAILURE);
	}
}

void *cuda_pageLockedMalloc(void *w, unsigned int size) {
	HANDLE_ERROR(cudaHostAlloc((void **) &w, size, cudaHostAllocDefault));
	return w;
}

void cuda_pageLockedFree(void *w) {
	HANDLE_ERROR(cudaFreeHost(w));
}

#endif
