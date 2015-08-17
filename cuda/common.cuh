/*
* This software is Copyright (c) 2011-2012 Lukas Odzioba <ukasz at openwall dot net>
* and it is hereby released to the general public under the following terms:
* Redistribution and use in source and binary forms, with or without modification, are permitted.
*/
#ifndef _CUDA_COMMON_CUH
#define _CUDA_COMMON_CUH
#include "cuda_runtime.h"

void HandleError(cudaError_t err, const char *file, int line);
void cuda_device_list();
void *cuda_pageLockedMalloc(void *w,unsigned int size);
void cuda_pageLockedFree(void *w);
int cuda_getAsyncEngineCount();

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))
#endif
