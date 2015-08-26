#pragma once
#include <stdint.h>
#include "sha256.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void lenovo_sum(Password *, Password *, Hash *, const char *, uint32_t, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif
