#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t H[8];
} Hash;

typedef struct {
    unsigned char L;
    unsigned char P[8];
} Password;

void sha256_sum(Password *p, Hash *h);

#ifdef __cplusplus
}
#endif
