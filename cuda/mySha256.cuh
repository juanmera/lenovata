#pragma once
#include <stdint.h>

typedef struct {
    uint32_t H[8];
} Hash;

typedef struct {
    unsigned char L;
    unsigned char P[8];
} Password;

extern "C" void sha256Digest(Password *p, Hash *h);
extern "C" void gpuLenovoHash(Password *pwds, Password *pout, Hash *targetHash, const char * modelSerial, uint32_t blocks, uint32_t threads, uint32_t passwords);
