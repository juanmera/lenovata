#pragma once
#include <stdint.h>

typedef struct {
    union {
        uint32_t h[8];
        unsigned char hb[32];
    };
} Hash;

typedef struct {
    uint l;
    unsigned char p[10];
} Password;

extern "C" void sha256Digest(Password *p, Hash *h);
extern "C" void lenovoFind(Password *pwds, Password *pout, Hash *targetHash, unsigned char * modelSerial);
