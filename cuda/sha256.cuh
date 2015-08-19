/*
* This software is Copyright (c) 2011-2012 Lukas Odzioba <ukasz at openwall dot net>
* and it is hereby released to the general public under the following terms:
* Redistribution and use in source and binary forms, with or without modification, are permitted.
* This file is shared by raw-sha224-cuda and raw-sha256-cuda formats
*/
#ifndef _SHA256_H
#define _SHA256_H

#include "stdint.h"
#include "common.cuh"

#define rol(x,n) ((x << n) | (x >> (32-n)))
#define ror(x,n) ((x >> n) | (x << (32-n)))
#define Ch(x,y,z) ( z ^ (x & ( y ^ z)) )
#define Maj(x,y,z) ( (x & y) | (z & (x | y)) )
#define Sigma0(x) ((ror(x,2))  ^ (ror(x,13)) ^ (ror(x,22)))
#define Sigma1(x) ((ror(x,6))  ^ (ror(x,11)) ^ (ror(x,25)))
#define sigma0(x) ((ror(x,7))  ^ (ror(x,18)) ^(x>>3))
#define sigma1(x) ((ror(x,17)) ^ (ror(x,19)) ^(x>>10))

#define STREAMS 16 / 4
#define THREADS 256 / 4
#define BLOCKS 1024 * 8 * STREAMS
#define KEYS_PER_CRYPT THREADS * BLOCKS

typedef struct {
  unsigned char v[19];
  unsigned char length;
} Sha256Password;

typedef union {
    uint32_t h[8];
    unsigned char hb[32];
} Sha256Digest;

void gpuSha256(Sha256Password * i, Sha256Digest * o);
#endif
