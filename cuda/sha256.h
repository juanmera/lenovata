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

#define THREADS 512
#define BLOCKS 2048*6 /* it must be something divisible by 3 */
#define KEYS_PER_CRYPT THREADS*BLOCKS
typedef struct{
  unsigned char v[19];
  unsigned char length;
} sha256_password;

typedef struct{
  uint32_t v[8]; 				///256bits
} sha256_hash;


#define BINARY_SIZE		32
#define SHA_HASH		sha256_hash

void gpu_rawsha256(sha256_password * i, SHA_HASH * o, int lap);
#endif
