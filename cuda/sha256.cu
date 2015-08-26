#include <stdio.h>
#include <stdlib.h>
#include "errors.cu"
#include "sha256.cuh"

// #define DEBUG

#define BlockSize 64
#define BlockSizeInt32 16
#define MessageAlloc 128
#define MessageAlloc32 32

#define SHR(x, n) (x >> n)
#define ROTR(x, n) ((x >> n) | (x << 32 - n))
#define Ch(x, y, z) ((x & y) ^ (~x & z))
#define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define S0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define s1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

// #define BSWAP_32(x) (x)
#define BSWAP_32(x) ((x >> 24) | ((x >> 8) & 0x00ff00) | ((x << 8) & 0xff0000) | (x << 24))

struct Message {
    unsigned char blocks;
    size_t len;
    union {
        unsigned char *text;
        uint32_t *u32;
    };
    __device__ Message() {
        len = 0;
        u32 = (uint32_t *)malloc(MessageAlloc);
#pragma unroll 32
        for (int i = 0; i < MessageAlloc32; ++i) {
            u32[i] = 0;
        }
    };

    __device__ ~Message() {
        free(u32);
    };

    __device__ void init_msg(const void *msg, size_t msg_len) {
        memcpy(text, msg, msg_len);
        len = msg_len;
    };

    __device__ void set_msg(const void *msg, size_t msg_len, size_t offset=0, size_t size=0) {
        memcpy(text + offset, msg, msg_len);
        if (size == 0) {
            size = msg_len;
        }
        len += size;
        blocks = len / BlockSize + 1;
        text[len] = 0x80;
        text[blocks * BlockSize - 2] = len >> 5;
        text[blocks * BlockSize - 1] = len << 3;
    };
};

static const uint HostPasswordSize = sizeof(Password);
static const uint HostHashSize = sizeof(Hash);
__constant__ uint PasswordSize = sizeof(Password);
__constant__ uint HashSize = sizeof(Hash);

__device__ void debug_dump_hash(Hash *hash) {
#ifdef DEBUG
    unsigned char * hb = (unsigned char *) hash->H;
    printf("Hash dump:");
    for (int i=0; i < 32; ++i) {
        printf(" %02x", hb[i]);
    }
    printf("\n");
#endif
}

__constant__ const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


__device__ void sha256_init(Hash *hash) {
    memcpy(hash->H, H0, 32);
}

__device__ void sha256_endian_swap(Hash *hash) {
#pragma unroll 8
    for (uint t = 0; t < 8; ++t) {
        hash->H[t] = BSWAP_32(hash->H[t]);
    }
}

__device__ void sha256_block(Hash *hash, const Message *msg) {
    uint32_t W[64], T1, T2;
    uint32_t * u32 = msg->u32;
    Hash interm;
    for (uint i = 0; i < msg->blocks; ++i) {
        printf("DBG sha256_block: %d\n", i);
// Step 1
#pragma unroll 16
        for (uint t = 0; t < 16; ++t) {
            W[t] = BSWAP_32(u32[t]);
            printf("DBG W(%d): %x\n", t, W[t]);
        }
        u32 += BlockSizeInt32;

#pragma unroll 48
        for (uint t = 16; t < 64; ++t) {
            W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16];
            printf("DBG W(%d): %x\n", t, W[t]);
        }
// Step 2
        memcpy(&interm, hash, HashSize);
// Step 3
#pragma unroll 64
        for (uint t = 0; t < 64; ++t) {
            T1 = interm.H[7] + S1(interm.H[4]) + Ch(interm.H[4], interm.H[5], interm.H[6]) + K[t] + W[t];
            T2 = S0(interm.H[0]) + Maj(interm.H[0], interm.H[1], interm.H[2]);
            interm.H[7] = interm.H[6];
            interm.H[6] = interm.H[5];
            interm.H[5] = interm.H[4];
            interm.H[4] = interm.H[3] + T1;
            interm.H[3] = interm.H[2];
            interm.H[2] = interm.H[1];
            interm.H[1] = interm.H[0];
            interm.H[0] = T1 + T2;
        }
// Step 4
#pragma unroll 8
        for (uint t = 0; t < 8; ++t) {
            hash->H[t] += interm.H[t];
        }
        printf("DBG interm.H[0]: %x\n", hash->H[0]);
    }
}

__global__ void kernel_sha256(Password *pwd, Hash *hash) {
    Message msg;
    msg.set_msg(pwd->P, pwd->L);
    sha256_init(hash);
    sha256_block(hash, &msg);
    sha256_endian_swap(hash);
}

void sha256_sum(Password *pwd, Hash *hash) {
    Password *dev_pwd;
    Hash *dev_hash;

    cudaMalloc(&dev_pwd, HostPasswordSize);
    cudaMalloc(&dev_hash, HostHashSize);
    cudaMemcpy(dev_pwd, pwd, HostPasswordSize, cudaMemcpyHostToDevice);
    kernel_sha256<<<1,1>>>(dev_pwd, dev_hash);
    HandleCudaError(cudaGetLastError());
    cudaMemcpy(hash, dev_hash, HostHashSize, cudaMemcpyDeviceToHost);
};
