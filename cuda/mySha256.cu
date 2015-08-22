#include <stdio.h>
#include <stdlib.h>
#include "mySha256.cuh"

#define ModelSerialSize 60
#define LenovoHashOffset 12
#define Lenovo1MessageSize 64
#define Lenovo2MessageSize 72
#define WordSize 32
#define BlockSize 64
#define BlockSizeInt32 16
#define TwoBlockSize 128

#define SHR(x, n) (x >> n)
#define ROTL(x, n) ((x << n) | (x >> 32 - n))
#define ROTR(x, n) ((x >> n) | (x << 32 - n))
#define Ch(x, y, z) ((x & y) ^ (~x & z))
#define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define S0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define s1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

#define BSWAP_32(x) ((x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24))

typedef struct {
    union {
        unsigned char *m;
        uint32_t *mi;
    };
    uint blocks;
} Message;

__constant__ uint PasswordSize = sizeof(Password);
__constant__ uint HashSize = sizeof(Hash);
__constant__ unsigned char ModelSerial[ModelSerialSize];
__constant__ Hash TargetHash;

__constant__ const uint32_t K[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ const uint32_t H0[] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


__device__ void setSize(Message *msg, uint size) {
    msg->m[size] = 0x80;
    msg->blocks = size / BlockSize + 1;
    size <<= 3;
    msg->m[msg->blocks * BlockSize - 2] = size >> 8;
    msg->m[msg->blocks * BlockSize - 1] = size;
}

__device__ __forceinline__ void init(Hash *h) {
    memcpy(h, H0, HashSize);
}

__device__ __forceinline__ void bswapHash(Hash *hash) {
    hash->H[0] = BSWAP_32(hash->H[0]);
    hash->H[1] = BSWAP_32(hash->H[1]);
    hash->H[2] = BSWAP_32(hash->H[2]);
    hash->H[3] = BSWAP_32(hash->H[3]);
    hash->H[4] = BSWAP_32(hash->H[4]);
    hash->H[5] = BSWAP_32(hash->H[5]);
    hash->H[6] = BSWAP_32(hash->H[6]);
    hash->H[7] = BSWAP_32(hash->H[7]);
}

__device__ void block(Hash *hash, Message *msg) {
    uint32_t W[64], T1, T2;
    uint32_t a, b, c, d, e, f, g, h;
    for (uint i = 0; i < msg->blocks; ++i) {
        // printf("DBG block: %d\n", i);
// Step 1
#pragma unroll 16
        for (uint t = 0; t < 16; ++t) {
            W[t] = BSWAP_32(msg->mi[t + BlockSizeInt32 * i]);
            // printf("DBG W(%d): %x\n", t, W[t]);
        }
#pragma unroll 48
        for (uint t = 16; t < 64; ++t) {
            W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16];
            // printf("DBG W(%d): %x\n", t, W[t]);
        }
// Step 2
        a = hash->H[0];
        b = hash->H[1];
        c = hash->H[2];
        d = hash->H[3];
        e = hash->H[4];
        f = hash->H[5];
        g = hash->H[6];
        h = hash->H[7];
// Step 3
#pragma unroll 64
        for (uint t = 0; t < 64; ++t) {
            T1 = h + S1(e) + Ch(e, f, g) + K[t] + W[t];
            T2 = S0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }
// Step 4
        hash->H[0] = a + hash->H[0];
        hash->H[1] = b + hash->H[1];
        hash->H[2] = c + hash->H[2];
        hash->H[3] = d + hash->H[3];
        hash->H[4] = e + hash->H[4];
        hash->H[5] = f + hash->H[5];
        hash->H[6] = g + hash->H[6];
        hash->H[7] = h + hash->H[7];
        // printf("DBG a: %x\n", hash->H[0]);
    }
}

__global__ void kernelSha256(Password *pwd, Hash *hash) {
    Message m;
    m.m = (unsigned char *)malloc(TwoBlockSize);
    for (int i = pwd->L; i < TwoBlockSize; ++i) {
        m.m[i] = 0;
    }
    memcpy(m.m, pwd->P, pwd->L);
    init(hash);
    setSize(&m, pwd->L);
    block(hash, &m);
    bswapHash(hash);
    free(m.m);
}

__global__ void kernelLenovoHash(Password *pwds, Password *pout) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    Password *pwd = &pwds[x];
    // printf("pwd: %02x (%d) / idx: %d\n", pwd->P[0], pwd->L, x);
    // printf("pwds: %s\n", pwds);
    Hash *hash1, *hash2;
    Message msg1, msg2;

    msg1.m = (unsigned char *)malloc(TwoBlockSize);
    for (int i=pwd->L; i < TwoBlockSize; ++i) {
        msg1.m[i] = 0;
    }
    memcpy(msg1.m, pwd->P, pwd->L);
    setSize(&msg1, Lenovo1MessageSize);
    hash1 = (Hash *)malloc(HashSize);
    init(hash1);
    block(hash1, &msg1);
    free(msg1.m);
    msg2.m = (unsigned char *) malloc(TwoBlockSize);
    for (int i=Lenovo2MessageSize; i < TwoBlockSize; ++i) {
        msg2.m[i] = 0;
    }
    // Only the first 12 are needed
    hash1->H[0] = BSWAP_32(hash1->H[0]);
    hash1->H[1] = BSWAP_32(hash1->H[1]);
    hash1->H[2] = BSWAP_32(hash1->H[2]);

    memcpy(msg2.m, hash1->H, LenovoHashOffset);
    memcpy(msg2.m + LenovoHashOffset, ModelSerial, ModelSerialSize);
    setSize(&msg2, Lenovo2MessageSize);
    hash2 = (Hash *)malloc(HashSize);
    init(hash2);
    block(hash2, &msg2);
    // bswapHash(hash2);
    free(msg2.m);
    free(hash1);

    int i = 0, equal = 1;
    do {
        if (hash2->H[i] != TargetHash.H[i]) {
            // printf("h2: %x / th: %x\n", hash2->H[i], TargetHash.H[i]);
            equal = 0;
        }
        i++;
    } while(equal && i < 8);

    // printf("pout: %s\n", pwd->P);
    if (equal) {
        memcpy(pout, pwd, PasswordSize);
    }
    free(hash2);
}

extern "C" void gpuLenovoHash(Password *hostPwds, Password *hostPout, Hash *targetHash, const char * modelSerial, uint32_t blocks, uint32_t threads, uint32_t passwords) {
    uint HostPasswordSize = sizeof(Password) * passwords;
    uint HostPasswordOut = sizeof(Password);
    uint HostHashSize = sizeof(Hash);

    Password *pwds, *pout;

    // printf("Passwords %d\n", passwords);
    // printf("In %x (%d)\n", *hostPwds->P, hostPwds->L);
    cudaMemcpyToSymbol(TargetHash.H, targetHash->H, HostHashSize);
    cudaMemcpyToSymbol(ModelSerial, modelSerial, ModelSerialSize);

    cudaMalloc(&pwds, HostPasswordSize);
    cudaMalloc(&pout, HostPasswordOut);

    cudaMemcpy(pwds, hostPwds, HostPasswordSize, cudaMemcpyHostToDevice);
    cudaMemcpy(pout, hostPout, HostPasswordOut, cudaMemcpyHostToDevice);

    kernelLenovoHash <<<blocks,threads>>>(pwds, pout);
    cudaMemcpy(hostPout, pout, HostPasswordOut, cudaMemcpyDeviceToHost);
    // printf("In %x (%d)\n", hostPwds->P[0], hostPwds->L);
    // printf("Out %s (%d)\n", hostPout->P, hostPout->L);

    cudaFree(pout);
    cudaFree(pwds);
}

extern "C" void sha256Digest(Password *pwd, Hash *hash) {
    kernelSha256<<<1024 * 8 * 16,256>>>(pwd, hash);
};
