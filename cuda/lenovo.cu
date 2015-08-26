#include <stdio.h>
#include "sha256.cu"
#include "lenovo.cuh"

#define ModelSerialSize 60
#define LenovoHashOffset 12
#define Lenovo1MessageSize 64
#define TwoBlockSize 128

__constant__ unsigned char ModelSerial[ModelSerialSize];
__constant__ uint32_t TargetHash[32];

__device__ void set_msg(unsigned char *text, const void *msg, unsigned char msg_len, unsigned char offset, unsigned char size) {
    memcpy(text + offset, msg, msg_len);
    text[size] = 0x80;
    text[126] = size >> 5;
    text[127] = size << 3;
};

__global__ void kernel_lenovo_inline(const Password *pwds, Password *pout) {
    uint32_t hash[32], interm[32], W[64], T1, T2, msg[MessageAlloc32];
    uint x = blockDim.x * blockIdx.x + threadIdx.x;

// #pragma unroll 32
    for (int i = 0; i < MessageAlloc32; ++i) {
        msg[i] = 0;
    }
    set_msg((unsigned char *)&msg, pwds[x].P, pwds[x].L, 0, Lenovo1MessageSize);

// #pragma unroll 2
    for (uint li = 0; li < 2; ++li) {
        memcpy(hash, H0, 32);

// #pragma unroll 2
        for (uint i = 0; i < 2; ++i) {
#ifdef DEBUG
            printf("DBG sha256_block: %d\n", i);
#endif
    // Step 1
// #pragma unroll 16
            for (uint t = 0; t < 16; ++t) {
                W[t] = BSWAP_32(msg[t + BlockSizeInt32 * i]);
#ifdef DEBUG
                printf("DBG W(%d): %x\n", t, W[t]);
#endif
            }

// #pragma unroll 48
            for (uint t = 16; t < 64; ++t) {
                W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16];
#ifdef DEBUG
                printf("DBG W(%d): %x\n", t, W[t]);
#endif
            }
    // Step 2
            memcpy(interm, hash, 32);
    // Step 3
// #pragma unroll 64
            for (uint t = 0; t < 64; ++t) {
                T1 = interm[7] + S1(interm[4]) + Ch(interm[4], interm[5], interm[6]) + K[t] + W[t];
                T2 = S0(interm[0]) + Maj(interm[0], interm[1], interm[2]);
                interm[7] = interm[6];
                interm[6] = interm[5];
                interm[5] = interm[4];
                interm[4] = interm[3] + T1;
                interm[3] = interm[2];
                interm[2] = interm[1];
                interm[1] = interm[0];
                interm[0] = T1 + T2;
            }
    // Step 4
// #pragma unroll 8
            for (uint t = 0; t < 8; ++t) {
                hash[t] += interm[t];
            }
#ifdef DEBUG
            printf("DBG interm.H[0]: %x\n", hash[0]);
#endif
        }
        if (li == 0) {
            msg[0] = BSWAP_32(hash[0]);
            msg[1] = BSWAP_32(hash[1]);
            msg[2] = BSWAP_32(hash[2]);
            set_msg((unsigned char *)&msg, ModelSerial, ModelSerialSize, LenovoHashOffset, ModelSerialSize + LenovoHashOffset);
        }
    }

    if (
        hash[0] == TargetHash[0] &&
        hash[1] == TargetHash[1] &&
        hash[2] == TargetHash[2] &&
        hash[3] == TargetHash[3] &&
        hash[4] == TargetHash[4] &&
        hash[5] == TargetHash[5] &&
        hash[6] == TargetHash[6] &&
        hash[7] == TargetHash[7]
    ) {
        memcpy(pout, &pwds[x], PasswordSize);
    }
}


__global__ void kernel_lenovo(const Password *pwds, Password *pout) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    // printf("pwd: %02x (%d) / idx: %d\n", pwd->P[0], pwd->L, x);
    // printf("pwds: %s\n", pwds);
    Hash hash;
    Message msg;
    msg.set_msg(pwds[x].P, pwds[x].L, 0, Lenovo1MessageSize);

    sha256_init(&hash);
    sha256_block(&hash, &msg);

    // Only the first 12 are needed
    msg.u32[0] = BSWAP_32(hash.H[0]);
    msg.u32[1] = BSWAP_32(hash.H[1]);
    msg.u32[2] = BSWAP_32(hash.H[2]);
    msg.len = LenovoHashOffset;
    msg.set_msg(ModelSerial, ModelSerialSize, LenovoHashOffset);

    sha256_init(&hash);
    sha256_block(&hash, &msg);

    if (
        hash.H[0] == TargetHash[0] &&
        hash.H[1] == TargetHash[1] &&
        hash.H[2] == TargetHash[2] &&
        hash.H[3] == TargetHash[3] &&
        hash.H[4] == TargetHash[4] &&
        hash.H[5] == TargetHash[5] &&
        hash.H[6] == TargetHash[6] &&
        hash.H[7] == TargetHash[7]
    ) {
        memcpy(pout, &pwds[x], PasswordSize);
    }
}

#define STREAMS 8
#define REPS 1000
static cudaStream_t stream[STREAMS];    ///streams for async cuda calls
static Password *pwds[STREAMS];
static Password *devPwds, *devPout;
static int setup = 0;
void lenovo_sum(Password *hostPwds, Password *hostPout, Hash *targetHash, const char * modelSerial, uint32_t blocks, uint32_t threads, uint32_t passwords) {
    int blocksPerStream = blocks / REPS / STREAMS;
    int pwdsPerStream = blocksPerStream * threads;
    int pwdsPerRep = pwdsPerStream * STREAMS;
    // int pwdsPerStreamSize = pwdsPerStream * HostPasswordSize;
    if (setup == 0) {
        setup = 1;
        cudaMemcpyToSymbol(TargetHash, targetHash->H, 32);
        cudaMemcpyToSymbol(ModelSerial, modelSerial, ModelSerialSize);
        cudaSetDeviceFlags(cudaDeviceMapHost);
        for (int i=0; i < STREAMS; ++i) {
            cudaStreamCreate(&stream[i]);
            cudaMalloc(&pwds[i], pwdsPerStream * HostPasswordSize);
        }
        // cudaMalloc(&devPwds, passwords * HostPasswordSize);
        // cudaMalloc(pwds, pwdsPerRep * HostPasswordSize);
    }
    cudaHostRegister(hostPout, HostPasswordSize, cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&devPout, hostPout, 0);


    for (int i=0; i < STREAMS; ++i) {
        cudaMemcpyAsync(pwds[i], hostPwds + i * pwdsPerStream, pwdsPerStream * HostPasswordSize, cudaMemcpyHostToDevice, stream[i]);
        kernel_lenovo_inline<<<blocksPerStream, threads, 0, stream[i]>>>(pwds[i], devPout);
    }

    for (int r=1; r < REPS; ++r) {
        for (int i=0; i < STREAMS; ++i) {
            cudaStreamSynchronize(stream[i]);
            if (hostPout->L == 0) {
                cudaMemcpyAsync(pwds[i], hostPwds + r * pwdsPerRep + i *  pwdsPerStream, pwdsPerStream * HostPasswordSize, cudaMemcpyHostToDevice, stream[i]);
                kernel_lenovo_inline<<<blocksPerStream, threads, 0, stream[i]>>>(pwds[i], devPout);
            } else {
                return;
            }
        }
    }


    for (int i=0; i < STREAMS; ++i) {
        cudaStreamSynchronize(stream[i]);
    }
    cudaHostUnregister(hostPout);
    if (hostPout->L > 0) {
        for (int i=0; i < STREAMS; ++i) {
            cudaStreamDestroy(stream[i]);
            cudaFree(pwds[i]);
        }
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

