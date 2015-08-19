#include <stdio.h>
#include <stdlib.h>
#include "mySha256.cuh"

int main(void) {
    uint HostPasswordSize = sizeof(Password);
    uint HostHashSize = sizeof(Hash);

    Password *pwd, *cudaPwd;
    Hash *h, *cudaH;

    cudaHostAlloc((void **) &pwd, HostPasswordSize, cudaHostAllocDefault);
    cudaHostAlloc((void **) &h, HostHashSize, cudaHostAllocDefault);
    // malloc(&h, HostHashSize);
    cudaMalloc(&cudaPwd, HostPasswordSize);
    cudaMalloc(&cudaH, HostHashSize);
    pwd->p[0] = 'a';
    pwd->l = 1;

    cudaMemcpy(cudaPwd, pwd, HostPasswordSize, cudaMemcpyHostToDevice);
    sha256Digest(cudaPwd, cudaH);
    cudaMemcpy(h, cudaH, HostHashSize, cudaMemcpyDeviceToHost);
    for (int i=0; i<32; ++i) {
        printf("%02x ", h->hb[i]);
    }
    printf("\n");
    for (int i=0; i<8; ++i) {
        printf("%08x ", h->h[i]);
    }
    printf("\n");

    cudaFree(cudaH);
    cudaFree(cudaPwd);
    cudaFreeHost(h);
    cudaFreeHost(pwd);
    return 0;
}
