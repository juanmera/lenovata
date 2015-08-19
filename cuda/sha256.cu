/*
* This software is Copyright (c) 2011-2012 Lukas Odzioba <ukasz at openwall dot net>
* and it is hereby released to the general public under the following terms:
* Redistribution and use in source and binary forms, with or without modification, are permitted.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "sha256.cuh"

__constant__ const uint32_t H[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


__constant__ const uint32_t K[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ void cudaSha256(Sha256Digest * out, Sha256Password * in);

__global__ void kernelSha256(Sha256Password * data, Sha256Digest * data_out) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	Sha256Digest *out = &data_out[idx];
	Sha256Password *in = &data[idx];
	cudaSha256(out, in);
}

__device__ void cudaSha256(Sha256Digest * out, Sha256Password * in) {

	uint32_t w[64];//this should be limited do 16 uints
	char dl = in->length;
	unsigned char *key = in->v;
	int j;
	for (j = 0; j < 15; j++) {
		w[j] = 0;
	}
	for (j = 0; j < dl; j++) {
		uint32_t tmp = 0;
		tmp |= (((uint32_t) key[j]) << ((3 - (j & 0x3)) << 3));
		w[j / 4] |= tmp;
	}
	w[dl / 4] |= (((uint32_t) 0x80) << ((3 - (dl & 0x3)) << 3));
	w[15] = 0x00000000 | (dl * 8);


	w[16] = sigma0(w[1]) + w[0];
	w[17] = sigma1(w[15]) + sigma0(w[2]) + w[1];
	w[18] = sigma1(w[16]) + sigma0(w[3]) + w[2];
	w[19] = sigma1(w[17]) + sigma0(w[4]) + w[3];
	w[20] = sigma1(w[18]) + sigma0(w[5]) + w[4];
	w[21] = sigma1(w[19]) + w[5];
	w[22] = sigma1(w[20]) + w[15];
	w[23] = sigma1(w[21]) + w[16];
	w[24] = sigma1(w[22]) + w[17];
	w[25] = sigma1(w[23]) + w[18];
	w[26] = sigma1(w[24]) + w[19];
	w[27] = sigma1(w[25]) + w[20];
	w[28] = sigma1(w[26]) + w[21];
	w[29] = sigma1(w[27]) + w[22];
	w[30] = sigma1(w[28]) + w[23] + sigma0(w[15]);
	w[31] = sigma1(w[29]) + w[24] + sigma0(w[16]) + w[15];

#pragma unroll 32
	for (uint32_t j = 32; j < 64; j++) {
		w[j] =
		    sigma1(w[j - 2]) + w[j - 7] + sigma0(w[j - 15]) + w[j -
		    16];
	}

	uint32_t a = H[0];
	uint32_t b = H[1];
	uint32_t c = H[2];
	uint32_t d = H[3];
	uint32_t e = H[4];
	uint32_t f = H[5];
	uint32_t g = H[6];
	uint32_t h = H[7];
#pragma unroll 64
	for (uint32_t j = 0; j < 64; j++) {
		uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + K[j] + w[j];
		uint32_t t2 = Sigma0(a) + Maj(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	out->h[0] = a + H[0];
	out->h[1] = b + H[1];
	out->h[2] = c + H[2];
	out->h[3] = d + H[3];
	out->h[4] = e + H[4];
	out->h[5] = f + H[5];
	out->h[6] = g + H[6];
	out->h[7] = h + H[7];
}

const uint32_t DATA_IN_SIZE = KEYS_PER_CRYPT * sizeof(Sha256Password);
const uint32_t DATA_OUT_SIZE = KEYS_PER_CRYPT * sizeof(Sha256Digest);
const uint32_t DATA_IN_SIZE_STREAM = DATA_IN_SIZE / STREAMS;
const uint32_t DATA_OUT_SIZE_STREAM = DATA_OUT_SIZE / STREAMS;
const uint32_t KEYS_PER_CRYPT_STREAM = KEYS_PER_CRYPT / STREAMS;

static cudaStream_t stream[STREAMS];	///streams for async cuda calls
static Sha256Password *asyncData[STREAMS];	///candidates
static Sha256Digest *asyncDataOut[STREAMS];	///sha256(candidates)

static void asyncSha256(Sha256Password * host_in, void *out) {
	int i;
	dim3 dimGrid(BLOCKS / STREAMS);
	dim3 dimBlock(THREADS);
	for (i=0; i < STREAMS; ++i) {
		HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		HANDLE_ERROR(cudaMalloc(&asyncData[i], DATA_IN_SIZE_STREAM));
		HANDLE_ERROR(cudaMalloc(&asyncDataOut[i], DATA_OUT_SIZE_STREAM));
		HANDLE_ERROR(cudaMemcpyAsync(asyncData[i], host_in + i *  KEYS_PER_CRYPT_STREAM, DATA_IN_SIZE_STREAM, cudaMemcpyHostToDevice, stream[i]));
		kernelSha256 <<< dimGrid, dimBlock, 0, stream[i] >>> (asyncData[i], asyncDataOut[i]);
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaMemcpyAsync((Sha256Digest *) out + i * KEYS_PER_CRYPT_STREAM, asyncDataOut[i], DATA_OUT_SIZE_STREAM, cudaMemcpyDeviceToHost, stream[i]));
	}
	for (i = 0; i < STREAMS; ++i) {
		HANDLE_ERROR(cudaStreamSynchronize(stream[i]));
		cudaStreamDestroy(stream[i]);
		cudaFree(asyncData[i]);
		cudaFree(asyncDataOut[i]);
	}
}

void gpuSha256(Sha256Password * i, Sha256Digest * o) {
		asyncSha256(i, o);
}

// static Sha256Password *serialData = NULL;	///candidates
// static Sha256Digest *serialDataOut = NULL;		///sha256(candidate) or sha224(candidate)

// static void serialSha256(Sha256Password * host_in, void *out) {
// 	Sha256Digest *host_out = (Sha256Digest *) out;
// 	cudaMalloc(&serialData, DATA_IN_SIZE);
// 	cudaMalloc(&serialDataOut, DATA_OUT_SIZE);
// 	cudaMemcpy(serialData, host_in, DATA_IN_SIZE, cudaMemcpyHostToDevice);

// 	kernelSha256 <<< BLOCKS, THREADS >>> (serialData, serialDataOut);
// 	cudaThreadSynchronize();
// 	HANDLE_ERROR(cudaGetLastError());

// 	cudaMemcpy(host_out, serialDataOut, DATA_OUT_SIZE, cudaMemcpyDeviceToHost);
// 	cudaFree(serialData);
// 	cudaFree(serialDataOut);
// }
