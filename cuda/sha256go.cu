#include "sha256go.cuh"

extern "C" void sha256(unsigned char *out) {
// int main(void) {
	int i,j;
	static Sha256Password *inbuffer, *initbuff;			/** binary ciphertexts **/
	static Sha256Digest *outbuffer, *printbuf;				/** calculated hashes **/

	inbuffer  = (Sha256Password *) cuda_pageLockedMalloc(inbuffer,  sizeof(Sha256Password) * KEYS_PER_CRYPT);
	outbuffer = (Sha256Digest *) cuda_pageLockedMalloc(outbuffer, sizeof(Sha256Digest) * KEYS_PER_CRYPT);
	// outbuffer = (Sha256Password *) cuda_pageLockedMalloc(outbuffer, sizeof(Sha256Password));
	// for (i =0; i< 10000; ++i) {
	initbuff = inbuffer;
	for (j=0; j < KEYS_PER_CRYPT; ++j) {
		initbuff->v[0] = 'a' + (j % 26);
		initbuff->length = 1;
		initbuff++;

	}
	initbuff = NULL;
	gpuSha256(inbuffer, outbuffer);
		// printf("%x\n", outbuffer->v[0]);
	// }

	printbuf = outbuffer;
	for (i=0; i<32; i+=8) {
		out[i]   = printbuf->hb[i+3];
		out[i+1] = printbuf->hb[i+2];
		out[i+2] = printbuf->hb[i+1];
		out[i+3] = printbuf->hb[i];
	}
	// for (i = 0; i < KEYS_PER_CRYPT; ++i) {
	// 	// printf("%d %x\n", i, printbuf->v[0]);
	// 	printbuf++;
	// }
	// printbuf = NULL;

	cuda_pageLockedFree(inbuffer);
	cuda_pageLockedFree(outbuffer);
	return;
}

