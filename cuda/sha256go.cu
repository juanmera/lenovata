#include "sha256go.h"

extern "C" void sha256(uint32_t *out) {
// int main(void) {
	int i,j;
	static sha256_password *inbuffer, *initbuff;			/** binary ciphertexts **/
	static SHA_HASH *outbuffer, *printbuf;				/** calculated hashes **/

	inbuffer = (sha256_password *) cuda_pageLockedMalloc(inbuffer, sizeof(sha256_password) * KEYS_PER_CRYPT);
	outbuffer = (SHA_HASH *) cuda_pageLockedMalloc(outbuffer, sizeof(SHA_HASH) * KEYS_PER_CRYPT);
	// for (i =0; i< 10000; ++i) {
	initbuff = inbuffer;
	for (j=0; j < KEYS_PER_CRYPT; ++j) {
		initbuff->v[0] = 'a' + (j % 26);
		initbuff->length = 1;
		initbuff++;

	}
	initbuff = NULL;
	gpu_rawsha256(inbuffer, outbuffer, 0);
		// printf("%x\n", outbuffer->v[0]);
	// }

	printbuf = outbuffer;
	for (i=0; i<8; ++i) {
		out[i] = printbuf->v[i];
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

