#include "<stdint.h>"

typedef struct {
    unsigned char Len;
    unsigned char Word[8];
} Word;

typedef struct {
    unsigned char set[256];
    unsigned char len;
} Charset;

typedef struct {
    Charset charset;
    unsigned char len;
    uint32_t offset;

    __device__ Wordlist(Charset charset, unsigned char len, uint32_t offset=0) {
        this.charset = charset;
        this.len = len;
        this.offset = offset;

        for
    }

    __device__ void get_word(uint32_t n, uint32_t Word *word) {

    }
} Wordlist;

