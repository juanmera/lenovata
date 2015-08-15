package intelsha

import (
    "unsafe"
)

var defaultDigest = []byte{
    0x67, 0xe6,0x09, 0x6a,
    0x85, 0xae,0x67, 0xbb,
    0x72, 0xf3,0x6e, 0x3c,
    0x3a, 0xf5,0x4f, 0xa5,
    0x7f, 0x52,0x0e, 0x51,
    0x8c, 0x68,0x05, 0x9b,
    0xab, 0xd9,0x83, 0x1f,
    0x19, 0xcd,0xe0, 0x5b,
}
var defaultData [128]byte

const chunk = 64
const BlockSize = 64


type Hash struct {
    digest [32]byte
    data [128]byte
    blocks uint64
    pData unsafe.Pointer
    pDigest unsafe.Pointer
}

func New() *Hash {
    m := &Hash{}
    m.pData = unsafe.Pointer(&m.data)
    m.pDigest = unsafe.Pointer(&m.digest)
    return m
}

func (m *Hash) reset() {
    copy(m.digest[:], defaultDigest)
    copy(m.data[:], defaultData[:])
}

func (m *Hash) Digest(data []byte) (digest [32]byte) {
    copy(m.digest[:], defaultDigest)
    copy(m.data[:], data)
    sha256RorxX8ms(m.pData, m.pDigest, m.blocks)

    digest[0],  digest[1],  digest[2],  digest[3]  = m.digest[3], m.digest[2], m.digest[1], m.digest[0]
    digest[4],  digest[5],  digest[6],  digest[7]  = m.digest[7], m.digest[6], m.digest[5], m.digest[4]
    digest[8],  digest[9],  digest[10], digest[11] = m.digest[11], m.digest[10], m.digest[9], m.digest[8]
    digest[12], digest[13], digest[14], digest[15] = m.digest[15], m.digest[14], m.digest[13], m.digest[12]
    digest[16], digest[17], digest[18], digest[19] = m.digest[19], m.digest[18], m.digest[17], m.digest[16]
    digest[20], digest[21], digest[22], digest[23] = m.digest[23], m.digest[22], m.digest[21], m.digest[20]
    digest[24], digest[25], digest[26], digest[27] = m.digest[27], m.digest[26], m.digest[25], m.digest[24]
    digest[28], digest[29], digest[30], digest[31] = m.digest[31], m.digest[30], m.digest[29], m.digest[28]
    return digest
}

func (m *Hash) SetDataSize(dataSize uint) {
    if dataSize > 2 * BlockSize - 9 {
        panic("Max data size 119")
    }
    m.reset()
    if dataSize < BlockSize - 8 {
        m.blocks = 1
    } else {
        m.blocks = 2
    }
    m.data[dataSize] = 0x80
    dataSize <<= 3
    m.data[m.blocks * BlockSize - 2] = byte(dataSize >> 8)
    m.data[m.blocks * BlockSize - 1] = byte(dataSize)
}

func sha256RorxX8ms(input_data unsafe.Pointer, digest unsafe.Pointer, num_blks uint64);


