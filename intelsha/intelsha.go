package intelsha
// #cgo LDFLAGS: -m64 /home/odin/dev/juanmera/lenovata/intelsha/intelsha.o
import "C"
import (
    "unsafe"
)

const InitialDigest = "\x67\xe6\x09\x6a\x85\xae\x67\xbb\x72\xf3\x6e\x3c\x3a\xf5\x4f\xa5\x7f\x52\x0e\x51\x8c\x68\x05\x9b\xab\xd9\x83\x1f\x19\xcd\xe0\x5b"
const DefaultData = "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

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
    copy(m.digest[:], InitialDigest)
    copy(m.data[:], DefaultData)
}

func (m *Hash) Digest(data []byte) ([32]byte) {
    copy(m.digest[:], InitialDigest)
    copy(m.data[:], data)
    Sha256RorxX8ms(m.pData, m.pDigest, m.blocks)
    return m.digest
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

func Sha256RorxX8ms(data unsafe.Pointer, digest unsafe.Pointer, blocks uint64);


