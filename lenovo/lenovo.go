package lenovo

import (
	"github.com/juanmera/lenovata/intelsha"
	"unsafe"
)

const ScanCodeIndex = "##1234567890####qwertyuiop####asdfghjkl;####zxcvbnm###### "
var Charset string

type Range struct {
    Start byte
    End byte
}

type Hash struct {
    dataA [128]byte
    dataB [128]byte
    digestA [128]byte
    Digest [32]byte
    digestAS []byte
    digestBS []byte
    dataAS []byte
    dataBS []byte
    salt []byte
    pDataA unsafe.Pointer
    pDataB unsafe.Pointer
    pDigestA unsafe.Pointer
    pDigestB unsafe.Pointer
}

func New(salt []byte) *Hash {
	m := &Hash{
		salt: salt,
	}
	m.digestAS = m.digestA[:]
	m.digestBS = m.Digest[:]
	m.dataAS = m.dataA[:]
	m.dataBS = m.dataB[:]
	m.pDataA = unsafe.Pointer(&m.dataA)
	m.pDataB = unsafe.Pointer(&m.dataA)
	m.pDigestA = unsafe.Pointer(&m.digestA)
	m.pDigestB = unsafe.Pointer(&m.Digest)

	m.dataA[64] = 0x80
	m.dataA[126] = 0x02
	m.digestA[72] = 0x80
	m.digestA[126] = 0x02
	m.digestA[127] = 0x40
	return m
}

func (m *Hash) Hash(pwd []byte) {
	copy(m.digestAS, intelsha.InitialDigest)
	copy(m.digestBS, intelsha.InitialDigest)
	copy(m.dataAS, pwd)
    intelsha.Sha256RorxX8ms(m.pDataA, m.pDigestA, 2)
    copy(m.digestAS[12:], m.salt)
    intelsha.Sha256RorxX8ms(m.pDigestA, m.pDigestB, 2)
}

func DecodePassword(pwd []byte) (decoded string) {
    for _, i := range pwd {
        decoded += string(ScanCodeIndex[i])
    }
    return decoded
}

func init() {
    Charset = genScanCodeCharset()
}

func genScanCodeCharset() string {
    var charset [38]byte
    c := 0
    ranges := []Range{
        {0x02, 0x0b},
        {0x10, 0x19},
        {0x1e, 0x1f},
        {0x20, 0x27},
        {0x2c, 0x32},
        {0x39, 0x39},
    }
    for _, r := range ranges {
        for i := r.Start; i <= r.End; i++ {
            charset[c] = i
            c += 1
        }
    }
    return string(charset[:])
}
