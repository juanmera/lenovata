package lenovo

import (
    "testing"
    "encoding/hex"
    "bytes"
)

var MNSN = []byte("                        TESTING 1234                        ")
func TestHashesOpt(t *testing.T) {
    testHashOpt(t, "\x1e", "fba0d8dc6a06deb95033c8f32f49a98905d6ea1dd529ef754dbfe7eb4356e6fd")
    testHashOpt(t, "\x14\x12\x1f\x14", "d2f53522816ec95f7b00e669cb8ad5f55d8417b06585d6e21e875520f1fde903")
    testHashOpt(t, "\x17\x18\x02\x03\x04", "60ea8d51ff419b7fa151152d8d1185532370e5bb829b1c06da88065387e61b22")
}

func testHashOpt(t *testing.T, input, expectedHex string) {
    h := New(MNSN)
    expected, _ := hex.DecodeString(expectedHex)
    h.Hash([]byte(input))
    if !bytes.Equal(h.Digest[:], expected) {
        t.Logf("%x\n \t   %x\n", h.Digest, expected)
        t.Fail()
    }
}


func BenchmarkHash(b *testing.B) {
    var expected [32]byte
    h := New(MNSN)
    hex.Decode(expected[:], []byte("fba0d8dc6a06deb95033c8f32f49a98905d6ea1dd529ef754dbfe7eb4356e6fd"))
    for i := 0; i < b.N; i++ {
        h.Hash([]byte("\x1e"))
        if h.Digest != expected {
            b.Logf("%x\n \t   %x\n", h.Digest, expected)
            b.Fail()
        }
    }
}
