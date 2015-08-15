package intelsha

import (
    "testing"
    "bytes"
    "crypto/sha256"
)

func TestSum(t *testing.T) {
    h := New()
    h.SetDataSize(1)
    testHash(t, h, "a")
    h.SetDataSize(2)
    testHash(t, h, "77")
    h.SetDataSize(31)
    testHash(t, h, "0123456789012345678901234567890")
    h.SetDataSize(32)
    testHash(t, h, "01234567890123456789012345678901")
    h.SetDataSize(64)
    testHash(t, h, "0123456789012345678901234567890123456789012345678901234567890123")
}

func testHash(t *testing.T, h *Hash, data string) {
    bdata := []byte(data)
    digest := h.Digest(bdata)
    expectedDigest := sha256.Sum256(bdata)
    if !bytes.Equal(digest[:], expectedDigest[:]) {
        t.Logf("%s:\n%x\n%x\n", data, digest, expectedDigest)
        t.Fail()
    }
}

func BenchmarkIntelSha(b *testing.B) {
    for i := 0; i < b.N; i++ {
        h := New()
        h.SetDataSize(64)
        h.Digest([]byte("0123456789012345678901234567890123456789012345678901234567890123"))
    }
}

func BenchmarkIntelShaMem(b *testing.B) {
    h := New()
    h.SetDataSize(64)
    for i := 0; i < b.N; i++ {
        h.Digest([]byte("0123456789012345678901234567890123456789012345678901234567890123"))
    }
}

func BenchmarkCryptSha(b *testing.B) {
    for i := 0; i < b.N; i++ {
        sha256.Sum256([]byte("0123456789012345678901234567890123456789012345678901234567890123"))
    }
}
