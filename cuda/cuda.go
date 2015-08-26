package cuda

// #cgo LDFLAGS:-fno-exceptions -L/opt/cuda/lib64 -L/opt/cuda/lib -lcudart -lcuda -lstdc++ /home/odin/dev/juanmera/lenovata/cuda/lenovo.o
// #cgo CFLAGS: -fno-exceptions -I/opt/cuda/include
// #include "lenovo.h"
import "C"
import (
    "unsafe"
)

type Password C.Password
type Hash C.Hash
type Uint32T C.uint32_t

type Call struct {
    targetHash *C.Hash
    pout *C.Password
    mnsn *C.char
    blocks, threads, passwordCount C.uint32_t
    PasswordOut Password
    PasswordCount uint32
}

func New(targetHash [8]uint32, mnsn []byte, blocks, threads uint32) *Call {
    passwordCount := blocks * threads
    c := &Call{
        targetHash: (*C.Hash)(unsafe.Pointer(&targetHash)),
        mnsn: (*C.char)(unsafe.Pointer(&mnsn[0])),
        blocks: C.uint32_t(blocks),
        threads: C.uint32_t(threads),
        passwordCount: C.uint32_t(passwordCount),
        PasswordCount: passwordCount,
    }
    c.pout = (*C.Password)(unsafe.Pointer(&c.PasswordOut))
    return c
}

func newHash(v [8]uint32) (h Hash) {
    for i := 0; i < 8; i++ {
        h.H[i] = C.uint32_t(v[i])
    }
    return h
}

func (c *Call) Run(pwds unsafe.Pointer) {
    C.lenovo_sum(
        (*C.Password)(pwds),
        c.pout,
        c.targetHash,
        c.mnsn,
        c.blocks,
        c.threads,
        c.passwordCount,
    )
}

func NewPassword(pwd []byte) (p Password) {
    p.L = C.uchar(len(pwd))
    for k, v := range pwd {
        p.P[k] = C.uchar(v)
    }
    return p
}

func PasswordGroup(pc chan []byte, size uint32) (chan []Password) {
    cout := make(chan []Password, 1)
    go func() {
        i := uint32(0)
        pwds := make([]Password, size)
        for pwd := range pc {
            pwds[i % size] = NewPassword(pwd)
            i++
            if (i % size) == 0 {
                cout <- pwds
                pwds = make([]Password, size)
            }
        }
        if (i % size) != 0 {
            cout <- pwds
        }
        close(cout)
    }()
    return cout
}

// func testMain() {
//     mnsn := []byte("                        TESTING 1234                        ")
//     pwds := []Password{
//         {L: C.uint32_t(1), P: [10]C.uchar{0x1e}},
//     }
//     targetHash := Hash{
//     [8]C.uint32_t{
//         0xfba0d8dc,
//         0x6a06deb9,
//         0x5033c8f3,
//         0x2f49a989,
//         0x05d6ea1d,
//         0xd529ef75,
//         0x4dbfe7eb,
//         0x4356e6fd,
//     }}
//     pout := LenovoHash(pwds, targetHash, mnsn, 1, 1, 1)
//     fmt.Printf("asd%s (%d)\n", pout.P, pout.L)
// }
