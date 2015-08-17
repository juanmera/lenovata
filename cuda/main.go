// #include <cuda.h>
// #cgo LDFLAGS: -L/opt/cuda/lib64 -L/opt/cuda/lib -lcuda -lcudart
// #cgo CFLAGS: -I/opt/cuda/include


package main

// #cgo LDFLAGS:-fno-exceptions -L/opt/cuda/lib64 -L/opt/cuda/lib -lcudart -lcuda -lstdc++ sha256go.o sha256.o common.o
// #cgo CFLAGS: -fno-exceptions -I/opt/cuda/include
// #include "sha256go.h"
import "C"
import (
    // "unsafe"
    "fmt"
)

// func main() {
//     buf := C.CString(string(make([]byte, 256)))
//     C.cuDeviceGetName(buf, 256, C.CUdevice(0))
//     fmt.Println("Hello, your GPU is:", C.GoString(buf))
// }
func main() {
    var out [8]uint32
    C.sha256((*C.uint32_t)(&out[0]))
    fmt.Printf("%x\n", out)
}
