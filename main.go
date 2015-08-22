package main

// #include "cuda/mySha256.h"
import "C"
import (
    "code.google.com/p/gcfg"
    "encoding/binary"
    "encoding/hex"
    "flag"
    "fmt"
    "github.com/juanmera/gordlist"
    "github.com/juanmera/lenovata/lenovo"
    "github.com/juanmera/lenovata/cuda"
    "io/ioutil"
    "os"
    "runtime"
    "strconv"
    "sync"
    "unsafe"
    "time"
)

type Config struct {
    Performance struct {
        Processes int
        CPUThreads int
        CudaThreads int
        GPUBlocks uint32
        GPUThreads uint32
        WordBuffer int
    }
    Bruteforce struct {
        MinLen uint64
        MaxLen uint64
    }
    Target struct {
        HashHex string
        ModelSerialHex string
    }
    Session struct {
        FileName string
        SaveTimeout uint64
    }
}

func main() {
    var offset uint64
    var configFile string
    flag.StringVar(&configFile, "config", "lenovata.cfg", "Config file")
    flag.Parse()

    var config Config
    gcfg.ReadFileInto(&config, configFile)

    if len(config.Target.HashHex) != 64 {
        panic("Hash must be 64 hex chars")
    }
    if len(config.Target.ModelSerialHex) != 120 {
        panic("Model and Serial numbers must be 120 hex chars")
    }

    progress, err := ioutil.ReadFile(config.Session.FileName)
    if os.IsNotExist(err) {
        offset = 0
    } else if err != nil {
        panic("Error in session file")
    } else {
        offset, _ = strconv.ParseUint(string(progress), 10, 64)
    }

    var targetHash [32]byte
    var wg sync.WaitGroup
    gordlist.Debug = true
    gordlist.Buffer = config.Performance.WordBuffer
    runtime.GOMAXPROCS(config.Performance.Processes)
    targetModelSerial, _ := hex.DecodeString(config.Target.ModelSerialHex)
    targetHashArray := hashArrayFromHex(config.Target.HashHex)
    hex.Decode(targetHash[:], []byte(config.Target.HashHex))

    fmt.Printf("LENOVATA\n")
    fmt.Printf("PERFORMANCE CPU Processes: %d CPU Threads: %d Cuda Threads: %d\n", config.Performance.Processes, config.Performance.CPUThreads, config.Performance.CudaThreads)
    fmt.Printf("PERFORMANCE GPU Blocks: %d Threads: %d\n", config.Performance.GPUBlocks, config.Performance.GPUThreads)
    fmt.Printf("BRUTEFORCE Min-Max Len: %d - %d Offset: %d\n", config.Bruteforce.MinLen, config.Bruteforce.MaxLen, offset)
    fmt.Printf("TARGET Hash: %s\n", config.Target.HashHex)
    fmt.Printf("TARGET Model & Serial: '%s'\n", targetModelSerial)
    fmt.Printf("\nStarting...\n\n")

    g := gordlist.New(lenovo.Charset)
    gcout := g.GenerateFrom(config.Bruteforce.MinLen, config.Bruteforce.MaxLen, offset)
    go saveProgress(config.Session.FileName, config.Session.SaveTimeout, g)
    wg.Add(config.Performance.CPUThreads + config.Performance.CudaThreads)
    for i := 0; i < config.Performance.CudaThreads; i++ {
        go func() {
            defer wg.Done()
            cudaCall := cuda.New(targetHashArray, targetModelSerial, config.Performance.GPUBlocks, config.Performance.GPUThreads)
            testHashCuda(gcout, cudaCall)
        }()
    }
    for i := 0; i < config.Performance.CPUThreads; i++ {
        go func() {
            defer wg.Done()
            testLenovoPassword(targetHash, targetModelSerial, gcout)
        }()
    }
    wg.Wait()
    fmt.Printf("Finish.")
}

func saveProgress(fileName string, timeout uint64, g *gordlist.Generator) {
    var progress string
    for {
        time.Sleep(time.Duration(timeout) * time.Second)
        progress = strconv.FormatUint(g.WordCount() - uint64(gordlist.Buffer), 10)
        ioutil.WriteFile(fileName, []byte(progress), 0644)
    }
}

func testHashCuda(gcout chan []byte, cudaCall *cuda.Call) {
    pwds := make([]cuda.Password, cudaCall.PasswordCount)
    i := uint32(0)
    for pwd := range gcout {
        pwds[i % cudaCall.PasswordCount] = cuda.NewPassword(pwd)
        i++
        if (i % cudaCall.PasswordCount) == 0 {
            cudaCall.Run(unsafe.Pointer(&pwds[0]))
            if cudaCall.PasswordOut.L > 0 {
                fmt.Printf("FOUND (GPU): (% x) %d\n", cudaCall.PasswordOut.P, C.int(cudaCall.PasswordOut.L))
                ioutil.WriteFile("FOUND.txt", []byte(fmt.Sprintf("% x (%d)\n", cudaCall.PasswordOut.P, cudaCall.PasswordOut.L)), 0644)
                os.Exit(0)
            }
        }

    }
}

func hashArrayFromHex(hashHex string) (hash [8]uint32) {
    for i := 0; i< 8; i++ {
        buf, _ := hex.DecodeString(hashHex[i*8:i*8+8])
        hash[i] = binary.BigEndian.Uint32(buf)
    }
    return hash
}


func testLenovoPassword(targetHash [32]byte, targetModelSerialHex []byte, in chan []byte) {
    h := lenovo.New(targetModelSerialHex)
    for pwd := range in {
        h.Hash(pwd)
            // FIXME: Use channels to handle success & terminate
        if h.Digest == targetHash {
            fmt.Printf("FOUND (CPU): (% x) %d\n", pwd, len(pwd))
            ioutil.WriteFile("FOUND.txt", []byte(fmt.Sprintf("% x (%d)\n", pwd, len(pwd))), 0644)
            os.Exit(0)
        }
    }
}
