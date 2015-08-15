package main

import (
    "github.com/juanmera/gordlist"
    "code.google.com/p/gcfg"
    "github.com/juanmera/lenovata/intelsha"
    "github.com/juanmera/lenovata/lenovo"
    "fmt"
    "os"
    "sync"
    "flag"
    "time"
    "runtime"
    "encoding/hex"
)

type Config struct {
    Performance struct {
        Processes int
        Routines int
    }
    Bruteforce struct {
        MinLen uint64
        MaxLen uint64
    }
    Target struct {
        HashHex string
        ModelSerialHex string
    }
}


func main() {
    var offset uint64
    var configFile string
    flag.StringVar(&configFile, "config", "lenovata.cfg", "Config file")
    flag.Uint64Var(&offset, "offset", 0, "Offset")
    flag.Parse()

    var config Config
    gcfg.ReadFileInto(&config, configFile)
    fmt.Printf("%v\n", config)
    if len(config.Target.HashHex) != 64 {
        panic("Hash must be 64 hex chars")
    }
    if len(config.Target.ModelSerialHex) != 120 {
        panic("Model and Serial numbers must be 120 hex chars")
    }

    var targetHash [32]byte
    var wg sync.WaitGroup
    gordlist.Debug = true
    runtime.GOMAXPROCS(config.Performance.Processes)
    hex.Decode(targetHash[:], []byte(config.Target.HashHex))
    targetModelSerial, _ := hex.DecodeString(config.Target.ModelSerialHex)

    fmt.Printf("LENOVATA\n")
    fmt.Printf("PERFORMANCE Processes: %d Routines: %d\n", config.Performance.Processes, config.Performance.Routines)
    fmt.Printf("BRUTEFORCE Min-Max Len: %d - %d Offset: %d\n", config.Bruteforce.MinLen, config.Bruteforce.MaxLen, offset)
    fmt.Printf("TARGET Hash: %s\n", config.Target.HashHex)
    fmt.Printf("TARGET Model & Serial: '%s'\n", targetModelSerial)
    fmt.Printf("\nStarting...\n\n")

    g := gordlist.New(lenovo.Charset)
    gcout := g.GenerateFrom(config.Bruteforce.MinLen, config.Bruteforce.MaxLen, offset)
    wg.Add(config.Performance.Routines)
    go showWordCount(g)
    for i := 0; i < config.Performance.Routines; i++ {
        go func() {
            defer wg.Done()
            testLenovoPassword(targetHash, targetModelSerial, gcout)
        }()
    }
    wg.Wait()
    fmt.Printf("Finish.")
}

func showWordCount(g *gordlist.Generator) {
    for {
        time.Sleep(60 * time.Second)
        fmt.Printf("Offset: %d\n", g.WordCount());
    }
}

func testLenovoPassword(targetHash [32]byte, targetModelSerialHex []byte, in chan []byte) {
    var targetModelSerial [72]byte
    hex.Decode(targetModelSerial[12:], targetModelSerialHex)
    hi := intelsha.New()
    ho := intelsha.New()
    hi.SetDataSize(64)
    ho.SetDataSize(72)
    for pwd := range in {
        if lenovo.Hash(hi, ho, targetModelSerial[:], pwd) == targetHash {
            fmt.Printf("FOUND: (% x) %s\n", pwd, lenovo.DecodePassword(pwd))
            os.Exit(0)
        }
    }
}
