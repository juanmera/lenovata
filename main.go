package main

import (
    "code.google.com/p/gcfg"
    "encoding/hex"
    "flag"
    "fmt"
    "github.com/juanmera/gordlist"
    "github.com/juanmera/lenovata/lenovo"
    "io/ioutil"
    "os"
    "runtime"
    "strconv"
    "sync"
    "time"
)

type Config struct {
    Performance struct {
        Processes int
        Routines int
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
    go saveProgress(config.Session.FileName, config.Session.SaveTimeout, g)
    for i := 0; i < config.Performance.Routines; i++ {
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

func testLenovoPassword(targetHash [32]byte, targetModelSerialHex []byte, in chan []byte) {
    h := lenovo.New(targetModelSerialHex)
    for pwd := range in {
        h.Hash(pwd)
            // FIXME: Use channels to handle success & terminate
        if h.Digest == targetHash {
            fmt.Printf("FOUND: (% x) %s\n", pwd, lenovo.DecodePassword(pwd))
            os.Exit(0)
        }
    }
}
