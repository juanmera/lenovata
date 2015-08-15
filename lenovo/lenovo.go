package lenovo

const ScanCodeIndex = "##1234567890####qwertyuiop####asdfghjkl;####zxcvbnm###### "
var Charset string

type Sha256Digester interface {
    Digest([]byte) [32]byte
}

type Range struct {
    Start byte
    End byte
}

func Hash(hi, ho Sha256Digester, sndPwd, pwd []byte) [32]byte {
    digest := hi.Digest(pwd)
    copy(sndPwd, digest[:12])
    return ho.Digest(sndPwd)
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
