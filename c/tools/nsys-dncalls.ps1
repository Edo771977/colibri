# nsys-dncalls.ps1 -- scompone le chiamate GPU di DeltaNet (dnproj e dnout)
# di una traccia Nsight Systems del motore qwen36. Tappa 0 di
# docs/qwen36-deltanet-gpu-plan.md: quanto di ogni chiamata NON e' kernel.
#
# SI LANCIA DA cmd, dalla cartella c\, dopo
#     nsys profile -t cuda ... -o s0 qwen36_clang.exe 256 4 prompt25.txt > s0-nsys.log 2>&1
#     nsys stats --report cuda_gpu_trace --format csv --output s0 s0.nsys-rep
# con UNA riga:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\nsys-dncalls.ps1 -Csv s0_cuda_gpu_trace.csv -Log s0-nsys.log [-BaseLog s0-base.log]
#
# COSA CERCA. Una chiamata densa (coli_cuda_matmul, pageable) e', sulla
# timeline della GPU e sul suo stream: una o piu' copie Host-to-Device, UN
# kernel, una o piu' copie Device-to-Host, in fila. Lo script riconosce le
# chiamate di decode dai byte copiati, con le forme di Qwen3.6-35B-A3B:
#   dnproj   su 2048 float (8 KiB), giu' 12288 float (48 KiB)
#   dnout    su 4096 float (16 KiB), giu' 2048 float (8 KiB), subito dopo
#            un dnproj sullo stesso stream (stesso layer)
#   attnout  stessa forma di dnout, ma non preceduto da un dnproj
# Il prefill (piu' righe per copia) e le altre chiamate non combaciano e
# vengono solo contate. Nessun kernel e' riconosciuto dal nome o dalla
# griglia: la larghezza R di COLI_CUDA_I8_ROWS cambia la griglia, non i byte.
#
# COSA STAMPA, in microsecondi, per dnproj e dnout: copia su, attesa fra
# copia e kernel, kernel, attesa fra kernel e copia giu', copia giu', span
# (inizio della copia su -> fine della copia giu'); per dnout anche la
# pausa dello stream prima della chiamata, cioe' il lavoro CPU di DeltaNet
# fra le due chiamate dello stesso layer.
# Con -Log (il log della corsa profilata, COLI_TIMERS=1): il tempo a parete
# medio per chiamata dai timer dn-split del motore, e quanto ne resta fuori
# dallo span e fuori dal kernel. Con -BaseLog (una corsa uguale ma NON
# profilata, subito prima): lo stesso a parete senza il profiler, contro il
# kernel della traccia.
#
# LIMITI. I tempi della traccia sono della GPU; la parete e' del motore, una
# media sulle chiamate di decode. "Fuori dallo span" comprende la copia di
# staging del driver per la memoria pageable, il ritorno dalle attese
# sincrone e i controlli di coli_cuda_matmul: la traccia CUDA da sola non li
# separa. Il profiler allunga le chiamate API: per questo esiste -BaseLog.

param(
    [Parameter(Mandatory = $true)] [string] $Csv,
    [string] $Log     = "",
    [string] $BaseLog = ""
)

$ErrorActionPreference = "Stop"
$inv = [Globalization.CultureInfo]::InvariantCulture

# Forme di Qwen3.6-35B-A3B (byte per chiamata di decode, S = 1).
$DnprojUp = 2048 * 4;  $DnprojDown = 12288 * 4
$OutUp    = 4096 * 4;  $OutDown    = 2048 * 4

if (-not (Test-Path -LiteralPath $Csv)) { throw "manca $Csv -- genera il CSV con: nsys stats --report cuda_gpu_trace --format csv --output s0 s0.nsys-rep" }
foreach ($f in @($Log, $BaseLog)) {
    if ($f -and -not (Test-Path -LiteralPath $f)) { throw "manca $f" }
}

# ---- colonne e unita' ------------------------------------------------------
$rows = @(Import-Csv -LiteralPath $Csv)
if ($rows.Count -eq 0) { throw "$Csv e' vuoto: la traccia non contiene attivita' CUDA." }
$cols = @($rows[0].PSObject.Properties.Name)
function Find-Col([string]$pattern) {
    $c = @($cols | Where-Object { $_ -like $pattern })
    if ($c.Count -eq 0) { throw ("colonna '{0}' assente in {1}. Colonne trovate: {2}" -f $pattern, $Csv, ($cols -join ", ")) }
    $c[0]
}
$cStart = Find-Col "Start*"
$cDur   = Find-Col "Duration*"
$cBytes = Find-Col "Bytes*"
$cGrid  = Find-Col "GrdX*"
$cStrm  = Find-Col "Strm*"
$cName  = Find-Col "Name*"

# Unita' dall'intestazione: "Start (ns)", "Bytes (MB)". Senza unita' riconosciuta
# lo script si ferma invece di indovinare un fattore mille.
function Get-Unit([string]$col, [hashtable]$table) {
    if ($col -match '\(([^)]+)\)') {
        $u = $Matches[1].Trim()
        foreach ($k in $table.Keys) { if ([string]::Equals($k, $u, [StringComparison]::Ordinal)) { return $table[$k] } }
    }
    throw ("unita' non riconosciuta nella colonna '{0}' di {1}." -f $col, $Csv)
}
$tUs = Get-Unit $cStart @{ "ns" = 0.001; "us" = 1.0; "ms" = 1000.0 }
if ((Get-Unit $cDur @{ "ns" = 0.001; "us" = 1.0; "ms" = 1000.0 }) -ne $tUs) { throw "Start e Duration hanno unita' diverse in $Csv." }
$bFactor = Get-Unit $cBytes @{ "B" = 1.0; "KB" = 1e3; "KiB" = 1024.0; "MB" = 1e6; "MiB" = 1048576.0; "GB" = 1e9 }

function To-Num([string]$s) {
    $v = 0.0
    if (-not [double]::TryParse($s, [Globalization.NumberStyles]::Float, $inv, [ref]$v)) { throw "valore non numerico '$s' in $Csv." }
    $v
}

# ---- operazioni per stream -------------------------------------------------
$byStream = @{}
$nOps = 0
foreach ($r in $rows) {
    $name = [string]$r.$cName
    $grid = [string]$r.$cGrid
    if ($grid.Trim() -ne "")                               { $kind = "K" }
    elseif ($name -match 'Host-to-Device|HtoD')            { $kind = "H" }
    elseif ($name -match 'Device-to-Host|DtoH')            { $kind = "D" }
    else                                                   { $kind = "O" }
    $s = (To-Num $r.$cStart) * $tUs
    $d = (To-Num $r.$cDur) * $tUs
    $b = 0.0
    if ($kind -eq "H" -or $kind -eq "D") { $b = (To-Num $r.$cBytes) * $bFactor }
    $key = [string]$r.$cStrm
    if (-not $byStream.ContainsKey($key)) { $byStream[$key] = New-Object System.Collections.Generic.List[object] }
    $byStream[$key].Add([pscustomobject]@{ S = $s; E = $s + $d; D = $d; B = $b; Kind = $kind })
    $nOps++
}

# Il CSV ha i byte arrotondati (0.008 MB per 8192 B): tolleranza di 600 B o 3 %.
function Near([double]$got, [double]$want) { [Math]::Abs($got - $want) -le [Math]::Max(600.0, 0.03 * $want) }

$calls = New-Object System.Collections.Generic.List[object]
$other = 0
foreach ($key in $byStream.Keys) {
    $ops = @($byStream[$key] | Sort-Object S)
    $prev = $null
    for ($i = 0; $i -lt $ops.Count; $i++) {
        if ($ops[$i].Kind -ne "K") { continue }
        $j = $i - 1; while ($j -ge 0 -and $ops[$j].Kind -eq "H") { $j-- }
        $k = $i + 1; while ($k -lt $ops.Count -and $ops[$k].Kind -eq "D") { $k++ }
        if ($j -eq $i - 1 -or $k -eq $i + 1) { continue }   # non e' un giro H2D -> kernel -> D2H
        $h = $ops[($j + 1)..($i - 1)]
        $dn = $ops[($i + 1)..($k - 1)]
        $hb = ($h | Measure-Object -Property B -Sum).Sum
        $db = ($dn | Measure-Object -Property B -Sum).Sum
        $c = [pscustomobject]@{
            Strm = $key; Class = ""
            HS = $h[0].S; HE = $h[-1].E; KS = $ops[$i].S; KE = $ops[$i].E; DS = $dn[0].S; DE = $dn[-1].E
            Lead = $null
        }
        if ($prev) { $c.Lead = $c.HS - $prev.DE }
        if ((Near $hb $DnprojUp) -and (Near $db $DnprojDown)) { $c.Class = "dnproj" }
        elseif ((Near $hb $OutUp) -and (Near $db $OutDown)) {
            if ($prev -and $prev.Class -eq "dnproj") { $c.Class = "dnout" } else { $c.Class = "attnout" }
        } else { $other++ }
        $calls.Add($c)
        $prev = $c
        $i = $k - 1
    }
}

function Get-Stats([double[]]$v) {
    $a = @($v | Sort-Object)
    $n = $a.Count
    if ($n -eq 0) { return $null }
    $pick = { param($q) $a[[Math]::Min($n - 1, [Math]::Floor($q * ($n - 1) + 0.5))] }
    [pscustomobject]@{
        N = $n; Mean = ($a | Measure-Object -Average).Average
        Med = & $pick 0.5; P10 = & $pick 0.1; P90 = & $pick 0.9; Min = $a[0]
    }
}
function Fmt([double]$x) { $x.ToString("0.0", $inv) }
function Row([string]$label, [double[]]$v) {
    $st = Get-Stats $v
    if (-not $st) { return }
    "  {0,-40} {1,6} {2,8} {3,8} {4,8} {5,8} {6,8}" -f $label, $st.N, (Fmt $st.Mean), (Fmt $st.Med), (Fmt $st.P10), (Fmt $st.P90), (Fmt $st.Min)
}

$dnproj = @($calls | Where-Object { $_.Class -eq "dnproj" })
$dnout  = @($calls | Where-Object { $_.Class -eq "dnout" })
$attn   = @($calls | Where-Object { $_.Class -eq "attnout" })

"traccia: $Csv | $nOps operazioni GPU su $($byStream.Count) stream"
"giri H2D -> kernel -> D2H trovati: dnproj $($dnproj.Count), dnout $($dnout.Count), attnout $($attn.Count), altri $other (prefill, lm_head, attnproj...)"
if ($dnproj.Count -eq 0) {
    throw "nessuna chiamata dnproj di decode nella traccia: il motore non ha messo dnproj sulla GPU, oppure le forme non sono quelle di Qwen3.6-35B-A3B."
}
""
"  microsecondi                                  n    media  mediana      p10      p90      min"
foreach ($set in @(@("dnproj", $dnproj), @("dnout", $dnout))) {
    $nm = $set[0]; $cs = @($set[1])
    if ($cs.Count -eq 0) { "  ${nm}: nessuna chiamata"; continue }
    Row "$nm  copia su (H2D)"            ($cs | ForEach-Object { $_.HE - $_.HS })
    Row "$nm  attesa copia su -> kernel" ($cs | ForEach-Object { $_.KS - $_.HE })
    Row "$nm  kernel"                    ($cs | ForEach-Object { $_.KE - $_.KS })
    Row "$nm  attesa kernel -> copia giu" ($cs | ForEach-Object { $_.DS - $_.KE })
    Row "$nm  copia giu (D2H)"           ($cs | ForEach-Object { $_.DE - $_.DS })
    Row "$nm  span (inizio su -> fine giu)" ($cs | ForEach-Object { $_.DE - $_.HS })
    if ($nm -eq "dnout") {
        Row "dnout  pausa prima (CPU fra le chiamate)" ($cs | Where-Object { $null -ne $_.Lead } | ForEach-Object { $_.Lead })
    }
}

# ---- confronto con i timer del motore --------------------------------------
function Read-Timers([string]$path) {
    $t = Get-Content -LiteralPath $path -Raw
    $o = [ordered]@{ Tokens = $null; Qkvz = $null; Out = $null; NProj = $null; NOut = $null; NCalls = $null; Step = $null
                     KeepAlive = $t.Contains("[cuda] keep-alive active:"); Heat = $t.Contains("[qtier] HEAT_FILE loaded:") }
    if ($t -match '\[timers\] decode: (\d+) tokens') { $o.Tokens = [int]$Matches[1] }
    if ($t -match 'dn-split: qkvz ([0-9.]+) \| a\+b [0-9.]+ \| norm [0-9.]+ \| out ([0-9.]+) ms/token') {
        $o.Qkvz = [double]::Parse($Matches[1], $inv); $o.Out = [double]::Parse($Matches[2], $inv)
    }
    if ($t -match 'dn-gpu: dnproj (\d+)/(\d+) \| dnout (\d+)/(\d+) calls') {
        $o.NProj = [int]$Matches[1]; $o.NCalls = [int]$Matches[2]; $o.NOut = [int]$Matches[3]
    }
    if ($t -match 'step\(\) total: ([0-9.]+) ms/token') { $o.Step = [double]::Parse($Matches[1], $inv) }
    foreach ($k in @("Tokens", "Qkvz", "NProj")) {
        if ($null -eq $o[$k]) { throw "$path non contiene le righe [timers] decode / dn-split / dn-gpu: la corsa va fatta con COLI_TIMERS=1 e un motore con i timer dn-split (#58)." }
    }
    [pscustomobject]$o
}

function Compare-Wall([string]$tag, $tm, [double]$kProj, [double]$sProj, [double]$kOut, [double]$sOut) {
    "{0}: {1} token di decode, step() {2} ms/token, keep-alive {3}, tabella heat {4}" -f $tag, $tm.Tokens,
        $(if ($null -ne $tm.Step) { Fmt $tm.Step } else { "?" }),
        $(if ($tm.KeepAlive) { "attivo" } else { "NON attivo" }), $(if ($tm.Heat) { "caricata" } else { "NON caricata" })
    if ($tm.NProj -ne $tm.NCalls -or $tm.NOut -ne $tm.NCalls) {
        "  ATTENZIONE: dn-gpu dnproj $($tm.NProj)/$($tm.NCalls), dnout $($tm.NOut)/$($tm.NCalls): alcune chiamate sono andate sulla CPU, la media a parete le mescola. Confronto saltato."
        return
    }
    $wProj = $tm.Qkvz * 1000.0 * $tm.Tokens / $tm.NCalls
    $wOut  = $tm.Out  * 1000.0 * $tm.Tokens / $tm.NCalls
    "  per chiamata (us)        parete   span GPU   kernel   fuori dallo span   non-kernel"
    "  dnproj                 {0,8} {1,10} {2,8} {3,18} {4,12}" -f (Fmt $wProj), (Fmt $sProj), (Fmt $kProj), (Fmt ($wProj - $sProj)), (Fmt ($wProj - $kProj))
    "  dnout                  {0,8} {1,10} {2,8} {3,18} {4,12}" -f (Fmt $wOut), (Fmt $sOut), (Fmt $kOut), (Fmt ($wOut - $sOut)), (Fmt ($wOut - $kOut))
    "  ({0} layer DeltaNet per token: non-kernel {1} ms/token in tutto)" -f ($tm.NCalls / $tm.Tokens).ToString("0.##", $inv),
        (Fmt ((($wProj - $kProj) + ($wOut - $kOut)) * $tm.NCalls / $tm.Tokens / 1000.0))
}

if ($Log -or $BaseLog) {
    $kProj = ($dnproj | ForEach-Object { $_.KE - $_.KS } | Measure-Object -Average).Average
    $sProj = ($dnproj | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average
    $kOut = 0.0; $sOut = 0.0
    if ($dnout.Count) {
        $kOut = ($dnout | ForEach-Object { $_.KE - $_.KS } | Measure-Object -Average).Average
        $sOut = ($dnout | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average
    }
    ""
    "medie della traccia contro la parete dei timer dn-split (medie anche loro):"
    if ($Log) {
        $tm = Read-Timers $Log
        Compare-Wall "corsa profilata ($Log)" $tm $kProj $sProj $kOut $sOut
        if ($tm.NProj -ne $dnproj.Count) {
            "  ATTENZIONE: la traccia ha $($dnproj.Count) dnproj di decode, il motore ne conta $($tm.NProj): traccia e log non sono della stessa corsa, o la forma di qualche chiamata non e' stata riconosciuta."
        }
    }
    if ($BaseLog) {
        $tb = Read-Timers $BaseLog
        Compare-Wall "corsa NON profilata ($BaseLog), kernel e span dalla traccia" $tb $kProj $sProj $kOut $sOut
    }
}
