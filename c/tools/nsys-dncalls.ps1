# nsys-dncalls.ps1 -- scompone le chiamate GPU di DeltaNet (dnproj e dnout)
# di una traccia Nsight Systems del motore qwen36. Tappa 0 di
# docs/qwen36-deltanet-gpu-plan.md: quanto di ogni chiamata NON e' kernel.
#
# SI LANCIA DA cmd, dalla cartella c\, dopo due corse uguali (COLI_TIMERS=1),
# la prima senza profiler, la seconda sotto Nsight:
#     qwen36_clang.exe 256 4 prompt25.txt > s0-base.log 2>&1
#     nsys profile -t cuda ... -o s0 qwen36_clang.exe 256 4 prompt25.txt > s0-nsys.log 2>&1
#     nsys stats --report cuda_gpu_trace --format csv --output s0 s0.nsys-rep
# con UNA riga:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\nsys-dncalls.ps1 -Csv s0_cuda_gpu_trace.csv -BaseLog s0-base.log
#
# Su Windows nsys NON scrive nella sua uscita quella del programma che
# profila: s0-nsys.log contiene solo le righe di nsys (Collecting data...,
# Generated ...), nessuna riga [timers] (28 settembre 2026; il 27 una
# ricerca nel log di nsys non aveva trovato nulla). Il log con i numeri del motore e' quindi quello della corsa senza
# profiler, -BaseLog. -Log resta per un log della corsa profilata che abbia
# davvero le righe [timers].
#
# COSA CERCA. Una chiamata densa (coli_cuda_matmul, pageable) e', sulla
# timeline della GPU e sul suo stream: una copia Host-to-Device, UN kernel
# quant_matmul*, una o piu' copie Device-to-Host. Lo script parte dal kernel
# e riconosce la chiamata dai byte copiati, con le forme di Qwen3.6-35B-A3B
# (S = 1):
#   forma P  su 2048 float (8 KiB), giu' 12288 float (48 KiB): dnproj
#   forma O  su 4096 float (16 KiB), giu' 2048 float (8 KiB): dnout o attnout
# Una chiamata di DECODE di DeltaNet e' una coppia: una forma P seguita, come
# chiamata successiva sullo stesso stream, da una forma O. Il prefill del
# motore passa anche lui da dnproj una riga alla volta, con le stesse copie
# del decode, ma il suo dnout resta sulla CPU: le sue forme P restano senza
# coppia e vengono solo contate. Una forma O senza P davanti e' un attnout.
# Il kernel si riconosce dal nome, non dalla griglia: la larghezza R di
# COLI_CUDA_I8_ROWS cambia la griglia, non il nome ne' i byte.
#
# COSA STAMPA, in microsecondi, per dnproj e dnout di decode: copia su,
# attesa fra copia e kernel, kernel, attesa fra kernel e copia giu', copia
# giu', span (inizio della copia su -> fine della copia giu'); per la coppia
# anche la pausa dello stream fra la fine di dnproj e l'inizio di dnout:
# il lavoro CPU di DeltaNet fra le due chiamate (a+b, gate, conv, ricorrenza,
# norma) PIU' la coda lato host di dnproj e la testa lato host di dnout.
# Le chiamate con altre operazioni GPU dentro lo span (i caricamenti di
# expert che il motore fa sullo stesso stream anche in decode) restano nelle
# medie, come restano nella parete del motore, e vengono contate.
# Con -BaseLog (una corsa uguale ma NON profilata, subito prima): il tempo a
# parete medio per chiamata dai timer dn-split del motore, contro il kernel
# e lo span della traccia, e quanto ne resta fuori dallo span e fuori dal
# kernel. Con -Log (la corsa profilata, se il suo log ha le righe [timers]):
# lo stesso con il profiler. Prima della tabella, un riepilogo per stream:
# operazioni, kernel e i nomi dei kernel piu' frequenti (per vedere, per
# esempio, se lo spin del keep-alive c'e'). Lo script avvisa (ATTENZIONE) se in un log mancano
# il keep-alive o la tabella heat, se c'e' lo staging pinned, se i due log
# non sono della stessa configurazione; salta il confronto se qualche
# chiamata e' andata sulla CPU o se traccia e log non contano le stesse
# chiamate. Il confronto dei conteggi si fa con ogni log passato: le due
# corse hanno la stessa configurazione, e il piazzamento non cambia fra una
# corsa e l'altra. Senza log non controlla niente del motore.
#
# LIMITI. I tempi della traccia sono della GPU; la parete e' del motore, una
# media sulle chiamate di decode. "Fuori dallo span" comprende tra l'altro
# la copia di staging del driver per la memoria pageable, la latenza di
# sottomissione delle chiamate API, il ritorno dalle attese sincrone, i
# controlli di coli_cuda_matmul, i due tm_now() del timer e l'attesa di un
# caricamento di expert ancora in corso sullo stream quando la chiamata
# comincia (per dnout finisce invece nella pausa): la traccia CUDA
# da sola non li separa. Il profiler allunga le chiamate API: per questo
# esiste -BaseLog.

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
$cDev   = @($cols | Where-Object { $_ -like "Device*" })[0]
$cCtx   = @($cols | Where-Object { $_ -like "Ctx*" })[0]
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
$nsInt = ($tUs -eq 0.001)
$bFactor = Get-Unit $cBytes @{ "B" = 1.0; "KB" = 1e3; "KiB" = 1024.0; "MB" = 1e6; "MiB" = 1048576.0; "GB" = 1e9 }
$bComma  = ($bFactor -ne 1.0)

# nsys scrive i decimali con il separatore della lingua di Windows: su un
# Windows italiano "Bytes (MB)" vale "0,008" e "508,559" (lm_head int8), fra
# virgolette nel CSV. Si accetta quindi il punto oppure, nella sola colonna
# dei byte e solo se l'unita' non e' il byte, UNA virgola decimale. Un
# separatore delle migliaia non si distingue da un decimale quando c'e' un
# solo gruppo ("1.234", "8,192"): per questo i tempi in ns devono essere
# interi (lo sono) e i byte in B non accettano la virgola.
function To-Num([string]$s, [string]$col, [switch]$Int, [switch]$Comma) {
    $v = 0.0
    $t = $s.Trim()
    if ($Int -and $t -notmatch '^-?\d+$') { throw "tempo non intero '$s' nella colonna '$col' di ${Csv}: in ns nsys scrive interi, un punto o una virgola qui sarebbe un separatore delle migliaia." }
    if ($Comma -and $t -match '^-?\d+,\d+$') { $t = $t.Replace(",", ".") }
    if ($t -notmatch '^-?\d+(\.\d+)?([eE][-+]?\d+)?$' -or
        -not [double]::TryParse($t, [Globalization.NumberStyles]::Float, $inv, [ref]$v)) {
        throw "valore non numerico '$s' nella colonna '$col' di $Csv (atteso un numero con il punto decimale, o con UNA virgola decimale nella colonna dei byte)."
    }
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
    $s = (To-Num $r.$cStart $cStart -Int:$nsInt) * $tUs
    $d = (To-Num $r.$cDur $cDur -Int:$nsInt) * $tUs
    $b = 0.0
    if ($kind -eq "H" -or $kind -eq "D") { $b = (To-Num $r.$cBytes $cBytes -Comma:$bComma) * $bFactor }
    # Lo stream 7 di due schede sono due stream: la chiave include scheda e contesto.
    $key = [string]$r.$cStrm
    if ($cDev) { $key = [string]$r.$cDev + "/" + $key }
    if ($cCtx) { $key = [string]$r.$cCtx + "/" + $key }
    if (-not $byStream.ContainsKey($key)) { $byStream[$key] = New-Object System.Collections.Generic.List[object] }
    $byStream[$key].Add([pscustomobject]@{ S = $s; E = $s + $d; D = $d; B = $b; Kind = $kind; Dense = ($kind -eq "K" -and $name -like "*quant_matmul*"); Name = $(if ($kind -eq "K") { $name } else { $null }) })
    $nOps++
}

# Il CSV ha i byte arrotondati (0.008 MB per 8192 B): tolleranza di 600 B o 5 %.
function Near([double]$got, [double]$want) { [Math]::Abs($got - $want) -le [Math]::Max(600.0, 0.05 * $want) }


# ---- chiamate dense, poi le coppie dnproj -> dnout ---------------------------
# Il perno e' il kernel denso (quant_matmul*). Sullo stesso stream 0 il thread
# che carica gli expert fa le sue copie sincrone (e offset_to_signed_s4)
# anche durante il decode, e possono cadere fra le copie di una chiamata: la
# copia su e' quindi la piu' vicina PRIMA del kernel con la dimensione di
# un input denso (8 o 16 KiB; gli expert copiano 512 e 64 KiB), la copia giu'
# la prima dopo, entrambe senza scavalcare un altro kernel denso. Le
# operazioni estranee dentro la chiamata sono contate.
$dnproj = New-Object System.Collections.Generic.List[object]
$dnout  = New-Object System.Collections.Generic.List[object]
$nP = 0; $nO = 0; $nX = 0; $nAttn = 0; $nLoneP = 0
foreach ($key in @($byStream.Keys)) {
    $ops = @($byStream[$key] | Sort-Object S)
    $dk = @(for ($i = 0; $i -lt $ops.Count; $i++) { if ($ops[$i].Dense) { $i } })
    $trips = New-Object System.Collections.Generic.List[object]
    for ($q = 0; $q -lt $dk.Count; $q++) {
        $i = $dk[$q]
        $lo = -1;          if ($q -gt 0)             { $lo = $dk[$q - 1] }
        $hi = $ops.Count;  if ($q + 1 -lt $dk.Count) { $hi = $dk[$q + 1] }
        $hIdx = -1; $suspect = -1
        for ($j = $i - 1; $j -gt $lo; $j--) {
            if ($ops[$j].Kind -ne "H" -or -not ((Near $ops[$j].B $DnprojUp) -or (Near $ops[$j].B $OutUp))) { continue }
            # Con expert int8 o int4 per riga la copia delle scale di un expert
            # e' di 8 KiB anche lei. Una copia subito dopo una copia grande o
            # dopo un kernel non denso (offset_to_signed_s4) e' sospetta: vale
            # solo se prima, fino al kernel denso precedente, non ce n'e'
            # un'altra (un caricamento finito appena prima della chiamata).
            if ($j -gt 0 -and (($ops[$j - 1].Kind -eq "H" -and $ops[$j - 1].B -gt 65536 * 1.05) -or
                               ($ops[$j - 1].Kind -eq "K" -and -not $ops[$j - 1].Dense))) {
                if ($suspect -lt 0) { $suspect = $j }
                continue
            }
            $hIdx = $j; break
        }
        if ($hIdx -lt 0) { $hIdx = $suspect }
        $dIdx = -1
        for ($j = $i + 1; $j -lt $hi; $j++) { if ($ops[$j].Kind -eq "D") { $dIdx = $j; break } }
        if ($hIdx -lt 0 -or $dIdx -lt 0) { $nX++; continue }
        $dEnd = $dIdx; while ($dEnd + 1 -lt $hi -and $ops[$dEnd + 1].Kind -eq "D") { $dEnd++ }
        $db = ($ops[$dIdx..$dEnd] | Measure-Object -Property B -Sum).Sum
        $hb = $ops[$hIdx].B
        $shape = "X"
        if     ((Near $hb $DnprojUp) -and (Near $db $DnprojDown)) { $shape = "P"; $nP++ }
        elseif ((Near $hb $OutUp)    -and (Near $db $OutDown))    { $shape = "O"; $nO++ }
        else { $nX++ }
        $trips.Add([pscustomobject]@{
            Shape = $shape; Foreign = ($dEnd - $hIdx + 1) - 2 - ($dEnd - $dIdx + 1)
            HS = $ops[$hIdx].S; HE = $ops[$hIdx].E; KS = $ops[$i].S; KE = $ops[$i].E; DS = $ops[$dIdx].S; DE = $ops[$dEnd].E
        })
    }
    for ($n = 0; $n -lt $trips.Count; $n++) {
        $t = $trips[$n]
        if ($t.Shape -eq "O") { $nAttn++; continue }          # O senza P davanti
        if ($t.Shape -ne "P") { continue }
        $u = $null; if ($n + 1 -lt $trips.Count) { $u = $trips[$n + 1] }
        if (-not $u -or $u.Shape -ne "O") { $nLoneP++; continue }
        $dnproj.Add($t)
        $dnout.Add([pscustomobject]@{ HS = $u.HS; HE = $u.HE; KS = $u.KS; KE = $u.KE; DS = $u.DS; DE = $u.DE; Foreign = $u.Foreign; Lead = $u.HS - $t.DE })
        $n++                                                   # la O e' consumata dalla coppia
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

# ---- log del motore: letti PRIMA, gli avvisi vanno in testa ------------------
function Read-Timers([string]$path) {
    $t = Get-Content -LiteralPath $path -Raw
    $o = [ordered]@{ Path = $path; Tokens = $null; Qkvz = $null; Out = $null; NProj = $null; NOut = $null; NCalls = $null; Step = $null
                     KeepAlive = $t.Contains("[cuda] keep-alive active:"); Heat = $t.Contains("[qtier] HEAT_FILE loaded:")
                     Pinned = $t.Contains("[cuda] dense pinned staging active:") }
    if ($t -match '\[timers\] decode: (\d+) tokens') { $o.Tokens = [int]$Matches[1] }
    if ($t -match 'dn-split: qkvz ([0-9.]+) \| a\+b [0-9.]+ \| norm [0-9.]+ \| out ([0-9.]+) ms/token') {
        $o.Qkvz = [double]::Parse($Matches[1], $inv); $o.Out = [double]::Parse($Matches[2], $inv)
    }
    if ($t -match 'dn-gpu: dnproj (\d+)/(\d+) \| dnout (\d+)/(\d+) calls') {
        $o.NProj = [int]$Matches[1]; $o.NCalls = [int]$Matches[2]; $o.NOut = [int]$Matches[3]
    }
    if ($t -match 'step\(\) total: ([0-9.]+) ms/token') { $o.Step = [double]::Parse($Matches[1], $inv) }
    foreach ($k in @("Tokens", "Qkvz", "NProj")) {
        if ($null -eq $o[$k]) { throw "$path non contiene le righe [timers] decode / dn-split / dn-gpu. Su Windows il log di nsys non ha l'uscita del motore: passa il log della corsa SENZA profiler con -BaseLog. Altrimenti la corsa va fatta con COLI_TIMERS=1 e un motore con i timer dn-split (#58)." }
    }
    if ($o.Tokens -le 0 -or $o.NCalls -le 0) { throw "${path}: nessun token di decode o nessuna chiamata DeltaNet registrata." }
    [pscustomobject]$o
}

$warn = New-Object System.Collections.Generic.List[string]
$logs = @()
if ($Log)     { $logs += Read-Timers $Log }
if ($BaseLog) { $logs += Read-Timers $BaseLog }
$comparable = @{}
foreach ($tm in $logs) {
    $ok = $true
    if (-not $tm.KeepAlive) { $warn.Add("$($tm.Path): manca '[cuda] keep-alive active:' -- la tappa 0 va misurata con COLI_CUDA_KEEPALIVE=1.") }
    if (-not $tm.Heat)      { $warn.Add("$($tm.Path): manca '[qtier] HEAT_FILE loaded:' -- la tappa 0 va misurata con la tabella heat.") }
    if ($tm.Pinned)         { $warn.Add("$($tm.Path): staging pinned attivo (COLI_CUDA_DENSE_PINNED): le chiamate non sono quelle pageable che il piano descrive.") }
    if ($tm.NProj -ne $tm.NCalls -or $tm.NOut -ne $tm.NCalls) {
        $warn.Add("$($tm.Path): dn-gpu dnproj $($tm.NProj)/$($tm.NCalls), dnout $($tm.NOut)/$($tm.NCalls) -- alcune chiamate sono andate sulla CPU e la media a parete le mescola: confronto saltato.")
        $ok = $false
    }
    $comparable[$tm.Path] = $ok
}
foreach ($tm in $logs) {
    if ($tm.NProj -ne $dnproj.Count -or $tm.NOut -ne $dnout.Count) {
        $warn.Add("$($tm.Path): il motore conta dnproj $($tm.NProj) e dnout $($tm.NOut) sulla GPU, la traccia ha $($dnproj.Count) coppie: traccia e log non vengono da corse con la stessa configurazione, o qualche coppia non e' stata riconosciuta. Confronto saltato.")
        $comparable[$tm.Path] = $false
    }
}
if ($Log -and $BaseLog) {
    $a = $logs[0]; $b = $logs[1]
    if ($a.Tokens -ne $b.Tokens -or $a.NCalls -ne $b.NCalls -or $a.KeepAlive -ne $b.KeepAlive -or $a.Heat -ne $b.Heat -or $a.Pinned -ne $b.Pinned) {
        $warn.Add("$Log e $BaseLog non hanno la stessa configurazione (token, chiamate, keep-alive, heat o pinned): il confronto fra le due pareti non e' un confronto.")
    }
}

# ---- stampa -----------------------------------------------------------------
"traccia: $Csv | $nOps operazioni GPU su $($byStream.Count) stream"
"chiamate dense (kernel quant_matmul*): forma P $nP, forma O $nO, altre $nX (lm_head, attnproj...)"
"coppie dnproj -> dnout di decode: $($dnproj.Count) | forme P senza dnout dopo (prefill riga per riga, o dnout sulla CPU): $nLoneP | attnout: $nAttn"
"chiamate con altre operazioni GPU dentro lo span (caricamenti di expert sullo stesso stream), incluse nelle medie: dnproj {0}, dnout {1}" -f @($dnproj | Where-Object { $_.Foreign -gt 0 }).Count, @($dnout | Where-Object { $_.Foreign -gt 0 }).Count
if (-not $logs.Count) { "(senza -BaseLog ne' -Log non e' controllato niente del motore: keep-alive, tabella heat, chiamate sulla CPU)" }
# Riepilogo per stream: il keep-alive gira su uno stream suo, gli expert su
# un altro, le chiamate dense sullo stream 0.
foreach ($key in @($byStream.Keys | Sort-Object)) {
    $st = $byStream[$key]
    $ks = @($st | Where-Object { $_.Kind -eq "K" })
    $top = @($ks | ForEach-Object {
            $n = [string]$_.Name
            if ($n -match '^(void )?([A-Za-z_][A-Za-z0-9_:]*)') { $n = $Matches[2] }
            $n } | Group-Object | Sort-Object -Property Count -Descending | Select-Object -First 3 | ForEach-Object { "{0} x{1}" -f $_.Name, $_.Count })
    "stream {0}: {1} operazioni, {2} kernel, {3} copie su, {4} copie giu | kernel piu' frequenti: {5}" -f $key, $st.Count, $ks.Count,
        @($st | Where-Object { $_.Kind -eq "H" }).Count, @($st | Where-Object { $_.Kind -eq "D" }).Count, $(if ($top.Count) { $top -join ", " } else { "nessuno" })
}
foreach ($w in $warn) { "ATTENZIONE: $w" }
if ($dnproj.Count -eq 0) {
    throw "nessuna coppia dnproj -> dnout di decode nella traccia: il motore non ha messo dnproj e dnout sulla GPU, oppure le forme non sono quelle di Qwen3.6-35B-A3B."
}
""
"  microsecondi                                  n    media  mediana      p10      p90      min"
$sets = @{ "dnproj" = $dnproj.ToArray(); "dnout" = $dnout.ToArray() }
foreach ($nm in @("dnproj", "dnout")) {
    $cs = $sets[$nm]
    Row "$nm  copia su (H2D)"                ($cs | ForEach-Object { $_.HE - $_.HS })
    Row "$nm  attesa copia su -> kernel"     ($cs | ForEach-Object { $_.KS - $_.HE })
    Row "$nm  kernel"                        ($cs | ForEach-Object { $_.KE - $_.KS })
    Row "$nm  attesa kernel -> copia giu"    ($cs | ForEach-Object { $_.DS - $_.KE })
    Row "$nm  copia giu (D2H)"               ($cs | ForEach-Object { $_.DE - $_.DS })
    Row "$nm  span (inizio su -> fine giu)"  ($cs | ForEach-Object { $_.DE - $_.HS })
    Row "$nm  span, solo senza caricamenti"  ($cs | Where-Object { $_.Foreign -eq 0 } | ForEach-Object { $_.DE - $_.HS })
}
Row "pausa fra fine dnproj e inizio dnout"   ($dnout | ForEach-Object { $_.Lead })

# ---- confronto con i timer del motore --------------------------------------
function Compare-Wall([string]$tag, $tm, [double]$kProj, [double]$sProj, [double]$kOut, [double]$sOut) {
    "{0}: {1} token di decode, step() {2} ms/token, keep-alive {3}, tabella heat {4}" -f $tag, $tm.Tokens,
        $(if ($null -ne $tm.Step) { Fmt $tm.Step } else { "?" }),
        $(if ($tm.KeepAlive) { "attivo" } else { "NON attivo" }), $(if ($tm.Heat) { "caricata" } else { "NON caricata" })
    if (-not $comparable[$tm.Path]) { "  confronto saltato: vedi ATTENZIONE in testa."; return }
    $wProj = $tm.Qkvz * 1000.0 * $tm.Tokens / $tm.NCalls
    $wOut  = $tm.Out  * 1000.0 * $tm.Tokens / $tm.NCalls
    "  per chiamata (us)        parete   span GPU   kernel   fuori dallo span   non-kernel"
    "  dnproj                 {0,8} {1,10} {2,8} {3,18} {4,12}" -f (Fmt $wProj), (Fmt $sProj), (Fmt $kProj), (Fmt ($wProj - $sProj)), (Fmt ($wProj - $kProj))
    "  dnout                  {0,8} {1,10} {2,8} {3,18} {4,12}" -f (Fmt $wOut), (Fmt $sOut), (Fmt $kOut), (Fmt ($wOut - $sOut)), (Fmt ($wOut - $kOut))
    "  span delle sole chiamate senza caricamenti dentro: dnproj {0}, dnout {1} -- con molte chiamate disturbate, 'non-kernel' e' un limite superiore" -f (Fmt $script:sProjClean), (Fmt $script:sOutClean)
    "  ({0} layer DeltaNet per token: non-kernel {1} ms/token in tutto)" -f ($tm.NCalls / $tm.Tokens).ToString("0.##", $inv),
        (Fmt ((($wProj - $kProj) + ($wOut - $kOut)) * $tm.NCalls / $tm.Tokens / 1000.0))
}

if ($logs.Count) {
    $kProj = ($dnproj | ForEach-Object { $_.KE - $_.KS } | Measure-Object -Average).Average
    $sProj = ($dnproj | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average
    $kOut  = ($dnout  | ForEach-Object { $_.KE - $_.KS } | Measure-Object -Average).Average
    $sOut  = ($dnout  | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average
    $cp = @($dnproj | Where-Object { $_.Foreign -eq 0 }); $co = @($dnout | Where-Object { $_.Foreign -eq 0 })
    $script:sProjClean = [double]::NaN; $script:sOutClean = [double]::NaN
    if ($cp.Count) { $script:sProjClean = ($cp | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average }
    if ($co.Count) { $script:sOutClean  = ($co | ForEach-Object { $_.DE - $_.HS } | Measure-Object -Average).Average }
    ""
    "medie della traccia contro la parete dei timer dn-split (medie anche loro):"
    if ($Log)     { Compare-Wall "corsa profilata ($Log)" $logs[0] $kProj $sProj $kOut $sOut }
    if ($BaseLog) { Compare-Wall "corsa NON profilata ($BaseLog), kernel e span dalla traccia" $logs[-1] $kProj $sProj $kOut $sOut }
}
