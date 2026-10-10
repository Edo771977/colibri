# nsys-moe.ps1 -- scompone i gruppi di expert che il motore qwen36 manda
# alla GPU (qt_issue -> qt_take) in una traccia Nsight Systems: quanto di
# ogni gruppo e' lancio, copie, kernel, e come il kernel cresce con il
# numero di expert del gruppo.
#
# SI LANCIA DA cmd, dalla cartella con la traccia, dopo due corse uguali
# (COLI_TIMERS=1), la prima senza profiler, la seconda sotto Nsight:
#     copy /Y heat.caldo.bin heat.bin >nul & qwen36_clang.exe 256 4 prompt25.txt > moe-base.log 2>&1
#     copy /Y heat.caldo.bin heat.bin >nul & nsys profile -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=true -o moe1 qwen36_clang.exe 256 4 prompt25.txt > moe-nsys.log 2>&1
#     nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv --force-export=true --output moe1 moe1.nsys-rep
# con UNA riga:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File C:\colibri-fork\c\tools\nsys-moe.ps1 -Csv moe1_cuda_gpu_trace.csv -ApiCsv moe1_cuda_api_trace.csv -BaseLog moe-base.log
#
# --cuda-graph-trace=node serve: il gruppo di decode e' un CUDA graph
# (coli_cuda_expert_group_issue_x), e senza quell'opzione Nsight lo mostra
# come un blocco solo, senza i kernel e le copie dentro. Su Windows nsys non
# scrive nel suo log l'uscita del programma che profila
# (docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt, sezione 11): i
# timer del motore vengono dalla corsa senza profiler, -BaseLog.
#
# COSA CERCA. Un gruppo, sullo stream della scheda (c/backend_cuda.cu,
# coli_cuda_expert_group_issue_x), e' in quest'ordine: una o due copie
# Host-to-Device (i descrittori, poi l'input x), il kernel grouped_hidden*
# (gate e up), il kernel o i kernel della proiezione down, una copia
# Device-to-Host (le righe di uscita). In qwen36 su quello stream (creato
# cudaStreamNonBlocking) passano solo i gruppi: i caricamenti degli expert
# (coli_cuda_tensor_upload), le chiamate dense e la catena dn_head della
# DeltaNet stanno sullo stream di default. Lo script parte dal kernel
# grouped_hidden*: la copia su subito prima e' l'input x (8 KiB in decode,
# contati quelli che non lo sono), quella prima ancora sono i descrittori
# se e' di 4 KiB al massimo (al massimo 8 expert x 88 byte = 704 byte: il
# limite e' solo una protezione, non serve con il motore di oggi). Dopo il
# kernel hidden, i kernel fino alla prima copia giu. Il numero di expert e'
# la GrdZ del kernel hidden (la griglia e' I x righe x expert), controllato
# con i byte della copia giu (un expert, una riga di 2048 float). Con
# COLI_CUDA_GROUP_ZC=1 il gruppo non ha copie: al posto di quelle su c'e' il
# kernel group_zc_stage, dopo il hidden un solo kernel grouped_down* e
# nessuna copia giu (le righe vanno direttamente nella memoria dell'host);
# lo script riconosce questa forma e la conta a parte.
#
# UNA SCHEDA SOLA. qt_issue manda un gruppo per scheda che ha expert del
# layer in VRAM: con piu' schede i gruppi per layer sono piu' di uno e il
# conteggio qui sotto non vale. Lo script si ferma se trova gruppi su piu'
# di uno stream.
#
# PREFILL E DECODE. Il motore manda un gruppo per layer anche per ogni
# posizione del prompt, con le stesse forme del decode: dalla traccia non si
# distinguono. Lo script legge dal -BaseLog i token del prompt e quelli
# generati, ricava i layer come gruppi / (prompt + generati) -- deve venire
# intero e uguale a -Layers (40 per Qwen3.6-35B-A3B) -- e prende come
# decode gli ultimi generati x layer gruppi. Un layer
# con tutti gli expert sulla CPU non manda il gruppo: allora il conto non
# torna e lo script si ferma. Se nella traccia ci sono kernel dn_head (la
# DeltaNet sulla GPU, COLI_DN_GPU=1, che gira solo in decode), controlla che
# il primo gruppo di decode venga dopo il primo dn_head e l'ultimo di prefill
# prima.
#
# IL LANCIO. Nella traccia del 9 ottobre cudaGraphLaunch ha una mediana di
# 33,7 us, cioe' 1,35 ms/token su 40 layer, contro 1,24 ms/token di tutto
# l'issue della corsa senza profiler: probabilmente la chiamata e' la parte
# maggiore di qt_issue, ma la traccia non lo dimostra, perche' sotto
# profiler la chiamata si allunga (qt_issue contiene anche il lock del tier,
# il tick LFRU al layer 0, due memcpy sull'host). Lo script collega ogni
# gruppo alla sua chiamata in -ApiCsv con la colonna CorrId, provando il
# kernel hidden e poi la prima copia su. Se cosi' se ne collegano meno del
# 90 %, e le chiamate cudaGraphLaunch sono tante quanti i gruppi, le abbina
# in ordine di tempo (la k-esima chiamata al k-esimo gruppo, purche' ogni
# chiamata cominci dopo la fine del gruppo precedente -- qt_take sincronizza
# prima della issue successiva -- e prima dell'inizio del suo; le chiamate
# con Result diverso da 0 non contano) e lo dice. Altrimenti le righe del
# lancio non vengono stampate. I totali per token del lancio si stampano
# solo se tutti i gruppi di decode sono collegati.
#
# COSA STAMPA, in microsecondi, per i gruppi di decode: la durata della
# chiamata di lancio, il ritardo fra la chiamata e la prima operazione sulla
# GPU, le copie su (e dentro: dalla prima copia all'inizio della copia x, la
# copia x, l'attesa fra la copia x e il kernel), dalla fine della chiamata al
# kernel hidden, il kernel hidden, la pausa fra i kernel, il kernel down,
# la copia giu, lo span sulla GPU; poi i kernel per numero di expert, con
# la banda di pesi e scale che ne risulta (-ExpertMatBytes byte per matrice:
# gate e up nel kernel hidden, down nell'altro); poi i gruppi con e senza
# attivita' di altri stream dentro il loro span (per esempio i caricamenti
# di expert; lo spin del keep-alive escluso); infine i totali per token
# accanto ai timer del -BaseLog. Fra i timer, cpu-miss + shared + wait e' la
# finestra fra la fine di issue e il ritorno dell'attesa: per ogni layer vale
# max(lavoro CPU, coda della GPU) piu' il risveglio dall'attesa, quindi la
# coda della GPU dopo issue sta fra wait e quella somma, e vale la somma
# solo se in ogni layer la GPU finisce dopo la CPU.
#
# LIMITI. Il profiler allunga le chiamate API e, con il tracciamento dei
# nodi, forse anche il graph: le durate API valgono come ordine di grandezza,
# accanto all'issue della corsa senza profiler. La banda presuppone le forme
# di Qwen3.6-35B-A3B int4 gs64 (512 KiB di pesi e 64 KiB di scale f32 per
# matrice) e conta pesi e scale letti una volta: non l'input x, che il
# kernel hidden rilegge dalla cache L2 in ogni blocco. Le due corse non sono
# la stessa: con CACHE_ROUTE=1 senza QT_UPLOAD_SYNC=1 la scelta degli expert
# dipende dai tempi, quindi anche gli expert per gruppo possono differire.

param(
    [Parameter(Mandatory = $true)] [string] $Csv,
    [Parameter(Mandatory = $true)] [string] $ApiCsv,
    [Parameter(Mandatory = $true)] [string] $BaseLog,
    # Byte di UNA matrice di un expert (pesi + scale). Default: Qwen3.6-35B-A3B
    # int4 gs64, 2048 x 512 pesi a 4 bit (512 KiB) + 2048 x 512 / 64 scale f32 (64 KiB).
    [long]   $ExpertMatBytes = 589824,
    # Byte della copia giu per expert in decode: una riga di hidden float.
    [long]   $RowBytes = 8192,
    # Layer MoE del modello: i gruppi per posizione devono essere questi.
    [int]    $Layers = 40
)

$ErrorActionPreference = "Stop"
$inv = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture   = $inv
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $inv

foreach ($f in @($Csv, $ApiCsv, $BaseLog)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "manca $f -- genera i CSV con: nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv --force-export=true --output moe1 moe1.nsys-rep" }
}

# ---- colonne, unita', numeri (come nsys-dncalls.ps1) ------------------------
function Find-Col([string[]]$cols, [string]$pattern, [string]$file) {
    $c = @($cols | Where-Object { $_ -like $pattern })
    if ($c.Count -eq 0) { throw ("colonna '{0}' assente in {1}. Colonne trovate: {2}" -f $pattern, $file, ($cols -join ", ")) }
    $c[0]
}
function Get-Unit([string]$col, [hashtable]$table, [string]$file) {
    if ($col -match '\(([^)]+)\)') {
        $u = $Matches[1].Trim()
        foreach ($k in $table.Keys) { if ([string]::Equals($k, $u, [StringComparison]::Ordinal)) { return $table[$k] } }
    }
    throw ("unita' non riconosciuta nella colonna '{0}' di {1}." -f $col, $file)
}
# nsys scrive i decimali con il separatore della lingua di Windows: su un
# Windows italiano "Bytes (MB)" vale "0,008". Si accetta quindi il punto
# oppure, nella sola colonna dei byte e solo se l'unita' non e' il byte, UNA
# virgola decimale. I tempi in ns devono essere interi.
function To-Num([string]$s, [string]$col, [string]$file, [switch]$Int, [switch]$Comma) {
    $v = 0.0
    $t = $s.Trim()
    if ($Int -and $t -notmatch '^-?\d+$') { throw "valore non intero '$s' nella colonna '$col' di ${file}: in ns nsys scrive interi, un punto o una virgola qui sarebbe un separatore delle migliaia." }
    if ($Comma -and $t -match '^-?\d+,\d+$') { $t = $t.Replace(",", ".") }
    if ($t -notmatch '^-?\d+(\.\d+)?([eE][-+]?\d+)?$' -or
        -not [double]::TryParse($t, [Globalization.NumberStyles]::Float, $inv, [ref]$v)) {
        throw "valore non numerico '$s' nella colonna '$col' di $file (atteso un numero con il punto decimale, o con UNA virgola decimale nella colonna dei byte)."
    }
    $v
}
$TimeUnits  = @{ "ns" = 0.001; "us" = 1.0; "ms" = 1000.0 }
$ByteUnits  = @{ "B" = 1.0; "KB" = 1e3; "KiB" = 1024.0; "MB" = 1e6; "MiB" = 1048576.0; "GB" = 1e9 }
function Short-Name([string]$n) {
    if ($n -match '^(void )?([A-Za-z_][A-Za-z0-9_:]*)') { return $Matches[2] }
    $n
}

# ---- traccia GPU ---------------------------------------------------------------
$rows = @(Import-Csv -LiteralPath $Csv)
if ($rows.Count -eq 0) { throw "$Csv e' vuoto: la traccia non contiene attivita' CUDA." }
$cols = @($rows[0].PSObject.Properties.Name)
$cStart = Find-Col $cols "Start*" $Csv
$cDur   = Find-Col $cols "Duration*" $Csv
$cCorr  = Find-Col $cols "Corr*" $Csv
$cBytes = Find-Col $cols "Bytes*" $Csv
$cGrdX  = Find-Col $cols "GrdX*" $Csv
$cGrdY  = Find-Col $cols "GrdY*" $Csv
$cGrdZ  = Find-Col $cols "GrdZ*" $Csv
$cStrm  = Find-Col $cols "Strm*" $Csv
$cName  = Find-Col $cols "Name*" $Csv
$cDev   = @($cols | Where-Object { $_ -like "Device*" })[0]
$cCtx   = @($cols | Where-Object { $_ -like "Ctx*" })[0]
$tUs = Get-Unit $cStart $TimeUnits $Csv
if ((Get-Unit $cDur $TimeUnits $Csv) -ne $tUs) { throw "Start e Duration hanno unita' diverse in $Csv." }
$nsInt   = ($tUs -eq 0.001)
$bFactor = Get-Unit $cBytes $ByteUnits $Csv
$bComma  = ($bFactor -ne 1.0)

$byStream = @{}
$allOps = New-Object System.Collections.Generic.List[object]
foreach ($r in $rows) {
    $name = [string]$r.$cName
    if (([string]$r.$cGrdX).Trim() -ne "")       { $kind = "K" }
    elseif ($name -match 'Host-to-Device|HtoD')   { $kind = "H" }
    elseif ($name -match 'Device-to-Host|DtoH')   { $kind = "D" }
    else                                          { $kind = "O" }
    $s = (To-Num $r.$cStart $cStart $Csv -Int:$nsInt) * $tUs
    $d = (To-Num $r.$cDur $cDur $Csv -Int:$nsInt) * $tUs
    $b = 0.0
    if ($kind -eq "H" -or $kind -eq "D") { $b = (To-Num $r.$cBytes $cBytes $Csv -Comma:$bComma) * $bFactor }
    $gy = 0; $gz = 0; $short = $null
    if ($kind -eq "K") {
        $gy = [int](To-Num $r.$cGrdY $cGrdY $Csv -Int)
        $gz = [int](To-Num $r.$cGrdZ $cGrdZ $Csv -Int)
        $short = Short-Name $name
    }
    $corr = ([string]$r.$cCorr).Trim()
    $dev = ""
    if ($cDev) { $dev = [string]$r.$cDev }
    if ($cCtx) { $dev = [string]$r.$cCtx + "/" + $dev }
    $key = $dev + "/" + [string]$r.$cStrm
    $op = [pscustomobject]@{ S = $s; E = $s + $d; B = $b; Kind = $kind; Name = $short; GY = $gy; GZ = $gz; Corr = $corr; Key = $key; Dev = $dev }
    if (-not $byStream.ContainsKey($key)) { $byStream[$key] = New-Object System.Collections.Generic.List[object] }
    $byStream[$key].Add($op)
    $allOps.Add($op)
}

# Il CSV ha i byte arrotondati (0.008 MB per 8192 B): tolleranza di 600 B o 5 %.
function Near([double]$got, [double]$want) { [Math]::Abs($got - $want) -le [Math]::Max(600.0, 0.05 * $want) }

# ---- gruppi ----------------------------------------------------------------------
$groups = New-Object System.Collections.Generic.List[object]
$nNoUp = 0; $nNoDown = 0; $nNoD2H = 0; $nXOdd = 0; $nZc = 0
foreach ($key in @($byStream.Keys)) {
    $ops = @($byStream[$key] | Sort-Object S)
    for ($i = 0; $i -lt $ops.Count; $i++) {
        $h = $ops[$i]
        if ($h.Kind -ne "K" -or $h.Name -notlike "grouped_hidden*") { continue }
        # copie su immediatamente prima: l'input x subito prima del kernel, e
        # prima ancora i descrittori (8 expert al massimo, 704 byte). Una copia
        # piu' grande di 4 KiB li' davanti non sarebbe dei descrittori e resta
        # fuori (con il motore di oggi non succede: vedi COSA CERCA).
        # COLI_CUDA_GROUP_ZC: al posto delle due copie su, il kernel
        # group_zc_stage (legge x e i descrittori dalla memoria pinned), e
        # nessuna copia giu (il kernel down scrive le righe nella memoria
        # dell'host). Le sue righe "copia x" sono il kernel di staging, e la
        # fase "fine kernel -> fine copia giu" vale 0.
        $zc = ($i -ge 1 -and $ops[$i - 1].Kind -eq "K" -and $ops[$i - 1].Name -like "group_zc_stage*")
        if ($zc) {
            $ups = @($ops[$i - 1])
        } else {
            if ($i -lt 1 -or $ops[$i - 1].Kind -ne "H") { $nNoUp++; continue }
            $ups = @($ops[$i - 1])
            if (-not (Near $ops[$i - 1].B $RowBytes)) { $nXOdd++ }
            if ($i -ge 2 -and $ops[$i - 2].Kind -eq "H" -and $ops[$i - 2].B -le 4096) { $ups = @($ops[$i - 2]) + $ups }
        }
        # kernel fino alla prima copia giu (o, a zero-copy, fino al gruppo dopo)
        $downs = @(); $k = $i + 1
        # Senza la copia giu a chiudere il gruppo, a zero-copy il gruppo finisce
        # al primo kernel che non e' un grouped_down* (la variante esiste solo
        # nei rami del graph, dove dopo il hidden c'e' un solo kernel down).
        while ($k -lt $ops.Count -and $ops[$k].Kind -eq "K" -and $ops[$k].Name -notlike "grouped_hidden*" -and
               $ops[$k].Name -notlike "group_zc_stage*" -and (-not $zc -or $ops[$k].Name -like "grouped_down*")) { $downs += $ops[$k]; $k++ }
        if ($downs.Count -eq 0) { $nNoDown++; continue }
        if ($zc) {
            $nZc++
            $db = [double]::NaN
            $dFirst = [pscustomobject]@{ S = $downs[-1].E; E = $downs[-1].E }; $dLast = $dFirst
        } else {
            if ($k -ge $ops.Count -or $ops[$k].Kind -ne "D") { $nNoD2H++; continue }
            $dFirst = $ops[$k]; $dLast = $ops[$k]; $db = $ops[$k].B
            while ($k + 1 -lt $ops.Count -and $ops[$k + 1].Kind -eq "D") { $k++; $dLast = $ops[$k]; $db += $ops[$k].B }
        }
        $groups.Add([pscustomobject]@{
            Key = $key; Dev = $h.Dev; FS = $ups[0].S; XS = $ups[-1].S; XE = $ups[-1].E; HS = $h.S; HE = $h.E; KS = $downs[0].S; KE = $downs[-1].E
            DS = $dFirst.S; DE = $dLast.E; Count = $h.GZ; Rows = $h.GY; DBytes = $db; NUp = $ups.Count
            CorrH = $h.Corr; CorrUp = $ups[0].Corr; DownName = (@($downs | ForEach-Object { $_.Name }) -join '+'); NDown = $downs.Count
            ApiS = [double]::NaN; ApiE = [double]::NaN; Foreign = $false; Zc = $zc
        })
    }
}
if ($groups.Count -eq 0) {
    throw "nessun gruppo di expert nella traccia (kernel grouped_hidden* con copie su prima e una copia giu dopo): il tier non ha mandato gruppi alla GPU, oppure la traccia e' stata presa senza --cuda-graph-trace=node e i gruppi sono blocchi di graph senza kernel."
}
$groups = [System.Collections.Generic.List[object]]@($groups | Sort-Object FS)
$groupStreams = @($groups | ForEach-Object { $_.Key } | Sort-Object -Unique)
if ($groupStreams.Count -gt 1) {
    $groupDevs = @($groups | ForEach-Object { $_.Dev } | Sort-Object -Unique)
    if ($groupDevs.Count -gt 1) {
        throw ("gruppi di expert su {0} schede ({1}): con piu' schede qt_issue manda un gruppo per scheda e per layer, e il conto prefill/decode di questo script non vale. Lo script e' per una scheda sola (COLI_GPUS=0)." -f $groupDevs.Count, ($groupDevs -join ", "))
    }
    throw ("gruppi di expert su {0} stream della stessa scheda ({1}): il motore li manda tutti su uno stream solo, e lo script cerca le copie e i kernel di un gruppo sul suo stream. Una traccia cosi' non e' quella che lo script sa leggere." -f $groupStreams.Count, ($groupStreams -join ", "))
}

# ---- log senza profiler -----------------------------------------------------------
$t = Get-Content -LiteralPath $BaseLog -Raw
if ($null -eq $t) { $t = "" }
$bl = [ordered]@{ Dec = $null; Pre = $null; Step = $null; Issue = $null; Miss = $null; Take = $null; Shared = $null; Wait = $null; Accum = $null; Moe = $null
                  KeepAlive = $t.Contains("[cuda] keep-alive active:"); Heat = $t.Contains("[qtier] HEAT_FILE loaded:") }
if ($t -match '\[timers\] decode: (\d+) tokens')  { $bl.Dec = [int]$Matches[1] }
if ($t -match '\[timers\] prefill: (\d+) tokens') { $bl.Pre = [int]$Matches[1] }
if ($t -match 'step\(\) total: ([0-9.]+) ms/token') { $bl.Step = [double]::Parse($Matches[1], $inv) }
if ($t -match 'moe_total\s+[0-9.]+ ms\s+([0-9.]+) ms/token') { $bl.Moe = [double]::Parse($Matches[1], $inv) }
if ($t -match 'qtier: issue ([0-9.]+) \| cpu-miss ([0-9.]+) \| take ([0-9.]+)(?: \| shared-ovl ([0-9.]+))? ms/token') {
    $bl.Issue = [double]::Parse($Matches[1], $inv); $bl.Miss = [double]::Parse($Matches[2], $inv); $bl.Take = [double]::Parse($Matches[3], $inv)
    if ($Matches[4]) { $bl.Shared = [double]::Parse($Matches[4], $inv) }
}
if ($t -match 'take split: wait ([0-9.]+) \| accum ([0-9.]+) ms/token') { $bl.Wait = [double]::Parse($Matches[1], $inv); $bl.Accum = [double]::Parse($Matches[2], $inv) }
if ($null -eq $bl.Dec -or $null -eq $bl.Pre -or $bl.Dec -le 0) {
    throw "$BaseLog non contiene le righe '[timers] decode: N tokens' e '[timers] prefill: N tokens': serve la corsa SENZA profiler con COLI_TIMERS=1."
}
$warn = New-Object System.Collections.Generic.List[string]
if (-not $bl.KeepAlive) { $warn.Add("${BaseLog}: manca '[cuda] keep-alive active:' -- senza keep-alive la scheda scende di frequenza e i kernel rallentano (qwen36-gpu-clocks-2026-09-27-raw.txt).") }
if (-not $bl.Heat)      { $warn.Add("${BaseLog}: manca '[qtier] HEAT_FILE loaded:' -- la corsa e' partita senza tabella heat.") }

# ---- prefill e decode -----------------------------------------------------------------
$pos = $bl.Pre + $bl.Dec
if ($groups.Count % $pos -ne 0) {
    throw ("la traccia ha {0} gruppi, che non sono (prompt {1} + generati {2}) x un numero intero di layer. O la corsa profilata non e' uguale a {3} (altro prompt, altro N_NEW), o in qualche posizione un layer non ha mandato il gruppo (tutti gli expert sulla CPU), o lo script non ha riconosciuto qualche gruppo (senza copie su {4}, senza kernel down {5}, senza copia giu {6})." -f $groups.Count, $bl.Pre, $bl.Dec, $BaseLog, $nNoUp, $nNoDown, $nNoD2H)
}
$L = [int]($groups.Count / $pos)
if ($L -ne $Layers) {
    throw ("la traccia ha {0} gruppi = (prompt {1} + generati {2}) x {3}, non x {4} layer (-Layers): la corsa profilata non e' uguale a {5}, oppure il modello non ha {4} layer MoE." -f $groups.Count, $bl.Pre, $bl.Dec, $L, $Layers, $BaseLog)
}
$nPre = [int]($bl.Pre * $L)
$pre = @($groups | Select-Object -First $nPre)
$dec = @($groups | Select-Object -Skip $nPre)
$dnCheck = "nessun kernel dn_head nella traccia: controllo saltato"
$dnHeads = @($allOps | Where-Object { $_.Kind -eq "K" -and $_.Name -like "dn_head*" } | Sort-Object S)
if ($dnHeads.Count -gt 0) {
    $first = $dnHeads[0].S
    $okDec = ($dec[0].FS -gt $first)
    $okPre = ($pre.Count -eq 0 -or $pre[-1].FS -lt $first)
    if ($okDec -and $okPre) { $dnCheck = "il primo kernel dn_head cade fra l'ultimo gruppo di prefill e il primo di decode: ok" }
    else {
        $dnCheck = "il primo kernel dn_head NON cade fra l'ultimo gruppo di prefill e il primo di decode"
        $warn.Add("la divisione prefill/decode presa dal conteggio non coincide con il primo dn_head: le statistiche di decode potrebbero contenere gruppi di prefill o viceversa. Puo' anche essere un falso allarme, se COLI_DN_GPU=1 non ha messo sulla GPU il layer 0 (la riga '[qwen36] DeltaNet decode on the GPU: N/30 layers' del log).")
    }
}

# ---- lancio: CorrId, poi ordine di tempo ---------------------------------------------
$apiRows = @(Import-Csv -LiteralPath $ApiCsv)
if ($apiRows.Count -eq 0) { throw "$ApiCsv e' vuoto." }
$acols = @($apiRows[0].PSObject.Properties.Name)
$aStart = Find-Col $acols "Start*" $ApiCsv
$aDur   = Find-Col $acols "Duration*" $ApiCsv
$aCorr  = Find-Col $acols "Corr*" $ApiCsv
$aName  = Find-Col $acols "Name*" $ApiCsv
$aRes   = @($acols | Where-Object { $_ -like "Result*" })[0]
$nFailed = 0
$atUs = Get-Unit $aStart $TimeUnits $ApiCsv
if ((Get-Unit $aDur $TimeUnits $ApiCsv) -ne $atUs) { throw "Start e Duration hanno unita' diverse in $ApiCsv." }
if ($atUs -ne $tUs) { throw "$Csv e $ApiCsv hanno unita' di tempo diverse." }
$launch = @{}
$graphLaunches = New-Object System.Collections.Generic.List[object]
foreach ($r in $apiRows) {
    $n = [string]$r.$aName
    if ($n -notlike "cudaGraphLaunch*") { continue }
    # Una chiamata fallita non ha lavoro sulla GPU: fuori, o l'abbinamento in
    # ordine slitterebbe di uno da li' in poi.
    if ($aRes -and ([string]$r.$aRes).Trim() -notin @("", "0")) { $nFailed++; continue }
    $s = (To-Num $r.$aStart $aStart $ApiCsv -Int:$nsInt) * $tUs
    $e = $s + (To-Num $r.$aDur $aDur $ApiCsv -Int:$nsInt) * $tUs
    $o = [pscustomobject]@{ S = $s; E = $e }
    $c = ([string]$r.$aCorr).Trim()
    if ($c) { $launch[$c] = $o }
    $graphLaunches.Add($o)
}
$linked = 0
foreach ($g in $groups) {
    $a = $null
    if ($g.CorrH -and $launch.ContainsKey($g.CorrH)) { $a = $launch[$g.CorrH] }
    elseif ($g.CorrUp -and $launch.ContainsKey($g.CorrUp)) { $a = $launch[$g.CorrUp] }
    if ($a) { $g.ApiS = $a.S; $g.ApiE = $a.E; $linked++ }
}
$linkHow = "per CorrId: $linked gruppi su $($groups.Count)"
if ($linked -lt 0.9 * $groups.Count) {
    foreach ($g in $groups) { $g.ApiS = [double]::NaN; $g.ApiE = [double]::NaN }
    $gl = @($graphLaunches | Sort-Object S)
    $ordOk = ($gl.Count -eq $groups.Count)
    # qt_take sincronizza prima della issue successiva: la chiamata giusta
    # comincia dopo la fine del gruppo precedente e prima dell'inizio del suo.
    if ($ordOk) {
        for ($q = 0; $q -lt $gl.Count; $q++) {
            if ($gl[$q].S -ge $groups[$q].FS -or ($q -gt 0 -and $gl[$q].S -le $groups[$q - 1].DE)) { $ordOk = $false; break }
        }
    }
    if ($ordOk) {
        for ($q = 0; $q -lt $gl.Count; $q++) { $groups[$q].ApiS = $gl[$q].S; $groups[$q].ApiE = $gl[$q].E }
        $linkHow = "per CorrId solo $linked su $($groups.Count); abbinati in ordine di tempo: $($gl.Count) chiamate cudaGraphLaunch per $($groups.Count) gruppi, ognuna fra la fine del gruppo precedente e l'inizio del suo"
    } else {
        $linkHow = "per CorrId solo $linked su $($groups.Count), e $($gl.Count) chiamate cudaGraphLaunch per $($groups.Count) gruppi (o una non cade fra la fine del gruppo precedente e l'inizio del suo): le righe del lancio non vengono stampate"
    }
}

# ---- attivita' di altri stream dentro lo span -------------------------------------------
# Unione degli intervalli delle operazioni di altri stream (lo spin del
# keep-alive escluso), poi per ogni gruppo una ricerca binaria.
$decStreams = @($dec | ForEach-Object { $_.Key } | Sort-Object -Unique)
$other = @{}
foreach ($key in $decStreams) {
    $iv = New-Object System.Collections.Generic.List[object]
    foreach ($k2 in @($byStream.Keys)) {
        if ($k2 -eq $key) { continue }
        foreach ($op in $byStream[$k2]) { if (-not ($op.Kind -eq "K" -and $op.Name -like "keepalive*")) { $iv.Add($op) } }
    }
    $merged = New-Object System.Collections.Generic.List[double[]]
    foreach ($op in @($iv | Sort-Object S)) {
        if ($merged.Count -gt 0 -and $op.S -le $merged[$merged.Count - 1][1]) {
            if ($op.E -gt $merged[$merged.Count - 1][1]) { $merged[$merged.Count - 1][1] = $op.E }
        } else { $merged.Add([double[]]@($op.S, $op.E)) }
    }
    $other[$key] = $merged
}
function Overlaps($list, [double]$s, [double]$e) {
    $lo = 0; $hi = $list.Count - 1; $idx = -1
    while ($lo -le $hi) { $mid = [int](($lo + $hi) / 2); if ($list[$mid][0] -lt $e) { $idx = $mid; $lo = $mid + 1 } else { $hi = $mid - 1 } }
    ($idx -ge 0 -and $list[$idx][1] -gt $s)
}
foreach ($g in $dec) { $g.Foreign = Overlaps $other[$g.Key] $g.FS $g.DE }
$decS = $dec[0].FS; $decE = $dec[-1].DE
$inWin = @($allOps | Where-Object { $_.S -lt $decE -and $_.E -gt $decS -and $decStreams -notcontains $_.Key })
$winUp  = @($inWin | Where-Object { $_.Kind -eq "H" -and $_.B -gt 16384 })
$winSm  = @($inWin | Where-Object { $_.Kind -eq "H" -and $_.B -le 16384 })
$winK   = @($inWin | Where-Object { $_.Kind -eq "K" -and $_.Name -notlike "keepalive*" })
$winKNames = @($winK | Group-Object Name | Sort-Object Count -Descending | Select-Object -First 4 | ForEach-Object { "{0} x{1}" -f $_.Name, $_.Count })

# ---- statistiche ------------------------------------------------------------------------
function Get-Stats([double[]]$v) {
    $a = @($v | Where-Object { -not [double]::IsNaN($_) } | Sort-Object)
    $n = $a.Count
    if ($n -eq 0) { return $null }
    $pick = { param($q) $a[[Math]::Min($n - 1, [Math]::Floor($q * ($n - 1) + 0.5))] }
    [pscustomobject]@{ N = $n; Mean = ($a | Measure-Object -Average).Average; Sum = ($a | Measure-Object -Sum).Sum
                       Med = & $pick 0.5; P10 = & $pick 0.1; P90 = & $pick 0.9; Min = $a[0] }
}
function Fmt([double]$x) { if ([double]::IsNaN($x)) { "-" } else { $x.ToString("0.0", $inv) } }
function Fmt2([double]$x) { if ([double]::IsNaN($x)) { "-" } else { $x.ToString("0.00", $inv) } }
function Row([string]$label, [double[]]$v) {
    $st = Get-Stats $v
    if (-not $st) { return }
    "  {0,-44} {1,6} {2,8} {3,8} {4,8} {5,8} {6,8}" -f $label, $st.N, (Fmt $st.Mean), (Fmt $st.Med), (Fmt $st.P10), (Fmt $st.P90), (Fmt $st.Min)
}

# ---- stampa ------------------------------------------------------------------------------
$nNon1 = @($dec | Where-Object { $_.Rows -ne 1 }).Count
$nBytesOff = @($dec | Where-Object { -not $_.Zc -and -not (Near $_.DBytes ($_.Count * $RowBytes)) }).Count
if ($nXOdd)     { $warn.Add("$nXOdd gruppi hanno una copia subito prima del kernel hidden che non e' l'input di $RowBytes byte del decode: la loro fase 'copie su' puo' contenere altro.") }
$downNames = @($dec | ForEach-Object { $_.DownName } | Sort-Object -Unique)
$nMultiDown = @($dec | Where-Object { $_.NDown -ne 1 }).Count
if ($nMultiDown -or $downNames.Count -gt 1) { $warn.Add("kernel dopo il hidden: $nMultiDown gruppi di decode ne hanno piu' di uno, sequenze $($downNames -join ', '). 'kernel down' e la sua banda li sommano tutti: con il motore int4 gs64 di oggi e' uno solo, grouped_down_g4r.") }
if ($nFailed)   { $warn.Add("$nFailed chiamate cudaGraphLaunch con Result diverso da 0, lasciate fuori dal collegamento.") }
if ($nNon1)     { $warn.Add("$nNon1 gruppi di decode hanno piu' di una riga per expert (GrdY > 1): non e' la forma di decode, i tempi per expert non valgono per loro.") }
if ($nBytesOff) { $warn.Add("$nBytesOff gruppi di decode hanno una copia giu che non e' expert x $RowBytes byte: il numero di expert dalla griglia non torna con i byte. -RowBytes e' giusto per questo modello?") }

"traccia: $Csv | $($allOps.Count) operazioni GPU su $($byStream.Count) stream"
"gruppi di expert: $($groups.Count) = (prompt $($bl.Pre) + generati $($bl.Dec)) x $L layer | prefill $($pre.Count), decode $($dec.Count)"
"  non riconosciuti: senza copie su $nNoUp, senza kernel down $nNoDown, senza copia giu $nNoD2H | a zero-copy (COLI_CUDA_GROUP_ZC, kernel di staging al posto delle copie): $nZc"
"  $dnCheck"
"lancio: $linkHow"
"altri stream durante il decode (lo stream di default porta anche le chiamate dense e la DeltaNet): {0} copie su oltre 16 KiB ({1} MB: caricamenti di expert e, al primo token, lo stato DeltaNet), {2} fino a 16 KiB, {3} kernel senza il keep-alive ({4})" -f $winUp.Count,
    (($winUp | Measure-Object -Property B -Sum).Sum / 1e6).ToString("0.0", $inv), $winSm.Count, $winK.Count, $(if ($winKNames.Count) { $winKNames -join ", " } else { "nessuno" })
"gruppi di decode con attivita' di altri stream nello span (keep-alive escluso): {0} su {1}" -f @($dec | Where-Object { $_.Foreign }).Count, $dec.Count
foreach ($w in $warn) { "ATTENZIONE: $w" }
""
"decode, microsecondi per gruppo                    n    media  mediana      p10      p90      min"
Row "chiamata cudaGraphLaunch (durata)"            ($dec | ForEach-Object { $_.ApiE - $_.ApiS })
Row "inizio chiamata -> prima operazione GPU"      ($dec | ForEach-Object { $_.FS - $_.ApiS })
Row "fine chiamata -> prima operazione GPU"        ($dec | ForEach-Object { $_.FS - $_.ApiE })
Row "copie su (prima copia -> kernel hidden)"      ($dec | ForEach-Object { $_.HS - $_.FS })
Row "  prima copia -> inizio copia x"            ($dec | ForEach-Object { $_.XS - $_.FS })
Row "  copia x o kernel di staging (durata)"     ($dec | ForEach-Object { $_.XE - $_.XS })
Row "  fine copia x -> kernel hidden"            ($dec | ForEach-Object { $_.HS - $_.XE })
Row "fine chiamata -> kernel hidden"               ($dec | ForEach-Object { $_.HS - $_.ApiE })
Row "kernel hidden (gate+up)"                      ($dec | ForEach-Object { $_.HE - $_.HS })
Row "pausa hidden -> down"                         ($dec | ForEach-Object { $_.KS - $_.HE })
Row "kernel down"                                  ($dec | ForEach-Object { $_.KE - $_.KS })
Row "fine kernel -> fine copia giu"                ($dec | ForEach-Object { $_.DE - $_.KE })
Row "span GPU (prima copia -> fine copia giu)"     ($dec | ForEach-Object { $_.DE - $_.FS })
Row "inizio chiamata -> fine copia giu"            ($dec | ForEach-Object { $_.DE - $_.ApiS })
Row "fine chiamata -> fine copia giu"              ($dec | ForEach-Object { $_.DE - $_.ApiE })
""
"kernel per numero di expert (decode), microsecondi; banda = byte di pesi e scale / mediana"
"  expert  gruppi   hidden med   down med   hidden+down med   p10    GB/s hidden   GB/s down"
foreach ($grp in @($dec | Group-Object Count | Sort-Object { [int]$_.Name })) {
    $c = [int]$grp.Name
    $sh = Get-Stats @($grp.Group | ForEach-Object { $_.HE - $_.HS })
    $sd = Get-Stats @($grp.Group | ForEach-Object { $_.KE - $_.KS })
    $sk = Get-Stats @($grp.Group | ForEach-Object { ($_.HE - $_.HS) + ($_.KE - $_.KS) })
    $bwH = 2.0 * $c * $ExpertMatBytes / $sh.Med / 1000.0
    $bwD = 1.0 * $c * $ExpertMatBytes / $sd.Med / 1000.0
    "  {0,6} {1,7} {2,12} {3,10} {4,17} {5,6} {6,13} {7,11}" -f $c, $grp.Count, (Fmt $sh.Med), (Fmt $sd.Med), (Fmt $sk.Med), (Fmt $sk.P10), (Fmt $bwH), (Fmt $bwD)
}
$preK = Get-Stats @($pre | ForEach-Object { ($_.HE - $_.HS) + ($_.KE - $_.KS) })
if ($preK) { "  prefill, tutti i gruppi: {0}, hidden+down mediana {1}, media {2}" -f $preK.N, (Fmt $preK.Med), (Fmt $preK.Mean) }
""
$mode = [int](@($dec | Group-Object Count | Sort-Object Count -Descending)[0].Name)
"gruppi di decode con $mode expert (il caso piu' frequente), con e senza attivita' di altri stream nello span:"
"                                    gruppi   hidden+down med   media   span med   media"
foreach ($fg in @($true, $false)) {
    $set = @($dec | Where-Object { $_.Count -eq $mode -and $_.Foreign -eq $fg })
    if ($set.Count -eq 0) { "  {0,-32} {1,7}" -f $(if ($fg) { "con" } else { "senza" }), 0; continue }
    $k = Get-Stats @($set | ForEach-Object { ($_.HE - $_.HS) + ($_.KE - $_.KS) })
    $sp = Get-Stats @($set | ForEach-Object { $_.DE - $_.FS })
    "  {0,-32} {1,7} {2,17} {3,7} {4,10} {5,7}" -f $(if ($fg) { "con" } else { "senza" }), $set.Count, (Fmt $k.Med), (Fmt $k.Mean), (Fmt $sp.Med), (Fmt $sp.Mean)
}
""
# ---- per token ------------------------------------------------------------------------
$nTok = $bl.Dec
function PerTok([double[]]$v) { $st = Get-Stats $v; if (-not $st) { return [double]::NaN }; $st.Sum / 1000.0 / $nTok }
# Il lancio per token solo se ogni gruppo di decode ha la sua chiamata: una
# somma su una parte dei gruppi divisa per tutti i token sarebbe sottostimata.
$allLinked = (@($dec | Where-Object { [double]::IsNaN($_.ApiS) }).Count -eq 0)
$ptApi   = PerTok ($dec | ForEach-Object { $_.ApiE - $_.ApiS })
$ptKern  = PerTok ($dec | ForEach-Object { ($_.HE - $_.HS) + ($_.KE - $_.KS) })
$ptSpan  = PerTok ($dec | ForEach-Object { $_.DE - $_.FS })
$ptAfter = PerTok ($dec | ForEach-Object { $_.DE - $_.ApiE })
if (-not $allLinked) { $ptApi = [double]::NaN; $ptAfter = [double]::NaN }
"per token di decode (somma dei gruppi / $nTok token), ms/token:"
"  traccia (corsa profilata): lancio {0} | kernel {1} | span GPU {2} | da fine chiamata a fine copia giu {3}" -f (Fmt2 $ptApi), (Fmt2 $ptKern), (Fmt2 $ptSpan), (Fmt2 $ptAfter)
if (-not $allLinked) { "  (lancio e 'da fine chiamata' per token non stampati: {0} gruppi di decode su {1} senza la loro chiamata cudaGraphLaunch)" -f @($dec | Where-Object { [double]::IsNaN($_.ApiS) }).Count, $dec.Count }
"  $BaseLog (senza profiler): step {0} | moe {1} | issue {2} | cpu-miss {3} | shared {4} | take {5} (wait {6}, accum {7})" -f `
    $(if ($null -ne $bl.Step) { Fmt2 $bl.Step } else { "?" }), $(if ($null -ne $bl.Moe) { Fmt2 $bl.Moe } else { "?" }),
    $(if ($null -ne $bl.Issue) { Fmt2 $bl.Issue } else { "?" }), $(if ($null -ne $bl.Miss) { Fmt2 $bl.Miss } else { "?" }),
    $(if ($null -ne $bl.Shared) { Fmt2 $bl.Shared } else { "?" }), $(if ($null -ne $bl.Take) { Fmt2 $bl.Take } else { "?" }),
    $(if ($null -ne $bl.Wait) { Fmt2 $bl.Wait } else { "?" }), $(if ($null -ne $bl.Accum) { Fmt2 $bl.Accum } else { "?" })
if ($null -ne $bl.Miss -and $null -ne $bl.Shared -and $null -ne $bl.Wait) {
    "  senza profiler, dalla fine di issue: lavoro CPU (cpu-miss + shared) {0}, poi attesa della GPU (wait) {1}. Per ogni layer la finestra e' max(lavoro CPU, coda della GPU) piu' il risveglio dall'attesa: la coda della GPU dopo issue sta fra {1} e {2} ms/token, e vale {2} solo se in ogni layer la GPU finisce dopo la CPU. Da confrontare con 'da fine chiamata a fine copia giu' della traccia (profilata)." -f `
        (Fmt2 ($bl.Miss + $bl.Shared)), (Fmt2 $bl.Wait), (Fmt2 ($bl.Miss + $bl.Shared + $bl.Wait))
}
