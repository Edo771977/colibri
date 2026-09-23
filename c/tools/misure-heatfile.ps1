# misure-heatfile.ps1 -- quanto vale una heat table persistente, con abbastanza
# ripetizioni da poterci moltiplicare sopra.
#
# SI LANCIA DA cmd, con UNA riga:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\misure-heatfile.ps1
#
# dalla cartella che contiene qwen36_clang.exe (cioe' c\). Lo script vive in
# c\tools\ ma lavora nella cartella PADRE, dove stanno l'eseguibile, i prompt
# e heat.bin -- passa -WorkDir se il tuo albero e' diverso.
#
# PERCHE' ESISTE. docs/experiments/qwen36-heatfile-2026-09-23-raw.txt misura
# -4.80 ms/token fra tier freddo e tier scaldato da un prompt di ALTRO
# argomento, ma con n=2 per braccio: l'intervallo al 95 % e' [1.74, 7.86].
# Il segno e' sicuro, la taglia no -- e un numero che si moltiplica per milioni
# di token non puo' avere un'incertezza di +-3 ms. Con 6 ripetizioni appaiate
# lo stesso scatter da' circa +-0.8.
#
# I DUE BRACCI:
#   COLD   HEAT_FILE NON impostata. Il motore non carica niente e, cosa che
#          conta per l'alternanza, non SCRIVE niente in uscita: qt_shutdown
#          (c/qwen36_tier.c:1235) scrive solo se la variabile c'e'. Se il
#          braccio freddo scrivesse, contaminerebbe il caldo successivo.
#   CROSS  HEAT_FILE=heat.bin, con dentro una copia della tabella congelata
#          costruita su prompt-caldo.txt -- argomento senza niente in comune
#          col prompt misurato. Ricopiata PRIMA DI OGNI RUN, perche' ogni run
#          la riscrive.
#
# ORDINE ABBA: l'ordine dentro la ripetizione si inverte, altrimenti un
# effetto di posizione nella sessione e' indistinguibile dall'effetto del
# braccio -- che e' esattamente il dubbio che il record non ha saputo
# chiudere, avendo solo tre punti freddi.

param(
    [string] $Snap        = "C:\modelli\qwen36_i4_gs64",
    [string] $Exe         = ".\qwen36_clang.exe",
    [string] $Prompt      = "prompt25.txt",
    [string] $PromptCaldo = "prompt-caldo.txt",
    [int]    $Cap         = 256,
    [int]    $Bits        = 4,
    [int]    $Reps        = 6,
    [int]    $NNew        = 128,
    [string] $WorkDir     = ""
)

$ErrorActionPreference = "Stop"
# Lo script sta in c\tools\, l'eseguibile e i prompt in c\. Una versione
# precedente faceva Set-Location $PSScriptRoot e poi cercava .\qwen36_clang.exe
# nella cartella sbagliata.
if (-not $WorkDir) { $WorkDir = Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path -LiteralPath $WorkDir)) { throw "cartella di lavoro non trovata: $WorkDir" }
Set-Location -LiteralPath $WorkDir
"cartella di lavoro: {0}" -f (Get-Location).Path

# ---- guardie ambiente ----------------------------------------------------
# COLI_CUDA_PROFILE disarma il grafo esperti e gonfia ogni tempo denso. Gli
# altri script di misura la impostano e non la ripuliscono; in PowerShell
# sopravvive alla sessione ed e' ereditata da ogni figlio.
foreach ($v in @("COLI_CUDA_PROFILE","PROF","COLI_GRAPH_DIAG")) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($val) { throw "$v=$val e' impostata: la misura non sarebbe confrontabile. Apri una shell pulita." }
}
if (-not (Test-Path $Exe))         { throw "manca $Exe -- make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
if (-not (Test-Path $Prompt))      { throw "manca $Prompt" }
if (-not (Test-Path $PromptCaldo)) { throw "manca $PromptCaldo -- serve un prompt di argomento DIVERSO da $Prompt" }
if (-not (Test-Path $Snap))        { throw "modello non trovato in $Snap -- passalo con -Snap <dir>" }

# ---- motore --------------------------------------------------------------
function Invoke-Engine([bool]$WithHeat, [string]$PromptFile, [string]$Log) {
    $env:SNAP        = $Snap
    $env:COLI_CUDA   = "1"
    $env:COLI_GPUS   = "0"
    $env:N_NEW       = "$NNew"
    $env:COLI_TIMERS = "1"
    Remove-Item Env:\COLI_PLACE -ErrorAction SilentlyContinue
    if ($WithHeat) { $env:HEAT_FILE = "heat.bin" }
    else           { Remove-Item Env:\HEAT_FILE -ErrorAction SilentlyContinue }

    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"      # il motore scrive tutto su stderr
    & $Exe $Cap $Bits $PromptFile 2>&1 | Out-File -Encoding utf8 $Log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) { throw "exit $code -- vedi $Log" }
    Get-Content $Log -Raw
}

# ---- lettura di un log ---------------------------------------------------
function Read-Run([string]$Text, [string]$Log) {
    if ($Text -notmatch 'CUDA VRAM expert tier active') {
        throw "$Log : il tier CUDA non e' attivo. Senza COLI_CUDA=1, o con cap diverso da n_experts, qt_init torna 0 in silenzio e staresti misurando la CPU."
    }
    if ($Text -notmatch '(?m)^\s*\[timers\]\s+step\(\) total:\s*([0-9.]+)\s*ms/token') {
        throw "$Log : riga step() non trovata. COLI_TIMERS non ha avuto effetto o il run non e' arrivato in fondo."
    }
    $step = [double]$Matches[1]

    $g = @{}
    foreach ($k in @("deltanet","attention","moe_total","lm_head")) {
        if ($Text -match "(?m)^\s*\[timers\]\s+$k\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token") { $g[$k] = [double]$Matches[1] }
        else { $g[$k] = [double]::NaN }
    }
    # I quattro sotto-timer vanno cercati DENTRO la riga qtier. Cercarli nel
    # log intero e' come cercare "take" in un testo inglese: la prima
    # occorrenza vince e non e' detto sia un numero di questo run.
    $qline = if ($Text -match '(?m)^\s*\[timers\]\s+qtier:\s*(.+)$') { $Matches[1] } else { "" }
    if (-not $qline) { throw "$Log : riga [timers] qtier non trovata." }
    foreach ($k in @("issue","cpu-miss","take","shared-ovl")) {
        if ($qline -match "$([regex]::Escape($k))\s+([0-9.]+)") { $g[$k] = [double]$Matches[1] } else { $g[$k] = [double]::NaN }
    }
    $vram  = if ($Text -match 'VRAM hit rate:\s*([0-9.]+)\s*%')             { [double]$Matches[1] } else { [double]::NaN }
    $swaps = if ($Text -match 'LFRU swaps\s+([0-9]+)')                      { [int]$Matches[1] }    else { -1 }
    $miss  = if ($Text -match 'miss\(CPU\)\s+([0-9]+)')                     { [int]$Matches[1] }    else { -1 }
    $upl   = if ($Text -match 'uploads\s+([0-9]+)')                         { [int]$Matches[1] }    else { -1 }
    $res   = if ($Text -match 'resident\s+([0-9]+)/([0-9]+)\s+experts')     { "$($Matches[1])/$($Matches[2])" } else { "?" }
    $toks  = if ($Text -match 'Speed:\s*([0-9.]+)\s*tok/s')                 { [double]$Matches[1] } else { [double]::NaN }
    $place = if ($Text -match '(?m)^(\[place\] auto:.*)$')                  { $Matches[1].Trim() }  else { "" }

    [pscustomobject]@{
        Step=$step; Dn=$g["deltanet"]; Attn=$g["attention"]; Moe=$g["moe_total"]; Head=$g["lm_head"]
        Issue=$g["issue"]; CpuMiss=$g["cpu-miss"]; Take=$g["take"]; ShOvl=$g["shared-ovl"]
        Vram=$vram; Swaps=$swaps; Miss=$miss; Uploads=$upl; Resident=$res; Toks=$toks; Place=$place
    }
}

# ---- fase 0: tabella caldo congelata -------------------------------------
if (-not (Test-Path "heat.caldo.bin")) {
    "heat.caldo.bin non c'e': la costruisco su $PromptCaldo (un run)..."
    Remove-Item heat.bin -ErrorAction SilentlyContinue
    $txt0 = Invoke-Engine $true $PromptCaldo "heatfile-warmup.log"
    $r0 = Read-Run $txt0 "heatfile-warmup.log"
    if (-not (Test-Path "heat.bin")) { throw "la scaldata non ha scritto heat.bin -- vedi heatfile-warmup.log" }
    Copy-Item heat.bin heat.caldo.bin -Force
    "scaldata ok ({0:N2} tok/s, hit {1:N1} %), tabella congelata in heat.caldo.bin" -f $r0.Toks, $r0.Vram
    ""
}

# ---- un braccio ----------------------------------------------------------
function Invoke-Arm([string]$Tag, [int]$Rep) {
    $log = "heatfile-$Tag-r$Rep.log"
    if ($Tag -eq "CROSS") {
        Copy-Item heat.caldo.bin heat.bin -Force     # identica in ogni run caldo
        $txt = Invoke-Engine $true $Prompt $log
        if ($txt -notmatch 'HEAT_FILE loaded') {
            throw "CROSS rip ${Rep}: nessun 'HEAT_FILE loaded' in $log. La tabella non e' stata letta: staresti misurando due volte il braccio freddo."
        }
    } else {
        Remove-Item heat.bin -ErrorAction SilentlyContinue
        $txt = Invoke-Engine $false $Prompt $log
        if ($txt -match 'HEAT_FILE loaded') {
            throw "COLD rip ${Rep}: 'HEAT_FILE loaded' in $log. Il braccio freddo e' contaminato."
        }
        if (Test-Path "heat.bin") {
            throw "COLD rip ${Rep}: il run ha scritto heat.bin. HEAT_FILE non era davvero fuori dall'ambiente e il prossimo CROSS sarebbe contaminato."
        }
    }
    $row = Read-Run $txt $log
    $row | Add-Member -NotePropertyName Rep -NotePropertyValue $Rep
    $row | Add-Member -NotePropertyName Arm -NotePropertyValue $Tag
    $row
}

# ---- il ciclo ------------------------------------------------------------
$rows = @()
for ($rep = 1; $rep -le $Reps; $rep++) {
    if ($rep % 2 -eq 1) { $rows += Invoke-Arm "COLD" $rep; $rows += Invoke-Arm "CROSS" $rep }
    else                { $rows += Invoke-Arm "CROSS" $rep; $rows += Invoke-Arm "COLD" $rep }
    $c = ($rows | Where-Object { $_.Rep -eq $rep -and $_.Arm -eq "COLD"  }).Step
    $x = ($rows | Where-Object { $_.Rep -eq $rep -and $_.Arm -eq "CROSS" }).Step
    "rip {0} ({1})  COLD {2,6:N2}  CROSS {3,6:N2}  delta {4,6:N2} ms/token" -f `
        $rep, $(if ($rep % 2 -eq 1) { "COLD prima" } else { "CROSS prima" }), $c, $x, ($x - $c) | Write-Host
}

# ---- controlli che devono valere su TUTTI i run --------------------------
""
"--- invarianti ---"
$places = @($rows | ForEach-Object { $_.Place } | Sort-Object -Unique)
if ($places.Count -ne 1) {
    "ATTENZIONE: la riga [place] auto: NON e' identica in tutti i run:"
    $places | ForEach-Object { "  $_" }
    "Il piazzamento del trunk e' cambiato fra i bracci, quindi 'attention' NON e' piu' un controllo."
} else {
    "[place] auto: identica in tutti i {0} run" -f @($rows).Count
}
$residents = @($rows | ForEach-Object { $_.Resident } | Sort-Object -Unique)
if ($residents.Count -ne 1) { "ATTENZIONE: residenza diversa fra i run: {0}" -f ($residents -join ", ") }
else { "residenza identica in tutti i run: {0}" -f $residents[0] }

# ---- statistica sui delta appaiati ---------------------------------------
$deltas = 1..$Reps | ForEach-Object {
    $rp = $_
    (($rows | Where-Object { $_.Rep -eq $rp -and $_.Arm -eq "CROSS" }).Step) -
    (($rows | Where-Object { $_.Rep -eq $rp -and $_.Arm -eq "COLD"  }).Step)
}

""
"--- delta appaiati (CROSS - COLD), ms/token ---"
($deltas | ForEach-Object { "{0:N2}" -f $_ }) -join "   "

$mean = ($deltas | Measure-Object -Average).Average
$sd   = if ($Reps -gt 1) { [math]::Sqrt((($deltas | ForEach-Object { [math]::Pow($_ - $mean, 2) }) | Measure-Object -Sum).Sum / ($Reps - 1)) } else { [double]::NaN }
$se   = $sd / [math]::Sqrt($Reps)
# quantili t al 95 % a due code, df 1..15
$tq = @(12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131)
$df = $Reps - 1
$t  = if ($df -ge 1 -and $df -le 15) { $tq[$df-1] } else { 1.96 }
""
"media   {0,6:N2}   sd {1,5:N2}   se {2,5:N2}   df {3}" -f $mean, $sd, $se, $df
"intervallo 95 %:  [{0:N2} ; {1:N2}]   ampiezza +-{2:N2}" -f ($mean-$t*$se), ($mean+$t*$se), ($t*$se)
"negativi {0}/{1}" -f (($deltas | Where-Object { $_ -lt 0 }).Count), $Reps

""
"--- controllo POSIZIONE (ABBA) ---"
"COLD prima:  {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"CROSS prima: {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 0 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"Se i due gruppi hanno segno diverso o grandezza molto diversa, non e' il braccio: e' la posizione."

# ---- medie per braccio, e il controllo che nel record NON tornava --------
""
"--- medie per braccio ---"
foreach ($grp in ($rows | Group-Object Arm | Sort-Object Name)) {
    $q = $grp.Group
    "{0,-5}  step {1,6:N2}  moe {2,6:N2}  attn {3,5:N2}  dn {4,5:N2}  head {5,5:N2}" -f $grp.Name,
        (($q | ForEach-Object { $_.Step }  | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Moe }   | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Attn }  | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Dn }    | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Head }  | Measure-Object -Average).Average)
    "       issue {0,5:N2}  cpu-miss {1,5:N2}  take {2,5:N2}  sh-ovl {3,5:N2}  hit {4,5:N1} %  swaps {5,4:N1}  miss {6}" -f
        (($q | ForEach-Object { $_.Issue }   | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.CpuMiss } | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Take }    | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.ShOvl }   | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Vram }    | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Swaps }   | Measure-Object -Average).Average),
        (($q | ForEach-Object { $_.Miss }    | Sort-Object -Unique) -join "/")
}

""
"--- il controllo che nel record NON tornava: costo per miss ---"
"Il costo per miss dovrebbe essere costante fra i bracci. Nel record da n=2"
"differiva del 28 % (79 contro 101 us). Qui, dividendo cpu-miss per i miss"
"dello STESSO run -- denominatori diversi, quindi e' un indice, non un costo:"
foreach ($grp in ($rows | Group-Object Arm | Sort-Object Name)) {
    $q = $grp.Group | Where-Object { $_.Miss -gt 0 }
    if ($q.Count -eq 0) { "{0,-5}  nessun miss" -f $grp.Name; continue }
    $idx = $q | ForEach-Object { 1000.0 * $_.CpuMiss / $_.Miss }
    "{0,-5}  indice {1,6:N3} us   (min {2:N3} max {3:N3} su {4} run)" -f $grp.Name,
        (($idx | Measure-Object -Average).Average), (($idx | Measure-Object -Minimum).Minimum),
        (($idx | Measure-Object -Maximum).Maximum), $q.Count
}
"Se i due indici restano distanti oltre la loro dispersione, il costo per miss"
"dipende davvero dal batching e nessun numero per-miss si puo' estrapolare."

""
"Log per run: heatfile-COLD-r*.log e heatfile-CROSS-r*.log"
