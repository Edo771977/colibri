# misure-attnproj.ps1 -- A/B: attnproj sulla CPU vs sulla GPU.
#
# Si lancia da cmd o da PowerShell, con una riga, senza preparare nulla:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File tools\misure-attnproj.ps1
#
# Zero righe di motore cambiate: attnproj_fuse, attnproj_bytes e
# trunk_place_attnproj esistono gia' in c/qwen36.c. L'unica differenza fra i
# bracci e' il nome "attnproj" dentro COLI_PLACE.
#
# c/qwen36.c:1443 dice che l'offerta di attnproj al placer automatico e'
# EXPLICIT ONLY perche' "this has not been measured, and #18's whole argument
# was that a default does not move on a prediction". Questo script e' quella
# misura. Nessun numero e' rivendicato qui: lo script produce la misura.

param(
    # Lo script vive in c/tools/ ma binario, prompt e heat table stanno in c/.
    [string] $WorkDir = (Split-Path -Parent $PSScriptRoot),
    # qwen36_clang.exe e' linkato contro le runtime di MSYS2 CLANG64. Senza
    # quella cartella in PATH il loader uccide il processo con 0xC0000135
    # PRIMA di main: niente output, log vuoto, exit -1073741515.
    [string] $ToolchainBin = "C:\msys64\clang64\bin",
    [string] $Snap   = "C:\modelli\qwen36_i4_gs64",
    [string] $Exe    = ".\qwen36_clang.exe",
    [string] $Prompt = "prompt25.txt",
    # Primo argomento posizionale del motore: cache di esperti per layer.
    # qwen36_tier.c:499 accende il tier CUDA solo se copre TUTTI gli esperti;
    # con un valore piu' basso il tier si spegne e tutto gira su CPU.
    [int]    $Cap    = 256,
    [int]    $Bits   = 4,
    [int]    $Reps   = 6,
    [int]    $NNew   = 128
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkDir

# ---- guardia ambiente ----------------------------------------------------
# COLI_CUDA_PROFILE accende la strumentazione a eventi CUDA E disarma il grafo
# esperti (graphable_ richiede !ctx->group_timed). Gli script di misura la
# impostano per le passate profilate e non la ripuliscono; in PowerShell
# sopravvive per tutta la sessione ed e' ereditata da ogni processo figlio.
foreach ($v in @("COLI_CUDA_PROFILE","PROF","COLI_GRAPH_DIAG")) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($val) { throw "$v=$val e' impostata: la misura non sarebbe confrontabile. Apri una shell pulita e rilancia." }
}

if ($ToolchainBin -and (Test-Path $ToolchainBin) -and ($env:PATH -notlike "*$ToolchainBin*")) {
    $env:PATH = "$ToolchainBin;$env:PATH"
    "aggiunto al PATH: $ToolchainBin"
}

if (-not (Test-Path $Exe))    { throw "manca $Exe -- ricostruisci: make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
if (-not (Test-Path $Prompt)) { throw "manca $Prompt" }
if (-not (Test-Path $Snap))   { throw "modello non trovato in $Snap -- passa -Snap <dir>" }

$A = "experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0"
$B = "experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0,attnproj=0"

function Invoke-Engine([string]$Place, [string]$Log) {
    $env:SNAP        = $Snap
    $env:COLI_PLACE  = $Place
    $env:COLI_CUDA   = "1"
    $env:COLI_GPUS   = "0"
    $env:HEAT_FILE   = "heat.bin"
    $env:N_NEW       = "$NNew"
    $env:COLI_TIMERS = "1"

    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"        # il motore scrive tutto su stderr
    & $Exe $Cap $Bits $Prompt 2>&1 | Out-File -Encoding utf8 $Log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -eq -1073741515) {
        throw "exit 0xC0000135 (STATUS_DLL_NOT_FOUND): a $Exe manca una DLL e il processo e' morto prima di main, quindi $Log e' vuoto. Di norma sono le runtime di MSYS2: passa -ToolchainBin <cartella bin>."
    }
    if ($code -ne 0) { throw "exit $code -- vedi $Log" }
    Get-Content $Log -Raw
}

# Quanti attnproj il MOTORE dice di aver piazzato. NON cercare la stringa
# "attnproj" nel log: PowerShell trasforma ogni riga di stderr in un
# ErrorRecord e il formattatore ci aggiunge "In ...\misure-attnproj.ps1:NN",
# cioe' il log contiene il nome di questo file e la guardia trova se stessa.
# L'unica prova e' la riga che qwen36.c:1533 stampa solo se placed > 0.
function Get-Placed([string]$Txt) {
    if ($Txt -match '\[place\]\s+(\d+)\s+attnproj \(q\+\+k\+\+v fused\) on GPU') { return [int]$Matches[1] }
    return 0
}

function Assert-TierUp([string]$Txt, [string]$Where, [string]$Log) {
    if ($Txt -match 'tier disabled') {
        throw "${Where}: il tier CUDA si e' disabilitato da solo (vedi $Log). Senza device COLI_PLACE non piazza niente e i due bracci sarebbero identici. Controlla -Cap: deve valere quanto il numero di esperti del modello."
    }
}

# ---- scaldata + congelamento della heat table ----------------------------
if (-not (Test-Path "heat.frozen.bin")) {
    "heat.frozen.bin non c'e': faccio la scaldata..."
    $txt = Invoke-Engine $A "attnproj-warmup.log"
    Assert-TierUp $txt "scaldata" "attnproj-warmup.log"
    if ($txt -notmatch 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') {
        throw "la scaldata non e' arrivata in fondo -- vedi attnproj-warmup.log"
    }
    "scaldata ok: {0} tok/s" -f $Matches[1]
    if (-not (Test-Path "heat.bin")) { throw "il motore non ha scritto heat.bin: il tier non ha fatto shutdown pulito. Vedi attnproj-warmup.log" }
    Copy-Item heat.bin heat.frozen.bin -Force
    "heat congelata in heat.frozen.bin"
    ""
}

function Run-Arm([string]$Place, [string]$Tag, [int]$Rep) {
    Copy-Item heat.frozen.bin heat.bin -Force   # heat identico in ogni run
    $log = "attnproj-$Tag-r$Rep.log"
    $txt = Invoke-Engine $Place $log
    Assert-TierUp $txt "$Tag rip $Rep" $log

    $placed = Get-Placed $txt
    if ($Tag -eq "B" -and $placed -eq 0) {
        throw "braccio B rip ${Rep}: il motore non ha annunciato nessun attnproj piazzato. COLI_PLACE non ha avuto effetto e i due bracci sarebbero identici. Vedi $log"
    }
    if ($Tag -eq "A" -and $placed -gt 0) {
        throw "braccio A rip ${Rep}: il motore ha piazzato $placed attnproj sul braccio di riferimento. Vedi $log"
    }
    if ($txt -match 'attnproj could not be fused or uploaded') {
        throw "$Tag rip ${Rep}: alcuni attnproj non sono stati caricati, ma i loro byte restano addebitati al budget esperti. Il confronto non sarebbe pulito. Vedi $log"
    }

    if ($txt -notmatch 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') { throw "$Tag rip ${Rep}: riga Speed non trovata in $log" }
    $toks = [double]$Matches[1]

    $attn = if ($txt -match '(?m)^\s*\[timers\]\s+attention\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token') { [double]$Matches[1] } else { [double]::NaN }
    $miss = if ($txt -match 'cpu-miss\s+([0-9.]+)') { [double]$Matches[1] } else { [double]::NaN }
    $hit  = if ($txt -match 'Expert cache hit rate:\s*([0-9]+\.[0-9]+)\s*%') { [double]$Matches[1] } else { [double]::NaN }

    [pscustomobject]@{ Rep=$Rep; Arm=$Tag; Ms=(1000.0/$toks); Toks=$toks; Attn=$attn; Miss=$miss; Hit=$hit; Placed=$placed }
}

# ORDINE ABBA: alternare i bracci non basta, perche' l'ORDINE dentro la
# ripetizione resterebbe fisso e un effetto di posizione sarebbe
# indistinguibile dall'effetto del braccio. Dispari A-poi-B, pari B-poi-A.
$rows = @()
for ($r = 1; $r -le $Reps; $r++) {
    if ($r % 2 -eq 1) { $rows += Run-Arm $A "A" $r; $rows += Run-Arm $B "B" $r }
    else              { $rows += Run-Arm $B "B" $r; $rows += Run-Arm $A "A" $r }
    $a = ($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "A" }).Ms
    $b = ($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "B" }).Ms
    "rip {0} ({1})  A {2,6:N2}  B {3,6:N2}  delta {4,6:N2} ms/token" -f `
        $r, $(if ($r % 2 -eq 1) { "A prima" } else { "B prima" }), $a, $b, ($b - $a) | Write-Host
}

$deltas = 1..$Reps | ForEach-Object { $r=$_
    (($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "B" }).Ms) -
    (($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "A" }).Ms) }

""
"--- delta appaiati (B - A) ---"
($deltas | ForEach-Object { "{0:N2}" -f $_ }) -join "   "
$s = $deltas | Sort-Object
$med = if ($Reps % 2 -eq 1) { $s[[int]($Reps/2)] } else { ($s[$Reps/2-1] + $s[$Reps/2]) / 2 }
"mediana  {0,6:N2} ms/token     negativi {1}/{2}" -f $med, (($deltas | Where-Object { $_ -lt 0 }).Count), $Reps

""
"--- controllo POSIZIONE ---"
"A-prima: {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"B-prima: {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 0 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"Segno diverso o grandezza molto diversa fra i due gruppi = hai misurato la posizione, non il braccio."

""
"--- medie per braccio ---"
$rows | Group-Object Arm | Sort-Object Name | ForEach-Object {
    "{0}:  attention {1,5:N2}   cpu-miss {2,5:N2}   hit {3,5:N1} %   {4,5:N2} tok/s   attnproj piazzati {5}" -f $_.Name,
        (($_.Group | ForEach-Object { $_.Attn } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Miss } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Hit  } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Toks } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Placed } | Measure-Object -Maximum).Maximum)
}
""
"Attesa: 'attention' scende di 2-3 ms in B. Se scende ma il totale no, il guadagno"
"e' stato mangiato da 'cpu-miss': sono i ~107 expert sfrattati dai 189 MB, ed e' un"
"costo reale. Hit rate che differisce di piu' di ~0.5 punti fra A e B = confronto"
"non lecito, rifare."
