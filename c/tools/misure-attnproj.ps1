# misure-attnproj.ps1  v4 -- A/B: attnproj sulla CPU vs sulla GPU.
#
# SI LANCIA DA cmd, con UNA riga, e non serve impostare niente a mano:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\misure-attnproj.ps1
#
# v4: fa da solo la scaldata della heat table se manca heat.frozen.bin, e
#     imposta da solo SNAP -- le versioni precedenti costringevano a passare
#     fra cmd e PowerShell per la preparazione, e ogni passaggio e' stato
#     un'occasione per incollare la sintassi sbagliata nella shell sbagliata.
# v3: guardia su COLI_CUDA_PROFILE / PROF / COLI_GRAPH_DIAG.
# v2: prompt come argomento posizionale, numero primario da "Speed: N tok/s".
#
# Zero righe di codice cambiate nel motore: attnproj_fuse / attnproj_bytes /
# trunk_place_attnproj esistono gia' in c/qwen36.c. L'unica differenza fra i
# bracci e' il nome "attnproj" dentro COLI_PLACE.
#
# Il commento a c/qwen36.c:1443 dice che l'offerta e' EXPLICIT ONLY perche'
# "this has not been measured, and #18's whole argument was that a default
# does not move on a prediction". Questo script e' quella misura.
#
# ORDINE ABBA: nei miei script precedenti i bracci si alternavano ma l'ORDINE
# dentro la ripetizione non si invertiva mai, quindi un effetto di posizione
# era indistinguibile dall'effetto del braccio.

param(
    # Lo script vive in c/tools/ ma il binario, il prompt e la heat table
    # stanno in c/: la directory di lavoro e' il PADRE dello script, non lo
    # script. Passare -WorkDir per lanciarlo da altrove.
    [string] $WorkDir = (Split-Path -Parent $PSScriptRoot),
    # qwen36_clang.exe e' linkato contro le runtime di MSYS2 CLANG64 (libomp
    # e compagnia). Senza quella cartella in PATH il loader di Windows uccide
    # il processo con 0xC0000135 (STATUS_DLL_NOT_FOUND) PRIMA di main: niente
    # output, niente log, exit -1073741515. Dipendeva dalla shell da cui si
    # lanciava lo script, ed e' costato due tentativi a vuoto il 21/09.
    [string] $ToolchainBin = "C:\msys64\clang64\bin",
    [string] $Snap   = "C:\modelli\qwen36_i4_gs64",
    [string] $Exe    = ".\qwen36_clang.exe",
    [string] $Prompt = "prompt25.txt",
    # Il primo argomento posizionale e' la cache di esperti per layer, e il
    # tier CUDA si attiva SOLO se copre tutti gli esperti del modello:
    # qwen36_tier.c:499 rifiuta cap != n_experts e torna 0, e da li' in poi il
    # motore gira interamente su CPU. Con cap=16 (il default del banner) la
    # scaldata ha fatto 4.16 tok/s contro i ~37 attesi, e COLI_PLACE non
    # piazzava niente perche' non c'era nessun device: dodici ripetizioni
    # sarebbero passate, con delta zero e nessun modo di accorgersene.
    [int]    $Cap    = 256,
    [int]    $Bits   = 4,
    [int]    $Reps   = 6,
    [int]    $NNew   = 128
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkDir

# ---- guardia ambiente ----------------------------------------------------
# COLI_CUDA_PROFILE accende la strumentazione a eventi CUDA E disarma il grafo
# esperti (graphable_ richiede !ctx->group_timed). Gli script di misura
# precedenti la impostano per le passate profilate e NON la ripuliscono: in
# PowerShell sopravvive per tutta la sessione ed e' ereditata da ogni processo
# figlio, cmd compreso. Il 21/09 e' costata un'ora di caccia a un bug
# inesistente in cuda-test.
foreach ($v in @("COLI_CUDA_PROFILE","PROF","COLI_GRAPH_DIAG")) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($val) { throw "$v=$val e' impostata: la misura non sarebbe confrontabile. Chiudi questa shell, aprine una pulita e rilancia." }
}

if ($ToolchainBin -and (Test-Path $ToolchainBin) -and ($env:PATH -notlike "*$ToolchainBin*")) {
    $env:PATH = "$ToolchainBin;$env:PATH"
    "aggiunto al PATH: $ToolchainBin"
}

if (-not (Test-Path $Exe))    { throw "manca $Exe -- ricostruisci: make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
if (-not (Test-Path $Prompt)) { throw "manca $Prompt" }
if (-not (Test-Path $Snap))   { throw "modello non trovato in $Snap -- passa il percorso con -Snap <dir>" }

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
        throw "exit 0xC0000135 (STATUS_DLL_NOT_FOUND): a $Exe manca una DLL e il processo e' morto prima di main, quindi $Log e' vuoto. Di norma sono le runtime di MSYS2: passa -ToolchainBin <cartella bin del compilatore> se non e' $ToolchainBin."
    }
    if ($code -ne 0) { throw "exit $code -- vedi $Log" }
    Get-Content $Log -Raw
}

# ---- scaldata + congelamento della heat table ----------------------------
if (-not (Test-Path "heat.frozen.bin")) {
    "heat.frozen.bin non c'e': faccio la scaldata (un run, ~1 minuto)..."
    $txt = Invoke-Engine $A "attnproj-warmup.log"
    if ($txt -match 'tier disabled') {
        throw "il tier CUDA si e' disabilitato da solo durante la scaldata (vedi attnproj-warmup.log): senza device questa misura non ha senso. Controlla -Cap."
    }
    if ($txt -notmatch 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') {
        throw "la scaldata non e' arrivata in fondo -- vedi attnproj-warmup.log"
    }
    "scaldata ok: {0} tok/s" -f $Matches[1]
    Copy-Item heat.bin heat.frozen.bin -Force
    "heat congelata in heat.frozen.bin"
    ""
}

function Run-Arm([string]$Place, [string]$Tag, [int]$Rep) {
    Copy-Item heat.frozen.bin heat.bin -Force   # heat identico in ogni run
    $log = "attnproj-$Tag-r$Rep.log"
    $txt = Invoke-Engine $Place $log

    # guardia: B DEVE piazzare attnproj, A non deve
    $named = [regex]::Matches($txt, 'attnproj').Count
    if ($Tag -eq "B" -and $named -eq 0) {
        throw "braccio B rip ${Rep}: 'attnproj' non compare nel log. Il placement NON e' attivo: staresti misurando due volte lo stesso braccio. Vedi $log"
    }
    if ($Tag -eq "A" -and $named -gt 0) {
        throw "braccio A rip ${Rep}: 'attnproj' compare nel log, il riferimento e' contaminato. Vedi $log"
    }

    if ($txt -match 'tier disabled') {
        throw "$Tag rip ${Rep}: il tier CUDA si e' disabilitato da solo (vedi $log). Senza device COLI_PLACE non piazza niente e i due bracci sono identici. Controlla -Cap: deve essere uguale al numero di esperti del modello."
    }
    if ($txt -notmatch 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') { throw "$Tag rip ${Rep}: riga Speed non trovata in $log" }
    $toks = [double]$Matches[1]

    $attn = if ($txt -match '(?m)^\s*\[timers\]\s+attention\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token') { [double]$Matches[1] } else { [double]::NaN }
    $miss = if ($txt -match 'cpu-miss\s+([0-9.]+)') { [double]$Matches[1] } else { [double]::NaN }
    $hit  = if ($txt -match 'Expert cache hit rate:\s*([0-9]+\.[0-9]+)\s*%') { [double]$Matches[1] } else { [double]::NaN }

    [pscustomobject]@{ Rep=$Rep; Arm=$Tag; Ms=(1000.0/$toks); Toks=$toks; Attn=$attn; Miss=$miss; Hit=$hit }
}

$rows = @()
for ($r = 1; $r -le $Reps; $r++) {
    if ($r % 2 -eq 1) { $rows += Run-Arm $A "A" $r; $rows += Run-Arm $B "B" $r }   # dispari: A poi B
    else              { $rows += Run-Arm $B "B" $r; $rows += Run-Arm $A "A" $r }   # pari:    B poi A
    $a = ($rows | ? { $_.Rep -eq $r -and $_.Arm -eq "A" }).Ms
    $b = ($rows | ? { $_.Rep -eq $r -and $_.Arm -eq "B" }).Ms
    "rip {0} ({1})  A {2,6:N2}  B {3,6:N2}  delta {4,6:N2} ms/token" -f `
        $r, $(if ($r % 2 -eq 1) { "A prima" } else { "B prima" }), $a, $b, ($b - $a) | Write-Host
}

$deltas = 1..$Reps | % { $r=$_
    (($rows | ? { $_.Rep -eq $r -and $_.Arm -eq "B" }).Ms) - (($rows | ? { $_.Rep -eq $r -and $_.Arm -eq "A" }).Ms) }

""
"--- delta appaiati (B - A) ---"
($deltas | % { "{0:N2}" -f $_ }) -join "   "
$s = $deltas | Sort-Object
$med = if ($Reps % 2 -eq 1) { $s[[int]($Reps/2)] } else { ($s[$Reps/2-1] + $s[$Reps/2]) / 2 }
"mediana  {0,6:N2} ms/token     negativi {1}/{2}" -f $med, (($deltas | ? { $_ -lt 0 }).Count), $Reps

""
"--- controllo POSIZIONE ---"
"A-prima: {0}" -f ((1..$Reps | ? { $_ % 2 -eq 1 } | % { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"B-prima: {0}" -f ((1..$Reps | ? { $_ % 2 -eq 0 } | % { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"Segno diverso o grandezza molto diversa fra i due gruppi = non e' il braccio, e' la posizione."

""
"--- medie per braccio ---"
$rows | Group-Object Arm | Sort-Object Name | % {
    "{0}:  attention {1,5:N2}   cpu-miss {2,5:N2}   hit {3,5:N1} %   {4,5:N2} tok/s" -f $_.Name,
        (($_.Group | % { $_.Attn } | Measure-Object -Average).Average),
        (($_.Group | % { $_.Miss } | Measure-Object -Average).Average),
        (($_.Group | % { $_.Hit  } | Measure-Object -Average).Average),
        (($_.Group | % { $_.Toks } | Measure-Object -Average).Average)
}
""
"Attesa: 'attention' scende di 2-3 ms in B. Se scende ma il totale no, il guadagno"
"e' stato mangiato da 'cpu-miss': sono i ~107 expert sfrattati dai 189 MB, e"
"quello e' un costo reale. Hit rate che differisce di piu' di ~0.5 punti fra A e B"
"= confronto non lecito, rifare."
