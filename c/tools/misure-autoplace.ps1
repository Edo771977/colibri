# misure-autoplace.ps1 -- A/B sul PLACER AUTOMATICO: binario pre-PR contro post-PR,
# con COLI_PLACE NON IMPOSTATO.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File tools\misure-autoplace.ps1
#
# PERCHE' ESISTE, dato che misure-attnproj.ps1 ha gia' misurato attnproj.
# Quella misura ha girato con COLI_PLACE ESPLICITO in entrambi i bracci, e un
# COLI_PLACE esplicito spegne il placer automatico del tutto. La PR #39 cambia
# il placer AUTOMATICO. Quindi quel braccio A non e' il default di oggi e quel
# braccio B non e' il default di domani: sotto auto, auto_place ripreza tutto e
# i 189 MB di attnproj competono con gli esperti attraverso auto_displaced_value,
# che nessuna di quelle dodici run ha esercitato.
#
# Qui i due bracci sono due BINARI e l'ambiente e' quello vero:
#     A = qwen36 pre-PR   -> auto non conosce attnproj, ne piazza 0
#     B = qwen36 post-PR  -> auto puo' prendere attnproj, e decide da solo
#
# COSA GUARDARE. Non solo il tempo: sotto auto la domanda e' se lo scambio
# conviene, cioe' se i ~100 esperti sfrattati costano meno di quanto attnproj
# fa risparmiare. Lo script stampa budget, numero di esperti residenti e hit
# rate per braccio proprio per quello. qwen36_tier.c:310 dice che senza heat
# table "the trunk always wins": su questa scheda il budget e' largo e lo
# scambio e' probabilmente gratis, ma e' esattamente cio' che va verificato.
#
# LA DLL E' LA STESSA nei due bracci: la #39 non tocca sorgenti CUDA, quindi
# coli_cuda.dll non va ricostruita. Un confonditore in meno, e voluto.
#
# COME COSTRUIRE I DUE BINARI (dalla cartella c\, DLL gia' presente):
#
#     git checkout origin/main
#     make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native
#     copy /Y qwen36.exe qwen36_pre.exe
#
#     git checkout claude/attnproj-auto-place
#     make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native
#     copy /Y qwen36.exe qwen36_post.exe
#
# Nessun numero e' rivendicato qui: lo script produce la misura.

param(
    [string] $WorkDir = (Split-Path -Parent $PSScriptRoot),
    [string] $ToolchainBin = "C:\msys64\clang64\bin",
    [string] $Snap   = "C:\modelli\qwen36_i4_gs64",
    [string] $ExePre  = ".\qwen36_pre.exe",
    [string] $ExePost = ".\qwen36_post.exe",
    [string] $Prompt = "prompt25.txt",
    [int]    $Cap    = 256,
    [int]    $Bits   = 4,
    [int]    $Reps   = 6,
    [int]    $NNew   = 128
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkDir

# ---- guardia ambiente ----------------------------------------------------
foreach ($v in @("COLI_CUDA_PROFILE","PROF","COLI_GRAPH_DIAG")) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($val) { throw "$v=$val e' impostata: disarma il grafo esperti o gonfia i timer. Apri una shell pulita e rilancia." }
}

# QUESTA e' la guardia che distingue questa misura da quella precedente.
# Con COLI_PLACE impostata il placer automatico non gira affatto e lo script
# misurerebbe di nuovo il placement esplicito, cioe' niente di nuovo.
$leftover = [Environment]::GetEnvironmentVariable("COLI_PLACE")
if ($leftover) {
    throw "COLI_PLACE=$leftover e' impostata. Questa misura DEVE girare senza, altrimenti il placer automatico -- l'unica cosa che la PR #39 cambia -- resta spento. Rimuovila: Remove-Item Env:COLI_PLACE"
}
Remove-Item Env:COLI_PLACE -ErrorAction SilentlyContinue

if ($ToolchainBin -and (Test-Path $ToolchainBin) -and ($env:PATH -notlike "*$ToolchainBin*")) {
    $env:PATH = "$ToolchainBin;$env:PATH"
    "aggiunto al PATH: $ToolchainBin"
}

foreach ($candidate in @($ExePre, $ExePost)) {
    if (-not (Test-Path $candidate)) { throw "manca $candidate -- vedi le istruzioni di build in testa a questo file" }
}
if (-not (Test-Path $Prompt)) { throw "manca $Prompt" }
if (-not (Test-Path $Snap))   { throw "modello non trovato in $Snap -- passa -Snap <dir>" }

# Due binari identici = stai misurando il rumore e lo vedresti come "nessun
# effetto". Succede se un make e' fallito e la copy ha ricopiato il vecchio.
$hashPre  = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExePre).Hash
$hashPost = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExePost).Hash
if ($hashPre -eq $hashPost) {
    throw "$ExePre e $ExePost sono lo STESSO binario (sha256 $hashPre). Ricostruiscili: uno da origin/main, uno dal branch della PR."
}
"pre  {0}  {1}" -f $hashPre.Substring(0,16), $ExePre
"post {0}  {1}" -f $hashPost.Substring(0,16), $ExePost
""

function Invoke-Engine([string]$Exe, [string]$LogPath) {
    $env:SNAP        = $Snap
    $env:COLI_CUDA   = "1"
    $env:COLI_GPUS   = "0"
    $env:HEAT_FILE   = "heat.bin"
    $env:N_NEW       = "$NNew"
    $env:COLI_TIMERS = "1"

    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"        # il motore scrive tutto su stderr
    & $Exe $Cap $Bits $Prompt 2>&1 | Out-File -Encoding utf8 $LogPath
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -eq -1073741515) {
        throw "exit 0xC0000135 (STATUS_DLL_NOT_FOUND): a $Exe manca una DLL e il processo e' morto prima di main, quindi $LogPath e' vuoto. Di norma sono le runtime di MSYS2: passa -ToolchainBin <cartella bin>."
    }
    if ($code -ne 0) { throw "exit $code -- vedi $LogPath" }
    Get-Content $LogPath -Raw
}

# Non cercare la stringa "attnproj" nel log: PowerShell trasforma ogni riga di
# stderr in un ErrorRecord e il formattatore ci aggiunge "In ...\script.ps1:NN",
# cioe' il log conterrebbe il nome di questo file e la guardia troverebbe se
# stessa. L'unica prova e' la riga che qwen36.c:1539 stampa solo se placed > 0.
function Get-Placed([string]$Text) {
    if ($Text -match '\[place\]\s+(\d+)\s+attnproj \(q\+\+k\+\+v fused\) on GPU') { return [int]$Matches[1] }
    return 0
}

function Assert-TierUp([string]$Text, [string]$Where, [string]$LogPath) {
    if ($Text -match 'tier disabled') {
        throw "${Where}: il tier CUDA si e' disabilitato da solo (vedi $LogPath). Con il tier spento il placer non piazza niente, i due bracci sono identici e il delta e' uno zero pulito. Controlla -Cap: deve valere quanto il numero di esperti del modello."
    }
}

# ---- scaldata + congelamento della heat table ----------------------------
# La heat table decide quali esperti sono residenti. Se cambia fra i bracci,
# il confronto misura la heat, non il placer.
if (-not (Test-Path "heat.frozen.bin")) {
    "heat.frozen.bin non c'e': faccio la scaldata con il binario pre-PR..."
    $warm = Invoke-Engine $ExePre "autoplace-warmup.log"
    Assert-TierUp $warm "scaldata" "autoplace-warmup.log"
    if ($warm -notmatch 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') {
        throw "la scaldata non e' arrivata in fondo -- vedi autoplace-warmup.log"
    }
    "scaldata ok: {0} tok/s" -f $Matches[1]
    if (-not (Test-Path "heat.bin")) { throw "il motore non ha scritto heat.bin: il tier non ha fatto shutdown pulito. Vedi autoplace-warmup.log" }
    Copy-Item heat.bin heat.frozen.bin -Force
    "heat congelata in heat.frozen.bin"
    ""
}

function Invoke-Arm([string]$Exe, [string]$Tag, [int]$Rep) {
    Copy-Item heat.frozen.bin heat.bin -Force   # heat identico in ogni run
    $log = "autoplace-$Tag-r$Rep.log"
    $txt = Invoke-Engine $Exe $log
    Assert-TierUp $txt "$Tag rip $Rep" $log

    $placed = Get-Placed $txt
    if ($Tag -eq "PRE" -and $placed -gt 0) {
        throw "braccio PRE rip ${Rep}: il binario pre-PR ha piazzato $placed attnproj. Non puo': senza la patch auto non riceve l'offerta. Hai copiato il binario sbagliato. Vedi $log"
    }
    if ($Tag -eq "POST" -and $placed -eq 0) {
        # NON e' un errore dello script: e' un RISULTATO, e va guardato.
        Write-Host "  ATTENZIONE rip ${Rep}: auto NON ha preso attnproj sul binario post-PR." -ForegroundColor Yellow
        Write-Host "  auto_place lo ha rifiutato (non ci sta, o gli esperti sfrattati valgono di piu')." -ForegroundColor Yellow
        Write-Host "  Se succede in tutte le ripetizioni, la PR non cambia niente su questa macchina" -ForegroundColor Yellow
        Write-Host "  e il delta sara' zero: e' una risposta valida, non un guasto. Vedi $log" -ForegroundColor Yellow
    }
    if ($txt -match 'attnproj could not be fused or uploaded') {
        throw "$Tag rip ${Rep}: alcuni attnproj non sono stati caricati, ma i loro byte restano addebitati al budget esperti. Il confronto non sarebbe pulito. Vedi $log"
    }

    # step() total = SOLO decode. La riga "Speed:" e' wall-clock e porta dentro
    # il prefill, che questa patch non tocca (il fuse e' montato solo a S == 1),
    # quindi una percentuale calcolata su quella non e' confrontabile.
    if ($txt -notmatch '(?m)\[timers\]\s+step\(\) total:\s*([0-9.]+)\s*ms/token') {
        throw "$Tag rip ${Rep}: riga 'step() total' non trovata in $log -- COLI_TIMERS non ha avuto effetto?"
    }
    $step = [double]$Matches[1]

    $speed  = if ($txt -match 'Speed:\s*([0-9]+\.[0-9]+)\s*tok/s') { [double]$Matches[1] } else { [double]::NaN }
    $attn   = if ($txt -match '(?m)^\s*\[timers\]\s+attention\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token') { [double]$Matches[1] } else { [double]::NaN }
    $shared = if ($txt -match '(?m)^\s*\[timers\]\s+shared\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token') { [double]$Matches[1] } else { [double]::NaN }
    $router = if ($txt -match '(?m)^\s*\[timers\]\s+router\s+[0-9.]+\s+ms\s+([0-9.]+)\s+ms/token') { [double]$Matches[1] } else { [double]::NaN }
    $miss   = if ($txt -match 'cpu-miss\s+([0-9.]+)') { [double]$Matches[1] } else { [double]::NaN }
    $hit    = if ($txt -match 'Expert cache hit rate:\s*([0-9]+\.[0-9]+)\s*%') { [double]$Matches[1] } else { [double]::NaN }

    # IL PUNTO DI QUESTA MISURA: quanti esperti restano residenti sotto auto.
    $budget  = [double]::NaN
    $experts = 0
    if ($txt -match 'budget\s+([0-9.]+)\s+GB for experts\s+\(~([0-9]+)\s+experts\)') {
        $budget  = [double]$Matches[1]
        $experts = [int]$Matches[2]
    }

    [pscustomobject]@{
        Rep=$Rep; Arm=$Tag; Step=$step; Speed=$speed; Attn=$attn; Shared=$shared
        Router=$router; Miss=$miss; Hit=$hit; Placed=$placed; Budget=$budget; Experts=$experts
    }
}

# ORDINE ABBA: alternare i bracci non basta, perche' l'ORDINE dentro la
# ripetizione resterebbe fisso e un effetto di posizione sarebbe
# indistinguibile dall'effetto del braccio. Dispari PRE-poi-POST, pari il contrario.
$rows = @()
for ($r = 1; $r -le $Reps; $r++) {
    if ($r % 2 -eq 1) { $rows += Invoke-Arm $ExePre "PRE" $r;  $rows += Invoke-Arm $ExePost "POST" $r }
    else              { $rows += Invoke-Arm $ExePost "POST" $r; $rows += Invoke-Arm $ExePre "PRE" $r }
    $stepPre  = ($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "PRE"  }).Step
    $stepPost = ($rows | Where-Object { $_.Rep -eq $r -and $_.Arm -eq "POST" }).Step
    "rip {0} ({1})  PRE {2,6:N2}  POST {3,6:N2}  delta {4,6:N2} ms/token" -f `
        $r, $(if ($r % 2 -eq 1) { "PRE prima" } else { "POST prima" }), $stepPre, $stepPost, ($stepPost - $stepPre) | Write-Host
}

$deltas = 1..$Reps | ForEach-Object { $i=$_
    (($rows | Where-Object { $_.Rep -eq $i -and $_.Arm -eq "POST" }).Step) -
    (($rows | Where-Object { $_.Rep -eq $i -and $_.Arm -eq "PRE"  }).Step) }

""
"--- delta appaiati su step() total, decode (POST - PRE) ---"
($deltas | ForEach-Object { "{0:N2}" -f $_ }) -join "   "
$sorted = $deltas | Sort-Object
$median = if ($Reps % 2 -eq 1) { $sorted[[int]($Reps/2)] } else { ($sorted[$Reps/2-1] + $sorted[$Reps/2]) / 2 }
"mediana  {0,6:N2} ms/token     negativi {1}/{2}     peggiore {3,6:N2}" -f `
    $median, (($deltas | Where-Object { $_ -lt 0 }).Count), $Reps, ($deltas | Measure-Object -Maximum).Maximum

""
"--- controllo POSIZIONE ---"
"PRE-prima:  {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"POST-prima: {0}" -f ((1..$Reps | Where-Object { $_ % 2 -eq 0 } | ForEach-Object { "{0:N2}" -f $deltas[$_-1] }) -join "  ")
"Segno diverso o grandezza molto diversa fra i due gruppi = hai misurato la posizione, non il braccio."

""
"--- cosa ha deciso il placer, e cosa e' costato ---"
$rows | Group-Object Arm | Sort-Object Name | ForEach-Object {
    "{0,-4}  attnproj {1,2}   budget {2,5:N2} GB   esperti {3,5}   hit {4,5:N1} %   cpu-miss {5,5:N2}" -f $_.Name,
        (($_.Group | ForEach-Object { $_.Placed  } | Measure-Object -Maximum).Maximum),
        (($_.Group | ForEach-Object { $_.Budget  } | Measure-Object -Average).Average),
        [int](($_.Group | ForEach-Object { $_.Experts } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Hit     } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Miss    } | Measure-Object -Average).Average)
}

""
"--- righe che NON devono muoversi ---"
$rows | Group-Object Arm | Sort-Object Name | ForEach-Object {
    "{0,-4}  shared {1,5:N2}   router {2,5:N2}   attention {3,5:N2}" -f $_.Name,
        (($_.Group | ForEach-Object { $_.Shared } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Router } | Measure-Object -Average).Average),
        (($_.Group | ForEach-Object { $_.Attn   } | Measure-Object -Average).Average)
}

""
"COME LEGGERE IL RISULTATO"
"  attnproj 0 su POST         -> auto ha rifiutato: la PR non cambia niente qui. Delta atteso 0."
"  esperti quasi uguali       -> lo scambio e' gratis su questa scheda: guarda solo il delta."
"  esperti giu' e hit rate giu' -> i 189 MB hanno sfrattato esperti che servivano: il costo e' reale"
"                                ed e' in cpu-miss. Un delta negativo NON basta, va pesato contro quello."
"  hit rate diverso di piu' di ~0.5 punti fra i bracci -> confronto non lecito, rifare."
"  shared/router che si muovono -> deriva del binario clang gia' documentata, non e' questa patch."
