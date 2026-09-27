# misure-heatfile.ps1 -- il guadagno di una heat table persistente sul tier
# CUDA di qwen36, con delta appaiati fra un braccio freddo e uno caldo.
#
# SI LANCIA DA cmd, con UNA riga:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\misure-heatfile.ps1
#
# dalla cartella che contiene qwen36_clang.exe (cioe' c\). Lo script vive in
# c\tools\ ma lavora nella cartella PADRE, dove stanno l'eseguibile, i prompt
# e heat.bin -- passa -WorkDir se il tuo albero e' diverso.
#
# PERCHE' ESISTE. Con due run per braccio
# (docs/experiments/qwen36-heatfile-2026-09-23-raw.txt) l'intervallo al 95 %
# del guadagno e' [1.74, 7.86] ms/token, calcolato in testa a
# docs/experiments/qwen36-heatfile-6rep-2026-09-23-raw.txt; quel file riporta
# la corsa a sei ripetizioni, fatta con una versione precedente di questo
# script.
#
# I BRACCI:
#   COLD   HEAT_FILE rimossa dall'ambiente. Lo script verifica che il run non
#          carichi la tabella e non scriva heat.bin.
#   CROSS  (default) HEAT_FILE=heat.bin, con dentro una copia della tabella
#          congelata heat.caldo.bin, costruita su -PromptCaldo -- un prompt di
#          argomento diverso da quello misurato -- se non esiste gia'. Se
#          esiste viene riusata solo se heat.caldo.bin.sha256, scritto accanto
#          quando la tabella e' stata costruita, combacia con questa corsa:
#          vedi "tabella congelata" piu' sotto. Misura quanto del guadagno
#          sopravvive a un cambio di argomento.
#   SELF   (-Self) come CROSS, ma la tabella (heat.self.bin) e' costruita sul
#          prompt misurato stesso, e viene ricostruita a ogni corsa.
# La tabella congelata viene ricopiata in heat.bin prima di ogni run caldo,
# cosi' ogni run caldo parte dalla stessa.
#
# ORDINE ABBA: l'ordine dei due bracci si alterna da una ripetizione
# all'altra (COLD prima nelle dispari, il braccio caldo prima nelle pari). Con
# un numero PARI di ripetizioni, un effetto di posizione dentro la coppia che
# resti costante lungo la sessione si annulla esattamente nella media dei
# delta.
#
# TESTO GENERATO: deve essere identico fra i run dello stesso braccio. Fra i
# due bracci lo script dice se coincide e, se no, da dove diverge, senza
# rifiutare la corsa.

param(
    [string] $Snap        = "C:\modelli\qwen36_i4_gs64",
    [string] $Exe         = ".\qwen36_clang.exe",
    # L'eseguibile che la build produce: $Exe deve esserne una copia identica
    # (stesso sha256). -Built "" rinuncia a questo confronto -- non al
    # controllo di staleness sui sorgenti -- e lo stamp dei parametri lo dichiara.
    [string] $Built       = "qwen36.exe",
    [string] $Prompt      = "prompt25.txt",
    # Il prompt su cui si costruisce la tabella del braccio CROSS. Con -Self non
    # si usa: la tabella si costruisce su -Prompt.
    [string] $PromptCaldo = "prompt-caldo.txt",
    [int]    $Cap         = 256,
    [int]    $Bits        = 4,
    [int]    $Reps        = 6,
    [int]    $NNew        = 128,
    [string] $WorkDir     = "",
    # Variabili d'ambiente che l'operatore dichiara innocue per questa misura.
    # Esplicito, per non dover disarmare il controllo intero.
    [string[]] $AllowEnv  = @(),
    # Braccio caldo SELF invece di CROSS: vedi l'intestazione.
    [switch] $Self
)

$ErrorActionPreference = "Stop"
# Il blocco finale va incollato nei record, che usano il PUNTO decimale, ma
# "{0:N2}" -f segue la CurrentCulture e su una macchina italiana stamperebbe la
# virgola; in docs/experiments/qwen36-dense-pinned-2026-09-21-raw.txt:126
# compaiono gia' numeri con la virgola. Si fissa la cultura invariante per
# questo thread: lo script e' a thread singolo. La lettura dei log usa cast
# [double], che sono invarianti di cultura.
[System.Threading.Thread]::CurrentThread.CurrentCulture   = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

# Servono almeno 2 ripetizioni per avere un intervallo, e un numero PARI
# perche' l'alternanza ABBA si bilanci.
if ($Reps -lt 2 -or ($Reps % 2) -ne 0) {
    throw "-Reps $Reps non e' valido: servono almeno 2 ripetizioni (con una sola non c'e' intervallo) e un numero PARI (con un numero dispari l'alternanza ABBA non si bilancia)."
}
if ($NNew -lt 2) {
    throw "-NNew $NNew non e' valido: servono almeno 2 token generati perche' ci sia un passo di decode da misurare."
}
# Lo script sta in c\tools\, l'eseguibile e i prompt in c\.
if (-not $WorkDir) { $WorkDir = Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path -LiteralPath $WorkDir)) { throw "cartella di lavoro non trovata: $WorkDir" }
# Normalizzato PRIMA del Set-Location: piu' sotto $WorkDir viene riusato, e un
# valore relativo (-WorkDir ..) si risolverebbe rispetto alla cwd NUOVA, cioe'
# alla cartella sbagliata.
$WorkDir = (Get-Item -LiteralPath $WorkDir).FullName
Set-Location -LiteralPath $WorkDir
"cartella di lavoro: {0}" -f (Get-Location).Path

# ---- guardie ambiente ----------------------------------------------------
# Rifiuta ogni variabile d'ambiente che combacia con $EnvSuspect, tranne quelle
# che lo script imposta ($EnvOwn), i percorsi del toolkit CUDA ($EnvBenign) e
# quelle che l'operatore dichiara innocue con -AllowEnv. I prefissi coprono le
# famiglie di nomi del motore, di CUDA e del runtime OpenMP. L'elenco dei nomi
# senza prefisso va tenuto allineato a mano ai getenv del motore, e quei nomi
# sono rifiutati tutti allo stesso titolo: la guardia non li classifica.
$EnvOwn = @("SNAP","COLI_CUDA","COLI_GPUS","COLI_TIMERS","COLI_PLACE","HEAT_FILE","N_NEW")
# Con -File, PowerShell passa "-AllowEnv A,B" come UN SOLO elemento di
# [string[]]: va rispezzato a mano.
$AllowEnv = @($AllowEnv | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
# Insieme CHIUSO di percorsi, non una famiglia con coda libera: l'unica
# variazione ammessa e' il suffisso di versione di CUDA_PATH. Ogni altro nome
# CUDA_ resta rifiutato, compreso CUDA_VISIBLE_DEVICES.
$EnvBenign = '^CUDA_(PATH(_V[0-9_]+)?|HOME|BIN_PATH|LIB_PATH|INC_PATH|CACHE_PATH)$'
# QT_ collide con il namespace del framework Qt, quindi al posto del prefisso
# si nominano le due variabili QT_ che compaiono nei getenv del motore.
$EnvSuspect = '^(COLI_|COLIBRI_|QWEN_|QWEN36_|QWEN38_|CUDA_|GOMP_|KMP_|OMP_|HEAT_|Q36_)' +
              '|^(QT_NO_WARMSTART|QT_UPLOAD_SYNC)$' +
              '|^(HOT|NOSTREAM|PROF|WARMUP|SMOOTH|CONF_LIMIT|IDOT|RAM_GB|WIDE|CTX|MODEL|SERVE|PPL|TOK|PILOT|CONSIST|CONSIST_TOL|DUMP|DUMP_LAYERS|DN_DBG|ENC_DEBUG|OPENAI|SNAP|N_NEW)$'
$EnvHits = @()
foreach ($e in Get-ChildItem Env: ) {
    $n = $e.Name
    if ($EnvOwn -contains $n)   { continue }
    if ($AllowEnv -contains $n) { continue }
    if ($n -match $EnvBenign)   { continue }
    if ($n -match $EnvSuspect)  { $EnvHits += "$n=$($e.Value)" }
}
# Un nome in -AllowEnv che non corrisponde a niente e' probabilmente un errore
# di battitura, e un typo non deve essere indistinguibile da una deroga.
$AllowUnused = @($AllowEnv | Where-Object { -not (Test-Path "Env:\$_") })
if ($AllowUnused.Count) {
    throw ("-AllowEnv nomina variabili che non sono impostate: {0}. Se e' un errore di battitura, la deroga che credevi di dare non c'e'." -f ($AllowUnused -join ", "))
}
if ($EnvHits.Count) {
    throw ("queste variabili d'ambiente possono cambiare la misura e sono impostate:`n  {0}`nApri una shell pulita, oppure dichiarale innocue con -AllowEnv <nome> (piu' nomi: separali con la virgola, senza spazi)." -f ($EnvHits -join "`n  "))
}

if (-not (Test-Path -LiteralPath $Exe)) { throw "manca $Exe -- make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
$exeItem = Get-Item -LiteralPath $Exe
if ($exeItem.PSIsContainer) { throw "$Exe e' una cartella, non un file." }
$exeLen = $exeItem.Length

# Un .exe vuoto o troncato -- succede: un '>' incollato in cmd fa da
# redirezione e tronca il file -- da' al lancio un errore illeggibile. Si
# controlla la struttura e non la taglia: i due byte 'MZ' che aprono ogni PE.
# Una soglia in byte non distingue un eseguibile da un file di testo, e la
# taglia legittima dipende dai flag di link.
if ($exeLen -eq 0) { throw "$Exe e' di 0 byte: qualcosa lo ha troncato -- ricostruisci: make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
$fs = [System.IO.File]::OpenRead($exeItem.FullName)
try { $b0 = $fs.ReadByte(); $b1 = $fs.ReadByte() } finally { $fs.Dispose() }
if ($b0 -ne 0x4D -or $b1 -ne 0x5A) {
    throw "$Exe ($exeLen byte) non inizia con la firma PE 'MZ': non e' un eseguibile Windows."
}

# Provenienza: l'eseguibile misurato deve essere la copia di quello che la
# build ha appena prodotto. Chiude il "copy /Y" dimenticato (binario stantio)
# e quello interrotto a meta' (PE tronco con la firma MZ intatta).
# L'eseguibile deve stare in $WorkDir: sorgenti e .build-config si leggono da
# li', e altrimenti lo stamp legherebbe al binario la configurazione di un
# altro albero.
$wdFull = (Get-Item -LiteralPath $WorkDir).FullName.TrimEnd([IO.Path]::DirectorySeparatorChar)
if ($exeItem.DirectoryName.TrimEnd([IO.Path]::DirectorySeparatorChar) -ne $wdFull) {
    throw "$Exe non sta nella cartella di lavoro ($WorkDir): .build-config e i sorgenti verrebbero letti da un albero diverso da quello dell'eseguibile, e la toolchain stampata non sarebbe la sua."
}
$exeHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
if ($Built) {
    if (-not (Test-Path -LiteralPath $Built)) {
        throw "manca ${Built}: non posso verificare che $Exe venga dalla build corrente. Compila, oppure passa -Built `"`" per rinunciare al confronto -- ma allora la provenienza non e' verificata e il record va scritto dicendolo."
    }
    $builtItem = Get-Item -LiteralPath $Built
    if ($builtItem.PSIsContainer) { throw "${Built} e' una cartella, non un file." }
    # $Exe e $Built devono essere due file DISTINTI: se sono lo stesso file
    # l'hash coincide per costruzione e la guardia si autoconferma. Un hardlink
    # o un symlink sfuggono a questo confronto (FullName differisce) e NON sono
    # coperti: in compenso rendono impossibile dimenticare la copia, che e' il
    # caso per cui la guardia esiste.
    if ($builtItem.FullName -eq $exeItem.FullName) {
        throw "$Exe e ${Built} sono lo stesso file: il confronto di provenienza si autoconfermerebbe. Passa -Built con il vero output della build, o -Built `"`" dichiarando che non e' verificato."
    }
    $builtHash = (Get-FileHash -LiteralPath $Built -Algorithm SHA256).Hash
    if ($builtHash -ne $exeHash) {
        # I path stanno fuori dalla stringa di formato: dentro, un '{0}' nel
        # nome del file verrebbe sostituito con l'hash.
        throw ("{0} NON e' la copia di {1} (sha256 {2} contro {3}). Manca il 'copy /Y', oppure la copia e' incompleta: misureresti un binario diverso da quello compilato." -f $Exe, $Built, $exeHash.Substring(0,16), $builtHash.Substring(0,16))
    }
}

# Staleness: nessun prerequisito puo' essere piu' recente dell'eseguibile.
# `copy` preserva il LastWriteTime della sorgente, quindi la data di $Exe e'
# quella della build. Il controllo replica cio' che make gia' fa, per il caso
# in cui make non e' stato invocato.
#
# La lista sono i prerequisiti di qwen36$(EXE) in c/Makefile, con
# backend_loader.o sostituito dai suoi prerequisiti; va tenuta allineata a
# mano. .build-config ne fa parte: se e' piu' recente dell'eseguibile, la
# configurazione registrata non e' quella del binario.
$QwenSrc = @(
    "qwen36.c","qwen36_tier.c","qwen36_tier.h","expert_ffn.h","simd_i8f.h",
    "decode_batch.h","serve_poll.h","cli_args.h","st.h","json.h","compat.h",
    "omp_tune.h","kv_prefix.h","pin_pool.h",
    "edge_adapter_internal.h","edge_adapters.h","edge_runtime.h",
    "segment_adapter_internal.h","segment_adapters.h","segment_runtime.h",
    # $(CUDA_OBJ) sotto CUDA_DLL=1 e' backend_loader.o (c/Makefile:596-598).
    # I suoi prerequisiti (c/Makefile:847) sono backend_loader.c,
    # backend_cuda.h, compat.h e .build-config; gli ultimi due sono gia' qui.
    "backend_loader.c","backend_cuda.h",
    ".build-config"
)
foreach ($src in $QwenSrc) {
    # Un prerequisito ASSENTE non va saltato in silenzio: significa che la
    # cartella di lavoro non e' l'albero da cui viene il binario, e il
    # controllo passerebbe a vuoto -- il vizio che questo script condanna
    # altrove.
    if (-not (Test-Path -LiteralPath $src)) {
        throw "manca $src nella cartella di lavoro ($WorkDir): non e' l'albero dei sorgenti di questo binario, quindi il controllo di staleness passerebbe a vuoto."
    }
    # -Force perche' su sistemi in cui un nome che inizia per punto e' nascosto
    # (non Windows) .build-config non si vede senza. Test-Path non lo accetta e
    # non ne ha bisogno.
    if ((Get-Item -LiteralPath $src -Force).LastWriteTime -gt $exeItem.LastWriteTime) {
        throw "$src e' piu' recente di ${Exe}: ricompila e ricopia, altrimenti misuri codice che non e' quello dell'albero."
    }
}

$ExeStamp = "eseguibile: {0} | {1} byte | {2} | sha256 {3}" -f `
    $Exe, $exeLen, $exeItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"), $exeHash.Substring(0,16)

# c/Makefile:823 compone BUILD_CONFIG -- compilatore, CFLAGS, LDFLAGS, CUDA,
# CUDA_DLL, ARCH e altro -- e :832 lo scrive in .build-config. Il nome
# qwen36_clang.exe e' quello di una copia fatta a mano e non prova la
# toolchain, quindi senza .build-config il record non puo' dichiararla.
$BuildCfg = (Get-Content -LiteralPath ".build-config" -Raw -Force)
if ($null -eq $BuildCfg -or -not $BuildCfg.Trim()) {
    throw ".build-config e' vuoto: la toolchain di questo binario non e' registrata da nessuna parte e il record non potrebbe dichiararla. Ricompila."
}
$BuildCfg = $BuildCfg.Trim()
$CfgStamp = "build-config: $BuildCfg"

# Con CUDA_DLL=1 i kernel stanno in coli_cuda.dll (COLI_BACKEND_DLL,
# c/backend_loader.c:57), caricata a runtime e costruita a parte con nvcc: non
# posso legarla alla build, ma la sua impronta va nel record. Se non e' accanto
# all'eseguibile la sua identita' resta ignota; che il tier GPU sia partito lo
# stabilisce la guardia su "CUDA VRAM expert tier active".
$DllStamp = "coli_cuda.dll: NON TROVATA accanto all'eseguibile -- la sua identita' NON e' registrata"
$DllHash  = "assente"
foreach ($d in @("coli_cuda.dll","coli_hip.dll")) {
    if (Test-Path -LiteralPath $d) {
        $di = Get-Item -LiteralPath $d
        $DllHash = (Get-FileHash -LiteralPath $d -Algorithm SHA256).Hash
        $DllStamp = "{0}: {1} byte | {2} | sha256 {3}" -f `
            $d, $di.Length, $di.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"), $DllHash.Substring(0,16)
        break
    }
}

$ExeStamp
$CfgStamp
$DllStamp

if ($Self) {
    if ($PSBoundParameters.ContainsKey('PromptCaldo')) {
        throw "-Self costruisce la tabella su -Prompt, quindi -PromptCaldo non si usa: togli uno dei due."
    }
    $PromptCaldo = $Prompt
}
$Warm   = if ($Self) { "SELF" } else { "CROSS" }
$Frozen = if ($Self) { "heat.self.bin" } else { "heat.caldo.bin" }

if (-not (Test-Path -LiteralPath $Prompt))      { throw "manca $Prompt" }
if (-not (Test-Path -LiteralPath $PromptCaldo)) { throw "manca $PromptCaldo -- serve un prompt di argomento DIVERSO da $Prompt, oppure -Self" }
if (-not (Test-Path -LiteralPath $Snap))        { throw "modello non trovato in $Snap -- passalo con -Snap <dir>" }

$hPrompt = (Get-FileHash -LiteralPath $Prompt -Algorithm SHA256).Hash
$hCaldo  = (Get-FileHash -LiteralPath $PromptCaldo -Algorithm SHA256).Hash
# CROSS presuppone una tabella costruita su un argomento diverso: con due prompt
# dallo stesso contenuto misurerebbe SELF sotto il nome sbagliato.
if (-not $Self -and $hPrompt -eq $hCaldo) {
    throw "$Prompt e $PromptCaldo hanno lo stesso contenuto: il braccio CROSS misurerebbe uno scaldamento sullo stesso argomento. Per quella misura usa -Self."
}

# Parametri, deroghe e impronte dei prompt vanno nel record insieme ai numeri:
# senza l'impronta, quale prompt sia stato usato resterebbe affidato alla
# parola dell'operatore. Nessun path entra in una stringa di formato.
$ParamStamp = "parametri: cap=$Cap bits=$Bits N_NEW=$NNew rip=$Reps braccio=$Warm | snap=$Snap" +
    ("  | prompt={0} sha256 {1}" -f $Prompt, $hPrompt.Substring(0,16)) +
    $(if ($Self) { "  | tabella costruita sul prompt misurato" } else { "  | caldo={0} sha256 {1}" -f $PromptCaldo, $hCaldo.Substring(0,16) }) +
    ("  | built={0}" -f $(if ($Built) { $Built } else { "(vuoto) -- PROVENIENZA NON VERIFICATA" })) +
    ("  | allow-env={0}" -f $(if ($AllowEnv.Count) { ($AllowEnv -join ",") + " -- DEROGA" } else { "nessuna" }))
$ParamStamp

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

    # Il log va RIMOSSO prima della chiamata: se il lancio fallisse, un log
    # della corsa precedente con lo stesso nome verrebbe letto come risultato
    # di questo run.
    Remove-Item -LiteralPath $Log -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $Log) {
        throw "non riesco a rimuovere $Log prima del run: non posso garantire che il log appartenga a questa corsa."
    }

    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"      # le righe su stderr non devono fermare lo script
    $code = $null
    try {
        # $exeItem.FullName, non $Exe: per un comando nativo PowerShell non
        # cerca nella cartella corrente ma sul PATH, e un omonimo sul PATH
        # verrebbe misurato al posto del file che le guardie hanno validato.
        # Tee-Object tiene gli oggetti: con 2>&1 le righe di stdout arrivano
        # come stringhe e quelle di stderr come ErrorRecord, quindi lo stdout
        # -- dove esce il testo generato -- si separa senza dipendere dall'ordine
        # in cui i due canali si mescolano nel log.
        & $exeItem.FullName $Cap $Bits $PromptFile 2>&1 | Tee-Object -Variable all | Out-File -Encoding utf8 $Log
        $code = $LASTEXITCODE
    } catch {
        # Il fallimento del LANCIO (PE non valido, DLL mancante) diventa qui un
        # messaggio leggibile, con la posizione nel file tagliata via. -csplit e
        # non -split: -split non distingue le maiuscole, e 'At ' combacerebbe con
        # l'"at " di "format error", tagliando la diagnosi invece della posizione.
        $ErrorActionPreference = $old
        $m = ($_.Exception.Message -csplit '\r?\nAt |At ' )[0].Trim()
        throw "$Exe non e' partito: $m"
    }
    $ErrorActionPreference = $old

    if (-not (Test-Path -LiteralPath $Log)) {
        throw "$Log non e' stato creato: il lancio di $Exe non ha prodotto output."
    }
    # $LASTEXITCODE e' $null solo alla PRIMA chiamata nativa della sessione;
    # dalla seconda conserva il valore precedente. Questo controllo copre quindi
    # solo il primo run -- il resto lo coprono il Remove-Item sopra e il catch.
    if ($null -eq $code) { throw "$Exe non ha registrato alcun exit code. Vedi $Log." }
    if ($code -ne 0) { throw "exit $code -- vedi $Log" }
    # Lo stdout, senza la riga [meta] in testa: load_meta (c/qwen36.c) ne
    # scrive una su stdout, prima della generazione, quando lo snapshot non ha
    # qwen36_meta.json. Si toglie al massimo quella, con le maiuscole esatte.
    $lines = @(@($all) | Where-Object { $_ -is [string] })
    $k = 0
    if ($lines.Count -and $lines[0] -cmatch '^\[meta\] ') { $k = 1 }
    $out = (@($lines | Select-Object -Skip $k)) -join "`n"
    [pscustomobject]@{ Text = (Get-Content $Log -Raw); Out = $out }
}

# ---- lettura di un log ---------------------------------------------------
function Read-Run([string]$Text, [string]$Log) {
    if ($Text -notmatch 'CUDA VRAM expert tier active') {
        throw "$Log : il tier CUDA non e' attivo: staresti misurando la CPU. Controlla le righe [qtier] del log."
    }
    # Prima del valore i pattern usano [ \t] e non \s: in .NET \s include \r e
    # \n, e con (?m) un '\s*' puo' scavalcare la fine riga e leggere il valore
    # dalla riga successiva. E niente '$' dopo [^\r\n]+: su un log CRLF la
    # classe si ferma davanti al \r e '$' non combacerebbe.
    if ($Text -notmatch '(?m)^[ \t]*\[timers\][ \t]+step\(\) total:[ \t]*([0-9.]+)[ \t]*ms/token') {
        throw "$Log : riga step() non trovata. COLI_TIMERS non ha avuto effetto o il run non e' arrivato in fondo."
    }
    $step = [double]$Matches[1]

    $g = @{}
    foreach ($k in @("deltanet","attention","moe_total","lm_head")) {
        if ($Text -match "(?m)^[ \t]*\[timers\][ \t]+$k[ \t]+[0-9.]+[ \t]+ms[ \t]+([0-9.]+)[ \t]+ms/token") { $g[$k] = [double]$Matches[1] }
        else { $g[$k] = [double]::NaN }
    }
    # I quattro sotto-timer vanno cercati DENTRO la riga qtier. Cercarli nel
    # log intero e' come cercare "take" in un testo inglese: la prima
    # occorrenza vince e non e' detto sia un numero di questo run.
    $qline = if ($Text -match '(?m)^[ \t]*\[timers\][ \t]+qtier:[ \t]*([^\r\n]+)') { $Matches[1] } else { "" }
    if (-not $qline.Trim()) { throw "$Log : riga [timers] qtier non trovata." }
    foreach ($k in @("issue","cpu-miss","take","shared-ovl")) {
        if ($qline -match "$([regex]::Escape($k))[ \t]+([0-9.]+)") { $g[$k] = [double]$Matches[1] } else { $g[$k] = [double]::NaN }
    }
    # Un timer illeggibile entrerebbe come NaN nelle medie senza alcun errore.
    foreach ($k in @("deltanet","attention","moe_total","lm_head","issue","cpu-miss","take","shared-ovl")) {
        if ([double]::IsNaN($g[$k])) { throw "$Log : il timer '$k' non e' leggibile. Entrerebbe come NaN nelle medie." }
    }
    $vram  = if ($Text -match 'VRAM hit rate:[ \t]*([0-9.]+)[ \t]*%')             { [double]$Matches[1] } else { [double]::NaN }
    # miss(CPU) serve all'indice per miss in coda, LFRU swaps alle medie per
    # braccio: nessuno dei due ripiega su un valore di comodo.
    if ($Text -notmatch 'miss\(CPU\)[ \t]+([0-9]+)') { throw "$Log : riga 'miss(CPU) N' non trovata. L'indice per miss passerebbe a vuoto." }
    $miss  = [int]$Matches[1]
    if ($Text -notmatch 'LFRU swaps[ \t]+([0-9]+)')   { throw "$Log : riga 'LFRU swaps N' non trovata." }
    $swaps = [int]$Matches[1]
    $res   = if ($Text -match 'resident[ \t]+([0-9]+)/([0-9]+)[ \t]+experts')     { "$($Matches[1])/$($Matches[2])" } else { "?" }
    $toks  = if ($Text -match 'Speed:[ \t]*([0-9.]+)[ \t]*tok/s')                 { [double]$Matches[1] } else { [double]::NaN }
    # TUTTE le righe [place] auto:, non solo la prima.
    $place = (([regex]::Matches($Text, '(?m)^\[place\] auto:.*$') |
               ForEach-Object { $_.Value.Trim() }) -join ' ;; ')

    # Banner, [place] e residenza alimentano gli invarianti in coda. Se un regex
    # non trovasse nulla, il valore sarebbe lo stesso per TUTTI i run e
    # l'invariante stamperebbe "identica in tutti i run" sull'ASSENZA del dato:
    # un invariante che passa a vuoto e' peggio di uno che manca, perche' finisce
    # nel record come se avesse verificato qualcosa. Per questo la loro assenza
    # e' un errore.
    # Il banner riporta otto parametri con cui il motore e' partito (cache,
    # bits, ctx, pilot, wide, hot, smooth, conf): pretenderlo identico fra i
    # run legge cosa il motore ha deciso, invece di indovinarlo dall'ambiente.
    # \s*$ e non $: su un log CRLF il \r sta prima della fine riga, e un
    # letterale seguito da '$' non combacerebbe. Il pattern di [place] finisce
    # in .*, che assorbe il \r.
    # Il banner non e' ancorato a inizio riga, cosi' combacia anche se la riga
    # porta un prefisso: Windows PowerShell 5.1 ne antepone uno al primo
    # ErrorRecord del log (non verificato qui).
    if ($Text -notmatch '(?m)(== qwen36 Phase-2 engine \|.*==)\s*$') {
        throw "$Log : il banner '== qwen36 Phase-2 engine |' non c'e'. Il motore non e' arrivato a stamparlo, oppure non e' qwen36."
    }
    $banner = $Matches[1].Trim()

    if (-not $place) { throw "$Log : riga '[place] auto:' non trovata. L'invariante sul piazzamento del trunk passerebbe a vuoto." }
    if ($res -eq "?") { throw "$Log : riga 'resident N/M experts' non trovata. L'invariante sulla residenza passerebbe a vuoto." }

    [pscustomobject]@{
        Step=$step; Dn=$g["deltanet"]; Attn=$g["attention"]; Moe=$g["moe_total"]; Head=$g["lm_head"]
        Issue=$g["issue"]; CpuMiss=$g["cpu-miss"]; Take=$g["take"]; ShOvl=$g["shared-ovl"]
        Vram=$vram; Swaps=$swaps; Miss=$miss; Resident=$res; Toks=$toks; Place=$place; Banner=$banner
    }
}

# ---- fase 0: tabella caldo congelata -------------------------------------
# Invoke-Engine rimuove il log del run che sta per fare, ma una corsa
# interrotta lasciava nella cartella i log della sessione PRECEDENTE accanto a
# quelli nuovi, indistinguibili, mentre la riga finale invita a trattarli come
# un insieme unico: la stessa trappola del log stantio, spostata dal run
# all'archiviazione.
$oldLogs = @(Get-ChildItem -LiteralPath $WorkDir -Filter "heatfile-*.log" -ErrorAction SilentlyContinue)
if ($oldLogs.Count) {
    "rimuovo {0} log della corsa precedente" -f $oldLogs.Count
    $oldLogs | Remove-Item -Force
}

function Get-TextHash([string]$s) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { -join ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($s)) | ForEach-Object { $_.ToString("x2") }) }
    finally { $sha.Dispose() }
}

# ---- tabella congelata ---------------------------------------------------
# E' la variabile indipendente dell'esperimento. Accanto ha un file che
# registra, una riga chiave=valore ciascuna: impronta del prompt, impronta
# dell'eseguibile, impronta di coli_cuda.dll (o "assente"), N_NEW, cap, bits,
# le deroghe -AllowEnv, impronta del PERCORSO dello snapshot e impronta della
# tabella. NON registra il contenuto dello snapshot. Si riusa solo se ogni
# chiave combacia con questa corsa; altrimenti la corsa si ferma e il messaggio
# dice quali chiavi non combaciano.
# Il percorso dello snapshot si registra senza separatori finali e, su
# Windows, in minuscolo: "C:\modelli\x", "C:\modelli\x\" e "C:\Modelli\x"
# danno la stessa impronta.
$SnapPath = (Get-Item -LiteralPath $Snap).FullName.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
if ([IO.Path]::DirectorySeparatorChar -eq [char]'\') { $SnapPath = $SnapPath.ToLowerInvariant() }
$Side = "$Frozen.sha256"
$Want = [ordered]@{
    prompt     = $hCaldo
    eseguibile = $exeHash
    dll        = $DllHash
    n_new      = "$NNew"
    cap        = "$Cap"
    bits       = "$Bits"
    allow_env  = $(if ($AllowEnv.Count) { (@($AllowEnv | Sort-Object) -join ",") } else { "nessuna" })
    snap       = Get-TextHash $SnapPath
}
$HeatStamp = ""
if ($Self -or -not (Test-Path -LiteralPath $Frozen)) {
    "costruisco $Frozen su $PromptCaldo (un run)..."
    Remove-Item heat.bin -Force -ErrorAction SilentlyContinue
    if (Test-Path "heat.bin") { throw "non riesco a rimuovere heat.bin: la scaldata partirebbe da una tabella vecchia invece che da zero." }
    $e0 = Invoke-Engine $true $PromptCaldo "heatfile-warmup.log"
    if ($e0.Text -match 'HEAT_FILE loaded') { throw "la scaldata ha caricato una tabella invece di partire da zero -- vedi heatfile-warmup.log" }
    $r0 = Read-Run $e0.Text "heatfile-warmup.log"
    if (-not (Test-Path "heat.bin")) { throw "la scaldata non ha scritto heat.bin -- vedi heatfile-warmup.log" }
    Copy-Item heat.bin $Frozen -Force
    $lines = @($Want.Keys | ForEach-Object { "{0}={1}" -f $_, $Want[$_] })
    $lines += "tabella={0}" -f (Get-FileHash -LiteralPath $Frozen -Algorithm SHA256).Hash
    Set-Content -LiteralPath $Side -Encoding ascii -Value $lines
    "scaldata ok ({0:N2} tok/s, hit {1:N1} %), tabella congelata in {2}" -f $r0.Toks, $r0.Vram, $Frozen
    ""
    $HeatStamp = "COSTRUITA in questa corsa su $PromptCaldo"
} else {
    if (-not (Test-Path -LiteralPath $Side)) {
        throw "$Frozen esiste ma accanto non c'e' ${Side}: non so in quali condizioni sia stata costruita. Cancellala, e lo script la ricostruira' su $PromptCaldo."
    }
    # Lo scrive lo script: una riga che non ha la forma chiave=valore, o una
    # chiave ripetuta, vuol dire che e' stato toccato a mano.
    $got = @{}
    foreach ($ln in @(Get-Content -LiteralPath $Side)) {
        if (-not $ln.Trim()) { continue }
        if ($ln -notmatch '^\s*([a-z_]+)=(\S+)\s*$') { throw "${Side}: riga non riconosciuta '$ln'. Cancella $Frozen e $Side, e lo script ricostruira' la tabella." }
        if ($got.ContainsKey($Matches[1])) { throw "${Side}: la chiave '$($Matches[1])' compare due volte. Cancella $Frozen e $Side, e lo script ricostruira' la tabella." }
        $got[$Matches[1]] = $Matches[2]
    }
    $check = [ordered]@{}
    foreach ($key in $Want.Keys) { $check[$key] = $Want[$key] }
    $check["tabella"] = (Get-FileHash -LiteralPath $Frozen -Algorithm SHA256).Hash
    $miss = @($check.Keys | Where-Object { -not $got.ContainsKey($_) })
    $diff = @($check.Keys | Where-Object { $got.ContainsKey($_) -and $got[$_] -ne $check[$_] })
    if ($miss.Count -or $diff.Count) {
        throw ("{0} non si puo' riusare: in {1} {2}{3}. Cancellala, e lo script la ricostruira'." -f $Frozen, $Side,
            $(if ($diff.Count) { "non combaciano con questa corsa: " + ($diff -join ", ") } else { "" }),
            $(if ($miss.Count) { $(if ($diff.Count) { "; " } else { "" }) + "mancano: " + ($miss -join ", ") } else { "" }))
    }
    $HeatStamp = "RIUSATA: $Side combacia su prompt, eseguibile, coli_cuda.dll, N_NEW, cap, bits, deroghe, percorso dello snapshot e tabella (non registra il contenuto dello snapshot)"
}
$hc = Get-Item -LiteralPath $Frozen
$HeatStamp = "{0}: {1} byte | {2} | sha256 {3} | {4}" -f `
    $Frozen, $hc.Length, $hc.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"),
    (Get-FileHash -LiteralPath $Frozen -Algorithm SHA256).Hash.Substring(0,16), $HeatStamp
$HeatStamp
""

# ---- un braccio ----------------------------------------------------------
function Invoke-Arm([string]$Tag, [int]$Rep) {
    $log = "heatfile-$Tag-r$Rep.log"
    if ($Tag -eq $Warm) {
        Copy-Item -LiteralPath $Frozen heat.bin -Force     # identica in ogni run caldo
        $e = Invoke-Engine $true $Prompt $log
        if ($e.Text -notmatch 'HEAT_FILE loaded') {
            throw "$Tag rip ${Rep}: nessun 'HEAT_FILE loaded' in $log. La tabella non e' stata letta: staresti misurando due volte il braccio freddo."
        }
    } else {
        Remove-Item heat.bin -Force -ErrorAction SilentlyContinue
        $e = Invoke-Engine $false $Prompt $log
        if ($e.Text -match 'HEAT_FILE loaded') {
            throw "COLD rip ${Rep}: 'HEAT_FILE loaded' in $log. Il braccio freddo e' contaminato."
        }
        if (Test-Path "heat.bin") {
            throw "COLD rip ${Rep}: il run ha scritto heat.bin, quindi HEAT_FILE non era fuori dall'ambiente del braccio freddo."
        }
    }
    $row = Read-Run $e.Text $log
    # Uno stdout vuoto farebbe passare a vuoto il confronto sul testo in coda.
    if (-not $e.Out.Trim()) {
        throw "$Tag rip ${Rep}: nessun testo generato su stdout (tolta la riga [meta]). Il confronto sul testo passerebbe a vuoto."
    }
    $row | Add-Member -NotePropertyName Rep -NotePropertyValue $Rep
    $row | Add-Member -NotePropertyName Arm -NotePropertyValue $Tag
    $row | Add-Member -NotePropertyName Out -NotePropertyValue $e.Out
    $row | Add-Member -NotePropertyName OutHash -NotePropertyValue (Get-TextHash $e.Out)
    $row
}

# ---- il ciclo ------------------------------------------------------------
$rows = @()
for ($rep = 1; $rep -le $Reps; $rep++) {
    if ($rep % 2 -eq 1) { $rows += Invoke-Arm "COLD" $rep; $rows += Invoke-Arm $Warm $rep }
    else                { $rows += Invoke-Arm $Warm $rep; $rows += Invoke-Arm "COLD" $rep }
    $c = ($rows | Where-Object { $_.Rep -eq $rep -and $_.Arm -eq "COLD" }).Step
    $x = ($rows | Where-Object { $_.Rep -eq $rep -and $_.Arm -eq $Warm  }).Step
    "rip {0} ({1})  COLD {2,6:N2}  {5,-5} {3,6:N2}  delta {4,6:N2} ms/token" -f `
        $rep, $(if ($rep % 2 -eq 1) { "COLD prima" } else { "$Warm prima" }), $c, $x, ($x - $c), $Warm | Write-Host
}

# ---- invarianti su tutti i run, poi il testo fra i bracci -----------------
""
"--- invarianti ---"
$ExeStamp
$CfgStamp
$DllStamp
$HeatStamp
$ParamStamp
# Un invariante violato ferma lo script: se non tiene, i numeri non vanno
# prodotti in forma incollabile. I confronti sono ORDINALI: Sort-Object
# -Unique, con la cultura invariante, tratta come uguali due righe che
# differiscono per maiuscole o per un carattere ignorabile.
function Get-Distinct($items) {
    $set = New-Object "System.Collections.Generic.HashSet[string]" -ArgumentList ([StringComparer]::Ordinal)
    foreach ($x in $items) { if ($set.Add([string]$x)) { [string]$x } }
}
$banners = @(Get-Distinct @($rows | ForEach-Object { $_.Banner }))
if ($banners.Count -ne 1) {
    throw ("il banner del motore NON e' identico in tutti i run:`n  {0}" -f ($banners -join "`n  "))
}
"banner del motore identico in tutti i run:"
"  {0}" -f $banners[0]
$places = @(Get-Distinct @($rows | ForEach-Object { $_.Place }))
if ($places.Count -ne 1) {
    throw ("la riga [place] auto: NON e' identica in tutti i run: i bracci non sono confrontabili.`n  {0}" -f ($places -join "`n  "))
}
"[place] auto: identica in tutti i {0} run" -f @($rows).Count
$residents = @(Get-Distinct @($rows | ForEach-Object { $_.Resident }))
if ($residents.Count -ne 1) {
    throw ("residenza diversa fra i run: {0} -- i bracci non hanno lo stesso numero di esperti in VRAM e il delta non e' attribuibile alla tabella heat." -f ($residents -join ", "))
}
"residenza identica in tutti i run: {0}" -f $residents[0]
# Il testo generato. Dentro ogni braccio deve essere identico: se le
# ripetizioni dello stesso braccio non decodificano la stessa sequenza, non
# sono ripetizioni. Fra i due bracci puo' differire, e lo script lo dichiara
# invece di rifiutare la corsa. Impronta e confronto sono sullo stdout come
# PowerShell lo legge (righe unite con LF, senza la riga [meta] in testa), non
# sui byte scritti dal motore: due stdout che differiscono solo nei fine riga,
# o in byte UTF-8 non validi, risultano uguali.
foreach ($grp in ($rows | Group-Object Arm | Sort-Object Name)) {
    $h = @(Get-Distinct @($grp.Group | ForEach-Object { $_.OutHash }))
    if ($h.Count -ne 1) {
        throw ("il testo generato NON e' identico fra i run del braccio {0}: le sue ripetizioni hanno scritto testi diversi. I testi sono nei log; qui l'impronta di ogni run.`n  {1}" -f $grp.Name,
            (($grp.Group | ForEach-Object { "{0,-5} rip {1}: sha256 {2}" -f $_.Arm, $_.Rep, $_.OutHash.Substring(0,16) }) -join "`n  "))
    }
}
$tCold = @($rows | Where-Object { $_.Arm -eq "COLD" })[0]
$tWarm = @($rows | Where-Object { $_.Arm -eq $Warm })[0]
"testo generato identico fra i run di ogni braccio (sha256 dello stdout letto, senza la riga [meta]):"
"  COLD  {0}" -f $tCold.OutHash.Substring(0,16)
"  {0,-5} {1}" -f $Warm, $tWarm.OutHash.Substring(0,16)
""
"--- testo fra i due bracci (dichiarato: non ferma la corsa) ---"
# Confronto ORDINALE: con la cultura invariante -ceq tratta come uguali due
# testi che differiscono per un carattere ignorabile (per esempio U+200B).
if ([string]::Equals($tCold.Out, $tWarm.Out, [StringComparison]::Ordinal)) {
    "testo generato IDENTICO fra i due bracci (sullo stdout letto)"
} else {
    $a = $tCold.Out; $b = $tWarm.Out
    $n = [Math]::Min($a.Length, $b.Length); $i = 0
    while ($i -lt $n -and [int]$a[$i] -eq [int]$b[$i]) { $i++ }
    $prefix = ($i -eq $n)
    # Non partire da meta' di una coppia surrogata: la meta' alta prima e'
    # comune ai due testi.
    if ($i -gt 0 -and (($i -lt $a.Length -and [char]::IsLowSurrogate($a[$i])) -or ($i -lt $b.Length -and [char]::IsLowSurrogate($b[$i])))) { $i-- }
    # Gli estratti non cominciano ne' finiscono a meta' di una coppia
    # surrogata; un'emoji fatta di piu' caratteri (tono della pelle, ZWJ) puo'
    # comunque restare spezzata. I caratteri di controllo diventano '?'.
    function Get-Excerpt([string]$s, [int]$from) {
        $len = [Math]::Min(40, $s.Length - $from)
        if ($len -gt 0 -and [char]::IsHighSurrogate($s[$from + $len - 1])) { $len-- }
        ($s.Substring($from, $len) -replace "`n", " / ") -replace '\p{Cc}', '?'
    }
    if ($prefix) {
        "testo generato DIVERSO fra i due bracci: il testo di {0} e' l'inizio dell'altro e finisce alla posizione {1} (contata da 0, in unita' UTF-16, sullo stdout letto)." -f $(if ($a.Length -lt $b.Length) { "COLD" } else { $Warm }), $n
    } else {
        "testo generato DIVERSO fra i due bracci: primo carattere diverso alla posizione {0} (contata da 0, in unita' UTF-16, sullo stdout letto)." -f $i
    }
    "  COLD  ...{0}" -f (Get-Excerpt $a $i)
    "  {0,-5} ...{1}" -f $Warm, (Get-Excerpt $b $i)
}

# ---- statistica sui delta appaiati ---------------------------------------
$deltas = 1..$Reps | ForEach-Object {
    $rp = $_
    (($rows | Where-Object { $_.Rep -eq $rp -and $_.Arm -eq $Warm  }).Step) -
    (($rows | Where-Object { $_.Rep -eq $rp -and $_.Arm -eq "COLD" }).Step)
}

""
"--- delta appaiati ($Warm - COLD), ms/token ---"
($deltas | ForEach-Object { "{0:N2}" -f $_ }) -join "   "

$mean = ($deltas | Measure-Object -Average).Average
$sd   = if ($Reps -gt 1) { [math]::Sqrt((($deltas | ForEach-Object { [math]::Pow($_ - $mean, 2) }) | Measure-Object -Sum).Sum / ($Reps - 1)) } else { [double]::NaN }
$se   = $sd / [math]::Sqrt($Reps)
# quantili t al 95 % a due code: df 1..15, poi i df dispari fino a 29
# (-Reps 30). Oltre la tavola si usa z=1.96 e l'etichetta dell'intervallo lo
# dichiara.
$tq = @(12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131)
$df = $Reps - 1
$tq2 = @{ 17 = 2.110; 19 = 2.093; 21 = 2.080; 23 = 2.069; 25 = 2.060; 27 = 2.052; 29 = 2.045 }
$tApprox = $false
$t = if ($df -ge 1 -and $df -le 15) { $tq[$df-1] }
     elseif ($tq2.ContainsKey($df))  { $tq2[$df] }
     else { $tApprox = $true; 1.96 }
""
"media   {0,6:N2}   sd {1,5:N2}   se {2,5:N2}   df {3}" -f $mean, $sd, $se, $df
"intervallo 95 %{3}:  [{0:N2} ; {1:N2}]   ampiezza +-{2:N2}" -f ($mean-$t*$se), ($mean+$t*$se), ($t*$se), $(if ($tApprox) { " (APPROSSIMATO: z=1.96, df $df fuori tavola)" } else { "" })
"negativi {0}/{1}" -f (($deltas | Where-Object { $_ -lt 0 }).Count), $Reps

""
"--- controllo POSIZIONE (ABBA) ---"
$gA = @(1..$Reps | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { $deltas[$_-1] })
$gB = @(1..$Reps | Where-Object { $_ % 2 -eq 0 } | ForEach-Object { $deltas[$_-1] })
"COLD prima:  {0}" -f (($gA | ForEach-Object { "{0:N2}" -f $_ }) -join "  ")
"{0,-5} prima: {1}" -f $Warm, (($gB | ForEach-Object { "{0:N2}" -f $_ }) -join "  ")
"differenza fra le medie dei due gruppi: {0:N2} ms/token" -f `
    (($gA | Measure-Object -Average).Average - ($gB | Measure-Object -Average).Average)
"Stima il doppio dell'effetto di posizione dentro la coppia. Con -Reps pari"
"quell'effetto si annulla nella media dei delta se resta costante lungo la"
"sessione; se varia, si annulla solo in parte."

# ---- medie per braccio e indice per miss ---------------------------------
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
"--- indice per miss: 1000 * cpu-miss / miss(CPU), per braccio ---"
"(ms per token di decode diviso i miss dell'intera corsa: denominatori"
"diversi, quindi e' un indice e non un costo per miss)"
foreach ($grp in ($rows | Group-Object Arm | Sort-Object Name)) {
    $q = $grp.Group | Where-Object { $_.Miss -gt 0 }
    if ($q.Count -eq 0) { "{0,-5}  nessun miss" -f $grp.Name; continue }
    $idx = $q | ForEach-Object { 1000.0 * $_.CpuMiss / $_.Miss }
    # Nessuna unita' di tempo: la dimensione e' us per token per miss. Il
    # record che pubblicava un costo per miss l'ha ritirato per questa ragione
    # (qwen36-heatfile-2026-09-23-raw.txt:598-604).
    "{0,-5}  indice {1,6:N3}      (min {2:N3} max {3:N3} su {4} run)" -f $grp.Name,
        (($idx | Measure-Object -Average).Average), (($idx | Measure-Object -Minimum).Minimum),
        (($idx | Measure-Object -Maximum).Maximum), $q.Count
}

""
"Log per run: heatfile-COLD-r*.log e heatfile-$Warm-r*.log"
