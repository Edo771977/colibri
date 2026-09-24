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
#          (c/qwen36_tier.c:1264) scrive solo se la variabile c'e'. Se il
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
    # L'eseguibile che la build produce. $Exe ne e' una copia fatta a mano, e
    # niente lo legava alla build: dimenticare il "copy /Y" fa misurare il
    # binario di due giorni prima con TUTTI gli invarianti verdi -- piu' verdi
    # del normale, perche' [place] e la residenza sono identiche in tutti i run
    # proprio in quanto e' lo stesso vecchio binario. Passare -Built "" per
    # disattivare il confronto, consapevolmente.
    [string] $Built       = "qwen36.exe",
    [string] $Prompt      = "prompt25.txt",
    [string] $PromptCaldo = "prompt-caldo.txt",
    [int]    $Cap         = 256,
    [int]    $Bits        = 4,
    [int]    $Reps        = 6,
    [int]    $NNew        = 128,
    [string] $WorkDir     = "",
    # Variabili d'ambiente che l'operatore dichiara innocue per questa misura.
    # Esplicito, per non dover disarmare il controllo intero.
    [string[]] $AllowEnv  = @()
)

$ErrorActionPreference = "Stop"

# Questo script esiste per produrre un intervallo di confidenza su delta
# appaiati in ordine ABBA. Con -Reps 1 stampava "media -0.41  sd NaN  se NaN
# df 0  intervallo [NaN ; NaN]" -- un blocco pronto da incollare con la
# statistica interamente assente -- e df 0 cadeva nel ramo z=1.96 su una sola
# osservazione. Con un numero dispari l'alternanza ABBA non si chiude.
if ($Reps -lt 2 -or ($Reps % 2) -ne 0) {
    throw "-Reps $Reps non e' valido: servono almeno 2 ripetizioni e un numero PARI, altrimenti l'alternanza ABBA non si chiude e l'intervallo non esiste."
}
# Lo script sta in c\tools\, l'eseguibile e i prompt in c\. Una versione
# precedente faceva Set-Location $PSScriptRoot e poi cercava .\qwen36_clang.exe
# nella cartella sbagliata.
if (-not $WorkDir) { $WorkDir = Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path -LiteralPath $WorkDir)) { throw "cartella di lavoro non trovata: $WorkDir" }
# Normalizzato PRIMA del Set-Location: piu' sotto $WorkDir viene riusato, e un
# valore relativo (-WorkDir ..) si risolverebbe rispetto alla cwd NUOVA, cioe'
# alla cartella sbagliata.
$WorkDir = (Get-Item -LiteralPath $WorkDir).FullName
Set-Location -LiteralPath $WorkDir
"cartella di lavoro: {0}" -f (Get-Location).Path

# ---- guardie ambiente ----------------------------------------------------
# Una DENYLIST scritta a mano non regge: il motore legge 55 variabili
# d'ambiente e l'elenco sarebbe sempre indietro. La versione committata
# precedente ne nominava tre, e due erano morte per questo eseguibile:
# COLI_GRAPH_DIAG non e' letta da nessun file C dell'albero, e PROF solo da
# c/colibri.c, che non entra in questa riga di link. La terza,
# COLI_CUDA_PROFILE, e' viva ma la legge coli_cuda.dll (c/backend_cuda.cu), non
# l'host: sfuggiva quindi a chi cercasse i getenv del solo qwen36.c.
#
# Omesse invece le variabili del runtime OpenMP. OMP_WAIT_POLICY e compagne le
# imposta deliberatamente c/colibri.c sulle piattaforme non-Apple, e qwen36.c
# non fa alcun setenv (verificato: zero occorrenze), quindi per qwen36.exe
# arrivano interamente dalla shell: il valore dell'operatore vince e non lascia
# traccia. NON si cita qui una cifra: il "+122% decode" a c/colibri.c:11024 e'
# un REGRESSO misurato su macOS/Apple Silicon con libomp LLVM sul motore
# GLM-5.2 int4, e su Windows/x86/CUDA quel dato non si trasferisce -- il repo
# li' considera la stessa variabile benefica.
#
# Quindi si rovescia: fallisce QUALUNQUE variabile che possa toccare il motore,
# tranne quelle che lo script imposta lui stesso e quelle che l'operatore
# dichiara innocue con -AllowEnv.
$EnvOwn = @("SNAP","COLI_CUDA","COLI_GPUS","COLI_TIMERS","COLI_PLACE","HEAT_FILE","N_NEW")
# Con -File, PowerShell passa "-AllowEnv A,B" come UN SOLO elemento di
# [string[]]: la valvola di sfogo era rotta per il modo d'uso documentato in
# testa a questo file. Va quindi rispezzata a mano.
$AllowEnv = @($AllowEnv | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
# CUDA_PATH e la sua famiglia le scrive l'installer del toolkit: sono percorsi,
# non interruttori. CUDA_VISIBLE_DEVICES non e' fra queste e viene rifiutata a
# ragione, perche' rimapperebbe il COLI_GPUS=0 che lo script imposta.
$EnvBenign = '^CUDA_(PATH|HOME|BIN_PATH|LIB_PATH|INC_PATH|CACHE_PATH|MODULE_LOADING_MODE)'
# I prefissi non bastano: QT_ collide con il namespace del framework Qt, che
# Anaconda e vari installer impostano, quindi si nominano le due sole variabili
# QT_ che il motore legge davvero; e sedici getenv del motore non hanno alcun
# prefisso riconoscibile. Fra quelle la piu' grave e' MODEL (c/qwen36.c), che
# cambia il modello misurato senza che nulla nel confronto fra i bracci lo
# riveli; poi CTX e Q36_MAXT, che muovono il tetto di contesto e
# l'allocazione KV, e SERVE/PPL/CONSIST, che cambiano modalita'.
# WIDE invece NON e' un buon esempio, contrariamente a quanto diceva una
# versione precedente di questo commento: g_wide compare solo dentro
# pilot_prefetch (c/qwen36.c), che gira solo con PILOT >= 1, e PILOT vale 0 per
# default ed e' rifiutata qui sopra -- quindi con questo script WIDE e' inerte.
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

# Un .exe che esiste ma e' vuoto o non avviabile da' "non e' un'applicazione
# valida per questo sistema operativo" con una traccia PowerShell illeggibile.
# Succede davvero: incollare in cmd un transcript che contiene il prompt
# "C:\...\c>" fa leggere quel ">" come redirezione e TRONCA il file che segue.
#
# La prima versione di questa guardia confrontava la dimensione con 1MB. Era
# una soglia indovinata, mai misurata contro un binario reale, e ha bocciato un
# build clang perfettamente sano da 368128 byte. Una soglia in byte non puo'
# fare questo lavoro: non distingue un eseguibile da un file di testo e non sa
# quanto debba essere grande un binario legittimo, che dipende dai flag di link
# (c/Makefile:142: WIN_STATIC aggiunge -static ai build gcc e non a quelli
# clang, e questo da solo la porta da 368 128 a 1 083 842 byte, 2,94 volte).
#
# Il test giusto e' la struttura, non la taglia: i due byte 'MZ' che aprono
# ogni PE.
if ($exeLen -eq 0) { throw "$Exe e' di 0 byte: qualcosa lo ha troncato -- ricostruisci: make -B qwen36.exe CC=clang CUDA_DLL=1 ARCH=native && copy /Y qwen36.exe qwen36_clang.exe" }
$fs = [System.IO.File]::OpenRead($exeItem.FullName)
try { $b0 = $fs.ReadByte(); $b1 = $fs.ReadByte() } finally { $fs.Dispose() }
if ($b0 -ne 0x4D -or $b1 -ne 0x5A) {
    throw "$Exe ($exeLen byte) non inizia con la firma PE 'MZ': non e' un eseguibile Windows."
}

# La guardia che serviva davvero, e che nessuna soglia puo' dare: l'eseguibile
# misurato e' la copia di quello che la build ha appena prodotto? Chiude due
# casi che la soglia lasciava passare -- il "copy /Y" dimenticato (binario
# stantio, numeri attribuiti a un commit mai eseguito) e il "copy /Y"
# interrotto a meta' (PE tronco ma con la firma MZ intatta).
# I sorgenti e .build-config si leggono relativi a $WorkDir, $Exe no: con un
# eseguibile fuori dall'albero lo script stampava la sua impronta e, subito
# sotto, la build-config di un ALTRO albero, come se appartenessero allo stesso
# binario. La riga sulla toolchain non vale nulla se non e' legata al file
# misurato.
$wdFull = (Get-Item -LiteralPath $WorkDir).FullName.TrimEnd([IO.Path]::DirectorySeparatorChar)
if ($exeItem.DirectoryName.TrimEnd([IO.Path]::DirectorySeparatorChar) -ne $wdFull) {
    throw "$Exe non sta nella cartella di lavoro ($WorkDir): .build-config e i sorgenti verrebbero letti da un albero diverso da quello dell'eseguibile, e la toolchain stampata non sarebbe la sua."
}
$exeHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
if ($Built) {
    if (-not (Test-Path -LiteralPath $Built)) {
        throw "manca ${Built}: non posso verificare che $Exe venga dalla build corrente. Compila, oppure passa -Built '' per rinunciare al controllo -- ma allora la provenienza non e' verificata e il record va scritto dicendolo."
    }
    $builtItem = Get-Item -LiteralPath $Built
    if ($builtItem.PSIsContainer) { throw "${Built} e' una cartella, non un file." }
    # $Exe e $Built devono essere due file DISTINTI: se sono lo stesso file
    # l'hash coincide per costruzione e la guardia si autoconferma. Un hardlink
    # o un symlink sfuggono a questo confronto (FullName differisce) e NON sono
    # coperti: in compenso rendono impossibile dimenticare la copia, che e' il
    # caso per cui la guardia esiste.
    if ($builtItem.FullName -eq $exeItem.FullName) {
        throw "$Exe e ${Built} sono lo stesso file: il confronto di provenienza si autoconfermerebbe. Passa -Built con il vero output della build, o -Built '' dichiarando che non e' verificato."
    }
    $builtHash = (Get-FileHash -LiteralPath $Built -Algorithm SHA256).Hash
    if ($builtHash -ne $exeHash) {
        # I path stanno fuori dalla stringa di formato: dentro, un '{0}' nel
        # nome del file verrebbe sostituito con l'hash.
        throw ("{0} NON e' la copia di {1} (sha256 {2} contro {3}). Manca il 'copy /Y', oppure la copia e' incompleta: misureresti un binario diverso da quello compilato." -f $Exe, $Built, $exeHash.Substring(0,16), $builtHash.Substring(0,16))
    }
}

# Staleness. `copy` preserva il LastWriteTime della sorgente, quindi la data di
# $Exe e' quella della BUILD, non della copia: questo confronto replica cio' che
# make gia' fa, e serve per il caso in cui nessuno ha invocato make.
#
# La lista e' i prerequisiti di qwen36$(EXE) in c/Makefile:1223. Scriverne
# cinque a mano su ventuno lasciava passare una modifica a decode_batch.h,
# simd_i8f.h, omp_tune.h, kv_prefix.h o st.h -- cioe' al percorso caldo del
# decode -- senza un avviso. .build-config e' in quella lista e va controllato
# come gli altri: se e' piu' recente dell'eseguibile, la configurazione
# registrata NON e' quella del binario, e lo stamp qui sotto mentirebbe.
$QwenSrc = @(
    "qwen36.c","qwen36_tier.c","qwen36_tier.h","expert_ffn.h","simd_i8f.h",
    "decode_batch.h","serve_poll.h","cli_args.h","st.h","json.h","compat.h",
    "omp_tune.h","kv_prefix.h","pin_pool.h",
    "edge_adapter_internal.h","edge_adapters.h","edge_runtime.h",
    "segment_adapter_internal.h","segment_adapters.h","segment_runtime.h",
    # $(CUDA_OBJ) sotto CUDA_DLL=1 e' backend_loader.o (c/Makefile:596-598), che
    # dipende da backend_loader.c e backend_cuda.h (c/Makefile:847). Ometterli
    # lasciava fuori dal controllo proprio il pezzo che carica la DLL dove sta
    # tutto il calcolo esperti.
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

# c/Makefile:823 definisce BUILD_CONFIG come CC|CFLAGS|LDFLAGS|CUDA|CUDA_DLL|
# ARCH|... e :832 lo scrive in .build-config; $(CC) e' il primo campo: e' l'unico posto dell'albero che registra CHI ha
# compilato. Il nome qwen36_clang.exe e' il nome di una copia fatta a mano e non
# prova nulla da solo. Senza questa riga un record non puo' dichiarare la
# toolchain, quindi l'assenza e' un errore e non una nota.
$BuildCfg = (Get-Content -LiteralPath ".build-config" -Raw -Force)
if ($null -eq $BuildCfg -or -not $BuildCfg.Trim()) {
    throw ".build-config e' vuoto: la toolchain di questo binario non e' registrata da nessuna parte e il record non potrebbe dichiararla. Ricompila."
}
$BuildCfg = $BuildCfg.Trim()
$CfgStamp = "build-config: $BuildCfg"

# Con CUDA_DLL=1 l'host che stiamo hashando non contiene il calcolo esperti:
# CUDA_OBJ e' il solo backend_loader.o (c/Makefile:596-598) e tutti i kernel
# stanno in coli_cuda.dll (COLI_BACKEND_DLL, c/backend_loader.c:57), caricata a
# runtime. Una DLL stantia cambia ogni tempo misurato con l'host verificato.
# Non posso legarla alla build -- e' costruita a parte con nvcc -- ma la sua
# impronta va nel record, che e' la differenza fra un dato e un'omissione.
$DllStamp = "coli_cuda.dll: NON TROVATA nella cartella di lavoro -- il loader la prende da altrove e la sua identita' NON e' registrata"
foreach ($d in @("coli_cuda.dll","coli_hip.dll")) {
    if (Test-Path -LiteralPath $d) {
        $di = Get-Item -LiteralPath $d
        $DllStamp = "{0}: {1} byte | {2} | sha256 {3}" -f `
            $d, $di.Length, $di.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"),
            (Get-FileHash -LiteralPath $d -Algorithm SHA256).Hash.Substring(0,16)
        break
    }
}

$ExeStamp
$CfgStamp
$DllStamp

if (-not (Test-Path -LiteralPath $Prompt))      { throw "manca $Prompt" }
if (-not (Test-Path -LiteralPath $PromptCaldo)) { throw "manca $PromptCaldo -- serve un prompt di argomento DIVERSO da $Prompt" }
if (-not (Test-Path -LiteralPath $Snap))        { throw "modello non trovato in $Snap -- passalo con -Snap <dir>" }

# La premessa dichiarata del braccio CROSS e' che la tabella sia stata
# costruita su un argomento ESTRANEO. Se i due prompt hanno lo stesso
# contenuto, CROSS misura uno scaldamento sullo stesso argomento e la premessa
# cade, senza che nulla nell'output lo riveli.
if ((Get-FileHash -LiteralPath $Prompt -Algorithm SHA256).Hash -eq
    (Get-FileHash -LiteralPath $PromptCaldo -Algorithm SHA256).Hash) {
    throw "$Prompt e $PromptCaldo hanno lo stesso contenuto: il braccio CROSS misurerebbe uno scaldamento sullo stesso argomento, non su uno estraneo."
}

# I parametri della corsa vanno nel record insieme ai numeri: -Cap 64 o
# -NNew 16 da riga di comando non lasciavano alcuna traccia nell'output.
# Le deroghe devono comparire nel record come tutto il resto: -Built "" e
# -AllowEnv disattivano controlli, e affidare all'operatore il compito di
# dichiararlo a voce e' esattamente cio' che questo script non fa per nient'altro.
$ParamStamp = "parametri: cap=$Cap bits=$Bits N_NEW=$NNew rip=$Reps | snap=$Snap | prompt=$Prompt caldo=$PromptCaldo" +
    ("  | built={0}" -f $(if ($Built) { $Built } else { "'' -- PROVENIENZA NON VERIFICATA" })) +
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

    # Il log va RIMOSSO prima della chiamata, non soltanto verificato dopo.
    # Quando il LANCIO del nativo fallisce (PE non valido, DLL mancante),
    # Out-File non tocca un log preesistente, e questo script non cancella mai
    # i log -- li annuncia in coda come artefatto da conservare. Quindi
    # heatfile-COLD-r1.log esiste sempre, dalla corsa precedente. Con
    # $LASTEXITCODE = 0 ereditato da una chiamata nativa riuscita prima (il
    # warmup, o l'altro braccio della coppia), una versione che verificava solo
    # l'ESISTENZA del log passava tutti i controlli e Get-Content restituiva la
    # misura di DUE GIORNI PRIMA come risultato di questo run. Verificare che il
    # log esista non basta: bisogna garantire che sia di questa corsa.
    Remove-Item -LiteralPath $Log -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $Log) {
        throw "non riesco a rimuovere $Log prima del run: non posso garantire che il log appartenga a questa corsa."
    }

    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"      # il motore scrive tutto su stderr
    $code = $null
    try {
        # $exeItem.FullName, non $Exe: per un comando nativo PowerShell NON
        # cerca nella cartella corrente, risolve un nome nudo sul PATH. Con un
        # omonimo sul PATH tutte le guardie validavano il file nella work dir e
        # la misura veniva da un altro binario, con lo stamp di provenienza che
        # dichiarava il primo -- esattamente lo scenario che le guardie esistono
        # per chiudere.
        & $exeItem.FullName $Cap $Bits $PromptFile 2>&1 | Out-File -Encoding utf8 $Log
        $code = $LASTEXITCODE
    } catch {
        # Fuori da un try, con ErrorActionPreference = Continue, il fallimento di
        # lancio del nativo prosegue lasciando $LASTEXITCODE non impostato e la
        # traccia "ResourceUnavailable ... failed to run ... Exec format error"
        # sulla console. Dentro un try viene catturato in entrambe le
        # preferenze, e il messaggio leggibile prende il posto della traccia.
        # Il testo .NET originale porta ancora in coda la posizione nel file:
        # la taglio, altrimenti la traccia rientra dalla finestra.
        $ErrorActionPreference = $old
        # -csplit, non -split: -split e' case-insensitive per default e la sua
        # alternativa 'At ' combacia con l'"at " dentro "form-at error",
        # lasciando "... Exec form" e buttando via la DIAGNOSI invece della
        # posizione nel file. Verificato eseguendolo.
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
    # miss(CPU) e LFRU swaps non possono ripiegare in silenzio su -1. Con
    # Miss = -1 il blocco "costo per miss" in coda -- il motivo dichiarato di
    # esistere di questo script -- filtra su Miss -gt 0, trova il gruppo vuoto
    # e stampa "nessun miss" proseguendo: il controllo principale evaporerebbe
    # senza un errore se il formato di quella riga cambiasse.
    # I token di decode: il motore li stampa e nessuno li leggeva. L'indice per
    # miss qui sotto e' 1000*(ms per token di DECODE)/(miss di TUTTA la corsa):
    # se i due bracci decodificano un numero diverso di token -- e possono, per
    # esempio fermandosi a un EOS diverso -- l'indice si muove per quel solo
    # motivo. E' il confondimento che il record rifiuta esplicitamente, con un
    # contatore che era nel log tutto il tempo.
    if ($Text -notmatch '(?m)^\s*\[timers\]\s+decode:\s*([0-9]+)\s+tokens') {
        throw "$Log : riga '[timers] decode: N tokens' non trovata. Senza il numero di token di decode l'indice per miss non e' confrontabile fra i bracci."
    }
    $dtok = [int]$Matches[1]
    if ($Text -notmatch 'miss\(CPU\)\s+([0-9]+)') { throw "$Log : riga 'miss(CPU) N' non trovata. Il controllo sul costo per miss passerebbe a vuoto." }
    $miss  = [int]$Matches[1]
    if ($Text -notmatch 'LFRU swaps\s+([0-9]+)')   { throw "$Log : riga 'LFRU swaps N' non trovata." }
    $swaps = [int]$Matches[1]
    $res   = if ($Text -match 'resident\s+([0-9]+)/([0-9]+)\s+experts')     { "$($Matches[1])/$($Matches[2])" } else { "?" }
    $toks  = if ($Text -match 'Speed:\s*([0-9.]+)\s*tok/s')                 { [double]$Matches[1] } else { [double]::NaN }
    # TUTTE le righe [place] auto:, non solo la prima. Il motore ne stampa una
    # per layer che resta sulla CPU e una di riepilogo del trunk
    # (c/qwen36_tier.c): prendendo la prima, l'invariante che il commento chiama
    # "il piazzamento del trunk" poteva confrontare una riga per-layer e non
    # vedere affatto il riepilogo.
    $place = (([regex]::Matches($Text, '(?m)^\[place\] auto:.*$') |
               ForEach-Object { $_.Value.Trim() }) -join ' ;; ')

    # Questi due alimentano gli invarianti in coda. Se il regex non trova
    # nulla, Place resta "" e Resident resta "?" per TUTTI i run: l'elenco
    # unico ha un solo elemento e lo script stampa "identica in tutti i run"
    # sull'ASSENZA del dato invece che sulla sua costanza. Un invariante che
    # passa a vuoto e' peggio di un invariante che manca, perche' viene
    # ricopiato nel record come se avesse verificato qualcosa.
    # Il motore dichiara la propria configurazione in una riga sola
    # (c/qwen36.c:4236: cap, bits, ctx, pilot, wide, hot, smooth, conf).
    # Pretenderla identica fra i run copre cap, bits, Q36_MAXT (il campo ctx e'
    # qwen36_max_ctx(), che legge Q36_MAXT e NON la variabile CTX), PILOT, WIDE,
    # HOT, SMOOTH e CONF_LIMIT. Delle sedici variabili che sfuggivano alla
    # denylist precedente il banner ne aggiunge tre: PILOT, WIDE e Q36_MAXT --
    # le altre erano gia' coperte per nome. Il guadagno non e' il numero: e' che
    # qui non si indovina cosa l'ambiente abbia fatto, si legge cosa il motore
    # ha deciso, e nessuna lista di nomi puo' dare la stessa cosa.
    # \s*$ e non $: in .NET, in modalita' multiline, $ aggancia la posizione
    # PRIMA del \n, quindi un letterale come "==" davanti a $ non puo' essere
    # seguito dal \r di un log CRLF -- e Out-File scrive Environment.NewLine,
    # che su Windows e' CRLF. Senza \s* questo invariante non combaciava MAI
    # sulla piattaforma di questo script e la corsa moriva al primo run. Il
    # pattern di [place] sopravvive perche' finisce in .*, che assorbe il \r.
    if ($Text -notmatch '(?m)^(== qwen36 Phase-2 engine \|.*==)\s*$') {
        throw "$Log : il banner '== qwen36 Phase-2 engine |' non c'e'. Il motore non e' arrivato a dichiarare la sua configurazione, oppure non e' qwen36."
    }
    $banner = $Matches[1].Trim()

    if (-not $place) { throw "$Log : riga '[place] auto:' non trovata. L'invariante sul piazzamento del trunk passerebbe a vuoto." }
    if ($res -eq "?") { throw "$Log : riga 'resident N/M experts' non trovata. L'invariante sulla residenza passerebbe a vuoto." }

    [pscustomobject]@{
        Step=$step; Dn=$g["deltanet"]; Attn=$g["attention"]; Moe=$g["moe_total"]; Head=$g["lm_head"]
        Issue=$g["issue"]; CpuMiss=$g["cpu-miss"]; Take=$g["take"]; ShOvl=$g["shared-ovl"]
        Vram=$vram; Swaps=$swaps; Miss=$miss; Resident=$res; Toks=$toks; Place=$place; Banner=$banner
        DecTok=$dtok
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

$HeatStamp = ""
if (-not (Test-Path "heat.caldo.bin")) {
    "heat.caldo.bin non c'e': la costruisco su $PromptCaldo (un run)..."
    Remove-Item heat.bin -Force -ErrorAction SilentlyContinue
    $txt0 = Invoke-Engine $true $PromptCaldo "heatfile-warmup.log"
    $r0 = Read-Run $txt0 "heatfile-warmup.log"
    if (-not (Test-Path "heat.bin")) { throw "la scaldata non ha scritto heat.bin -- vedi heatfile-warmup.log" }
    Copy-Item heat.bin heat.caldo.bin -Force
    "scaldata ok ({0:N2} tok/s, hit {1:N1} %), tabella congelata in heat.caldo.bin" -f $r0.Toks, $r0.Vram
    ""
    $HeatStamp = "COSTRUITA in questa corsa su $PromptCaldo"
} else {
    # Una heat.caldo.bin preesistente veniva riusata senza alcuna verifica e
    # senza comparire da nessuna parte. E' la variabile INDIPENDENTE
    # dell'esperimento -- l'unico input che differisce fra i bracci -- e il
    # motore ne valida solo magic, n_layers e n_experts, quindi qualunque
    # tabella per questa forma di modello si carica e stampa "HEAT_FILE
    # loaded", superando anche quel controllo. Non posso provare da dove
    # venga; posso impedirle di restare invisibile.
    $HeatStamp = "RIUSATA da una corsa precedente -- la sua provenienza NON e' verificata"
}
$hc = Get-Item -LiteralPath "heat.caldo.bin"
$HeatStamp = "heat.caldo.bin: {0} byte | {1} | sha256 {2} | {3}" -f `
    $hc.Length, $hc.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"),
    (Get-FileHash -LiteralPath "heat.caldo.bin" -Algorithm SHA256).Hash.Substring(0,16), $HeatStamp
$HeatStamp
""

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
        Remove-Item heat.bin -Force -ErrorAction SilentlyContinue
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
$ExeStamp
$CfgStamp
$DllStamp
$HeatStamp
$ParamStamp
# Un invariante VIOLATO era un avviso, e lo script proseguiva fino a stampare
# l'intero blocco statistico pronto da incollare: la riga ATTENZIONE si perde
# in quaranta righe di output, mentre dice essa stessa che in quel caso
# 'attention' non e' piu' un controllo. Se un invariante non tiene, i numeri
# non vanno prodotti in forma incollabile.
$banners = @($rows | ForEach-Object { $_.Banner } | Sort-Object -Unique)
if ($banners.Count -ne 1) {
    throw ("il motore NON ha girato con la stessa configurazione in tutti i run:`n  {0}" -f ($banners -join "`n  "))
}
"configurazione motore identica in tutti i run:"
"  {0}" -f $banners[0]
$places = @($rows | ForEach-Object { $_.Place } | Sort-Object -Unique)
if ($places.Count -ne 1) {
    throw ("la riga [place] auto: NON e' identica in tutti i run -- il piazzamento del trunk e' cambiato fra i bracci, quindi 'attention' non e' piu' un controllo e i bracci non sono confrontabili:`n  {0}" -f ($places -join "`n  "))
}
"[place] auto: identica in tutti i {0} run" -f @($rows).Count
$residents = @($rows | ForEach-Object { $_.Resident } | Sort-Object -Unique)
if ($residents.Count -ne 1) {
    throw ("residenza diversa fra i run: {0} -- i bracci non hanno lo stesso numero di esperti in VRAM e il delta non e' attribuibile alla tabella heat." -f ($residents -join ", "))
}
"residenza identica in tutti i run: {0}" -f $residents[0]
# Se i bracci non decodificano lo stesso numero di token, l'indice per miss
# qui sotto confronta due rapporti con denominatori diversi e non significa
# niente -- il record lo dice a chiare lettere, e il contatore era nel log.
$dtoks = @($rows | ForEach-Object { $_.DecTok } | Sort-Object -Unique)
if ($dtoks.Count -ne 1) {
    throw ("i run non hanno decodificato lo stesso numero di token: {0}. L'indice per miss non e' confrontabile fra i bracci." -f ($dtoks -join ", "))
}
"token di decode identici in tutti i run: {0}" -f $dtoks[0]

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
# Oltre la tavola si usava 1.96, lo z asintotico, continuando a stampare
# "intervallo 95 %": a df 17 il t vero e' 2.110, quindi l'intervallo usciva
# l'8 % piu' stretto di quello che dichiarava. Estesa ai df pari raggiungibili
# con -Reps pari (17, 19, 21, 23, 25, 29) e, oltre, si dice cosa si sta usando.
$tq2 = @{ 17 = 2.110; 19 = 2.093; 21 = 2.080; 23 = 2.069; 25 = 2.060; 29 = 2.045 }
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
    # nessuna unita' di tempo: 1000*(ms per token di decode)/(miss di tutta la
    # corsa) ha dimensione us per token per miss, non us. Stamparlo come "us"
    # due righe sotto i costi per miss veri del record (79 e 101 us) metteva
    # tre ordini di grandezza sotto la stessa etichetta.
    "{0,-5}  indice {1,6:N3}      (min {2:N3} max {3:N3} su {4} run)" -f $grp.Name,
        (($idx | Measure-Object -Average).Average), (($idx | Measure-Object -Minimum).Minimum),
        (($idx | Measure-Object -Maximum).Maximum), $q.Count
}
"Se i due indici restano distanti oltre la loro dispersione, il costo per miss"
"dipende davvero dal batching e nessun numero per-miss si puo' estrapolare."

""
"Log per run: heatfile-COLD-r*.log e heatfile-CROSS-r*.log"
