# prova-40.ps1 -- la prova della #40, comprese le DUE prove inverse.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File tools\prova-40.ps1
#
# PERCHE' ESISTE. La #40 aggiunge due guardie all'invariante dei buffer del
# grafo, e nessuna delle due e' mai stata vista fallire. Un test che non ho
# mai visto diventare rosso non ho motivo di credere che guardi qualcosa:
# quattro delle cinque round di review su questa PR hanno trovato proprio
# questo, asserzioni vere per costruzione.
#
# Quindi qui ogni guardia viene prima vista passare e poi ROTTA APPOSTA, e il
# criterio e' che diventi rossa con il messaggio giusto.
#
# LE DUE META' SONO DIVERSE E VANNO PROVATE DIVERSAMENTE:
#
#   tests/test_graph_reserve_wrappers.py  -- legge backend_cuda.cu e vieta una
#       reserve() nuda su uno degli otto buffer. Non serve la scheda.
#       Non vede la funzione di invalidazione svuotata: quel buco e'
#       dichiarato nel suo docstring, e la FASE 4 lo dimostra invece di
#       crederci sulla parola.
#
#   tests/test_grouped_g4_cuda.cu  -- fa crescere i buffer davvero e pretende
#       che il grafo sia RI-CATTURATO. Questa meta' serve la scheda, ed e'
#       l'unica che vede la funzione svuotata.
#
# COSA FA, in ordine:
#   1. il test Python passa sul file intatto
#   2. MUTAZIONE A: reserve_graph -> reserve nuda su ctx->gate.
#      Il test Python deve diventare ROSSO E NOMINARE LA RIGA.
#   3. make cuda-test sul file intatto (la meta' positiva, e la casella del
#      template del PR)
#   4. MUTAZIONE B: la funzione di invalidazione svuotata.
#      Il test Python deve restare VERDE  -> il buco dichiarato e' reale
#      Il test g4 deve diventare ROSSO    -> la meta' CUDA lo prende
#   5. ripristino e riverifica che il g4 torni verde
#
# Il file viene salvato prima di toccarlo e ripristinato in un finally, quindi
# anche un Ctrl-C lo rimette a posto. Alla fine lo script controlla da solo
# che backend_cuda.cu sia tornato identico all'originale.

param(
    [string] $WorkDir = (Split-Path -Parent $PSScriptRoot),
    [switch] $SkipFullCudaTest    # salta la FASE 3 (lunga) e tiene le altre
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkDir

$sorgente = "backend_cuda.cu"
if (-not (Test-Path $sorgente)) { throw "non trovo ${sorgente}: lanciami dalla cartella c\ o passa -WorkDir" }

$backup = Join-Path ([System.IO.Path]::GetTempPath()) "backend_cuda.cu.prova40"
Copy-Item $sorgente $backup -Force
$digestIniziale = (Get-FileHash -Algorithm SHA256 -LiteralPath $sorgente).Hash

function Ripristina {
    Copy-Item $backup $sorgente -Force
    $d = (Get-FileHash -Algorithm SHA256 -LiteralPath $sorgente).Hash
    if ($d -ne $digestIniziale) { throw "RIPRISTINO FALLITO: $sorgente non e' tornato all'originale. La copia buona e' in $backup" }
}

# Applica una sostituzione letterale unica. Se il testo cercato non c'e', o
# c'e' piu' di una volta, si ferma: una mutazione che non muta farebbe
# "passare" la prova inversa per il motivo sbagliato.
#
# LE ANCORE STANNO SU UNA RIGA SOLA, DI PROPOSITO. Un'ancora a cavallo di due
# righe porta dentro un \n, e su un checkout Windows il sorgente puo' essere in
# CRLF: la ricerca non troverebbe niente, il controllo "compare 1 volta"
# fermerebbe lo script, e nel caso peggiore una mutazione che non muta farebbe
# sembrare provata una guardia mai rotta. E' lo stesso difetto che ha reso
# rossa la CI della #41.
function Muta([string]$Cerca, [string]$Sostituisci, [string]$Nome) {
    $testo = [System.IO.File]::ReadAllText((Resolve-Path $sorgente))
    $n = ([regex]::Matches($testo, [regex]::Escape($Cerca))).Count
    if ($n -ne 1) { throw "${Nome}: il testo da mutare compare $n volte, non 1. Il sorgente e' cambiato: aggiorna questo script invece di forzarlo." }
    [System.IO.File]::WriteAllText((Resolve-Path $sorgente), $testo.Replace($Cerca, $Sostituisci))
    "  mutazione applicata: $Nome"
}

# ---- ESECUZIONE DI COMANDI ESTERNI ---------------------------------------
# Start-Process -PassThru, non `& cmd /c`. Sulla macchina di destinazione la
# forma `& cmd /c ...` ha restituito codice di uscita VUOTO e output vuoto per
# OGNI comando -- python, make, tutti -- e le fasi hanno continuato a girare
# su quei vuoti: `$null -ne 0` e' vero, quindi una fase ha perfino stampato OK.
# Un risultato falso e' peggio di un errore.
#
# $p.ExitCode viene dall'oggetto processo e non dipende da $LASTEXITCODE, che
# e' quello che si era svuotato. I due flussi vanno su file separati perche'
# Start-Process non li sa unire, e vengono poi concatenati.
function Esegui([string]$Comando, [int]$TimeoutSec = 1800) {
    $base = [System.IO.Path]::GetTempFileName()
    $bat = "$base.cmd"; $fout = "$base.out"; $ferr = "$base.err"; $frc = "$base.rc"
    # IL BATCH SCRIVE IL PROPRIO CODICE DI USCITA IN UN FILE, e quel file e'
    # l'autorita'. $p.ExitCode di Start-Process -PassThru ha restituito 0 su
    # comandi chiaramente falliti (il test Python stampava un AssertionError
    # accanto a un codice 0), e `exit /b %ERRORLEVEL%` da solo non l'ha
    # risolto: la lettura resta 0. Un file scritto da dentro il processo non
    # dipende da come PowerShell sincronizza l'oggetto processo.
    #
    # L'ordine conta: %ERRORLEVEL% va catturato PRIMA di scriverlo, perche'
    # l'echo stesso lo azzererebbe.
    Set-Content -LiteralPath $bat -Encoding ASCII -Value @(
        "@echo off",
        $Comando,
        "set RC=%ERRORLEVEL%",
        "> ""$frc"" echo %RC%",
        "exit /b %RC%")
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $bat) -PassThru `
                       -NoNewWindow -RedirectStandardOutput $fout -RedirectStandardError $ferr
    # -Wait senza limite trasforma un processo bloccato in uno script
    # bloccato, che e' cio' che e' successo alla fase 5: nessun messaggio,
    # nessuna diagnosi, solo attesa. Con un limite il blocco diventa un
    # risultato: il processo viene ucciso e la fase risulta rossa dicendo
    # che e' scaduta.
    $scaduto = $false
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $scaduto = $true
        try { $p.Kill($true) } catch { try { $p.Kill() } catch {} }
        try { $p.WaitForExit(15000) | Out-Null } catch {}
    }
    # stdout e stderr separati e ETICHETTATI. Concatenandoli si perdeva
    # l'informazione piu' utile del primo giro reale: il test mutato aveva
    # stderr pieno e stdout VUOTO, che e' la firma di una terminazione
    # anomala (stdout verso file e' bufferizzato e va perso, stderr no).
    $out = ""; $err = ""
    if (Test-Path $fout) { $t = Get-Content -Raw -LiteralPath $fout; if ($t) { $out = $t } }
    if (Test-Path $ferr) { $t = Get-Content -Raw -LiteralPath $ferr; if ($t) { $err = $t } }
    $testo = ""
    if ($out) { $testo += "--- stdout ---`n" + $out }
    if ($err) { $testo += "--- stderr ---`n" + $err }
    if (-not $testo) { $testo = "(nessun output su nessuno dei due flussi)" }

    # IL CODICE VA LETTO PRIMA DELLA PULIZIA. In una stesura intermedia la
    # cancellazione dei temporanei stava sopra questa lettura, quindi il file
    # non c'era mai e si ricadeva sempre su $p.ExitCode -- cioe' la
    # correzione non correggeva niente.
    $codice = [int]$p.ExitCode
    if (Test-Path $frc) {
        $t = (Get-Content -Raw -LiteralPath $frc).Trim()
        if ($t -match '^\d+$') { $codice = [int]$t }
    }

    foreach ($f in @($base, $bat, $fout, $ferr, $frc)) {
        Remove-Item -Force -LiteralPath $f -ErrorAction SilentlyContinue
    }
    if ($scaduto) {
        return @{ Testo = ("TIMEOUT dopo $TimeoutSec s. Output raccolto fino a li':`n" + $testo)
                  Codice = -1; Scaduto = $true; Esa = "TIMEOUT"
                  StdOut = [string]$out; StdErr = [string]$err }
    }
    return @{ Testo = [string]$testo; Codice = $codice; Scaduto = $false
              Esa = ("0x{0:X8}" -f [uint32]([int64]$codice -band 0xFFFFFFFF))
              StdOut = [string]$out; StdErr = [string]$err }
}

# Stampa le ultime righe dell'output di un comando fallito. Una fase rossa
# senza il motivo obbliga a rifare tutto a mano.
function Mostra($Risultato, [int]$Righe = 12) {
    $t = $Risultato.Testo
    if ([string]::IsNullOrWhiteSpace($t)) { "       (nessun output, codice $($Risultato.Codice))"; return }
    ($t -split "`r?`n" | Where-Object { $_ -ne "" } | Select-Object -Last $Righe) |
        ForEach-Object { "       $_" }
}

$fallite = @()
function Verifica([bool]$Condizione, [string]$Cosa) {
    if ($Condizione) { "  OK   $Cosa" }
    else { "  ROSSA $Cosa"; $script:fallite += $Cosa }
}

# ---- L'ALBERO E' QUELLO GIUSTO? ------------------------------------------
# Questo script dimostra la #40, quindi il working tree DEVE contenere la #40.
# Senza questa guardia lo script girava comunque su un checkout di main: la
# fase 1 usciva rossa perche' il test non esiste li', e la mutazione non
# trovava la sua ancora perche' il codice non esiste li'. Due risultati
# perfettamente spiegabili e completamente fuorvianti -- esattamente il
# genere di "rosso" che fa perdere un'ora.
$testoSorgente = [System.IO.File]::ReadAllText((Resolve-Path $sorgente))
$mancanti = @()
if ($testoSorgente -notmatch 'graph_bufs_moved')      { $mancanti += "graph_bufs_moved in $sorgente" }
if ($testoSorgente -notmatch 'reserve_graph')         { $mancanti += "reserve_graph in $sorgente" }
if (-not (Test-Path "tests\test_graph_reserve_wrappers.py")) { $mancanti += "tests\test_graph_reserve_wrappers.py" }
if ($mancanti.Count -gt 0) {
    $elenco = $mancanti -join ", "
    throw ("questo checkout NON contiene la #40 (manca: $elenco). Lo script " +
           "dimostra quella PR, quindi il working tree deve essere il suo. " +
           "Dalla cartella c\:`n`n" +
           "    git fetch origin claude/graph-buf-gen-invariant`n" +
           "    git checkout claude/graph-buf-gen-invariant`n`n" +
           "poi rilancia. (Lo script di misura non va ricopiato: tools\prova-40.ps1 " +
           "non e' tracciato su quel branch, quindi il checkout non lo tocca.)")
}

try {

"=== FASE 0: lo strumento funziona? ========================================="
"PowerShell $($PSVersionTable.PSVersion)  |  $sorgente su $(Split-Path -Leaf (Get-Location))"
# Collaudo dell'esecutore PRIMA di credergli. La corsa precedente ha visto
# ogni comando restituire codice vuoto e output vuoto, e le fasi hanno
# continuato lo stesso: una ha stampato OK confrontando il vuoto con zero.
# Se l'esecutore non sa eseguire `echo`, niente sotto vale nulla.
$probe = Esegui "echo prova40harnessok"
if ($probe.Codice -ne 0 -or $probe.Testo -notmatch "prova40harnessok") {
    throw ("l'esecutore di comandi non funziona su questa macchina: " +
           "`echo` ha reso codice '$($probe.Codice)' e output '$($probe.Testo)'. " +
           "Nessuna fase sotto sarebbe attendibile, quindi mi fermo qui invece " +
           "di stampare risultati costruiti sul vuoto. Riporta queste due righe.")
}
"  OK   l'esecutore rende codice e output (echo -> $($probe.Codice))"
# E soprattutto: un comando che FALLISCE deve rendere un codice diverso da
# zero. Questo e' il collaudo che mancava, ed e' esattamente il difetto che
# ha reso falsi i risultati del primo giro: lo script leggeva 0 da comandi
# falliti e stampava OK. Senza questa prova, ogni "OK" sotto e' indistinguibile
# da un errore silenzioso.
# `cmd /c exit 3` e NON `exit 3`: quest'ultimo chiuderebbe il batch prima
# dell'epilogo che scrive il codice, e la sonda fallirebbe su uno strumento
# sano. Un processo figlio che esce 3 lascia %ERRORLEVEL% a 3 e prosegue.
$neg = Esegui "cmd /c exit 3"
if ($neg.Codice -ne 3) {
    throw ("l'esecutore non propaga il codice di uscita: `exit 3` ha reso " +
           "'$($neg.Codice)'. Ogni confronto sotto sarebbe privo di significato " +
           "e le fasi stamperebbero OK avendo misurato il numero sbagliato. " +
           "Riporta questa riga.")
}
"  OK   l'esecutore propaga il fallimento (exit 3 -> $($neg.Codice))"

# E gli strumenti ci sono? `python` non e' detto che sia il nome giusto.
$python = $null
foreach ($c in @("python", "python3", "py -3")) {
    $v = Esegui "$c --version"
    if ($v.Codice -eq 0) { $python = $c; "  OK   interprete Python: $c ($($v.Testo.Trim()))"; break }
}
if (-not $python) { throw "nessuno fra python, python3 e py -3 risponde a --version. Senza Python le fasi 1, 2 e 4 non possono girare." }
$mk = Esegui "make --version"
if ($mk.Codice -ne 0) { throw "make non risponde a --version (codice $($mk.Codice)). Serve per le fasi 3 e 4." }
"  OK   make presente"
""

"=== FASE 1: il test Python passa sul file intatto ==========================="
$r = Esegui "$python -m unittest tests.test_graph_reserve_wrappers"
Verifica ($r.Codice -eq 0) "test_graph_reserve_wrappers verde sul sorgente intatto"
if ($r.Codice -ne 0) { Mostra $r }

""
"=== FASE 2: MUTAZIONE A -- reserve nuda su un buffer del grafo =============="
"Sostituisco reserve_graph(ctx,&ctx->gate,...) con reserve(&ctx->gate,...) in"
"expert_group_impl. E' esattamente il difetto che la PR esiste per impedire."
Muta "ob=(size_t)S*proj->O*sizeof(float);if(!reserve_graph(dc,&dc->y,&dc->y_cap,ob))return 0;" `
     "ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;" `
     "A (reserve nuda su dc->y in attention_absorb_batch_run)"
$r = Esegui "$python -m unittest tests.test_graph_reserve_wrappers"
Verifica ($r.Codice -ne 0) "il test diventa rosso sulla reserve nuda"
Verifica ($r.Testo -match 'backend_cuda\.cu:\d+:\s*reserve\(&dc->y') "e NOMINA la riga colpevole"
($r.Testo -split "`n" | Select-String -Pattern 'backend_cuda\.cu:\d+' | Select-Object -First 2) | ForEach-Object { "       $_" }
Ripristina
"  sorgente ripristinato"

""
if (-not $SkipFullCudaTest) {
  "=== FASE 3: make cuda-test sul file intatto (la meta' positiva) ============"
  "Lunga: compila ed esegue tutta la batteria CUDA. -SkipFullCudaTest la salta."
  $r = Esegui "make cuda-test"
  Verifica ($r.Codice -eq 0) "make cuda-test verde"
  if ($r.Codice -ne 0) { Mostra $r 25 }
  else { ($r.Testo -split "`n" | Select-String -Pattern 'grouped-g4' ) | ForEach-Object { "       $_" } }
  ""
}

"=== FASE 4: MUTAZIONE B -- la funzione di invalidazione svuotata ============"
"graph_bufs_moved non incrementa piu' buf_gen e non distrugge piu' i grafi."
"Dopo una crescita dei buffer il grafo vecchio ha ancora la firma buona, quindi"
"viene RIGIOCATO su memoria liberata: e' il bug che la PR chiude."
Muta "ctx->buf_gen++;" `
     "if(1){(void)ctx;return;}  /* MUTAZIONE B: invalidazione svuotata */ ctx->buf_gen++;" `
     "B (graph_bufs_moved svuotata)"

"-- il test Python deve RESTARE VERDE: e' il buco che dichiara di avere"
$r = Esegui "$python -m unittest tests.test_graph_reserve_wrappers"
Verifica ($r.Codice -eq 0) "il test Python NON vede la funzione svuotata (buco dichiarato, confermato)"

"-- il test g4 deve diventare ROSSO: e' l'unico che la vede"
# Anche questa passava per `& cmd /c "...2>&1"`, la forma che non rendeva
# nulla: la fase 4 moriva qui dicendo di non trovare la riga di compilazione,
# quando il problema era che `make -n` non veniva letto affatto.
$dryrun = Esegui "make -n cuda-test"
$riga = ($dryrun.Testo -split "`r?`n") | Where-Object { $_ -match 'grouped_g4_test' -and $_ -match 'test_grouped_g4_cuda\.cu' } | Select-Object -First 1
if (-not $riga) {
    "  la riga di compilazione non e' stata trovata. 'make -n cuda-test' ha reso codice $($dryrun.Codice):"
    Mostra $dryrun 15
    throw "non riesco a ricavare da 'make -n cuda-test' la riga che compila test_grouped_g4_cuda.cu (vedi sopra)."
}
"       compilo con: $riga"
$r = Esegui $riga
Verifica ($r.Codice -eq 0) "il g4 mutato compila"
if ($r.Codice -eq 0) {
    $r = Esegui ".\grouped_g4_test.exe" 420
    Verifica ($r.Codice -ne 0) "il g4 diventa ROSSO con la funzione svuotata"
    Verifica ($r.Testo -match 'did not re-capture') "e il messaggio e' quello giusto (did not re-capture)"
    # SEMPRE, non solo le righe che combaciano. Alla prima esecuzione reale
    # il g4 e' diventato rosso ma il filtro non ha trovato nulla, quindi il
    # MOTIVO del rosso e' rimasto sconosciuto -- e un rosso di cui non si
    # conosce la causa non dimostra niente. Un crash del grafo rigiocato su
    # memoria liberata, per esempio, non stampa "FAIL": muore prima.
    "       --- output del g4 mutato (codice $($r.Codice) = $($r.Esa)) ---"
    if ((-not $r.StdOut) -and $r.StdErr) {
        "       NOTA: stdout vuoto con stderr presente. Puo' indicare una"
        "       terminazione anomala -- stdout verso file e' bufferizzato e si"
        "       perde, stderr no -- il che sarebbe coerente con un grafo"
        "       rigiocato su memoria liberata, che muore prima di stampare"
        "       FAIL. E' un'ipotesi: il codice di uscita qui sopra la conferma"
        "       o la smentisce, e va letto insieme a questa riga."
    }
    Mostra $r 40
}

Ripristina
"  sorgente ripristinato"

""
"=== FASE 5: il g4 torna verde sul sorgente intatto =========================="
# Questa fase e' una CONFERMA, non la prova: la meta' verde e' gia' stabilita
# dalla FASE 3, che ha compilato ed eseguito lo stesso test sul sorgente
# intatto dentro `make cuda-test`. Se qui si blocca o scade, il risultato
# delle fasi 3 e 4 resta valido.
#
# E un blocco qui ha una causa plausibile: la riga di compilazione porta
# -arch=native, che INTERROGA la scheda per ricavarne l'architettura. La
# fase 4 ha appena fatto fallire un test che rigioca un grafo su memoria
# liberata; se quello e' finito con un accesso illegale, il contesto CUDA
# puo' restare in uno stato in cui quella query non ritorna. Alla prima
# esecuzione reale questa fase e' rimasta appesa senza dire niente.
$r = Esegui $riga
Verifica ($r.Codice -eq 0) "il g4 ricompila"
if ($r.Codice -eq 0) {
    $r = Esegui ".\grouped_g4_test.exe" 420
    Verifica ($r.Codice -eq 0) "il g4 e' di nuovo VERDE una volta ripristinato"
    Mostra $r 20
}

} finally {
    Ripristina
}

""
"============================================================================"
if ($fallite.Count -eq 0) {
    "TUTTO VERDE. Le due guardie sono state viste passare E viste fallire."
    "Il buco dichiarato dal test Python e' reale: la FASE 4 lo ha riprodotto."
} else {
    "PROVA INCOMPLETA. Punti falliti:"
    $fallite | ForEach-Object { "  - $_" }
    "Le fasi 3 e 4 sono quelle che dimostrano: il test g4 verde sul sorgente"
    "intatto e ROSSO con l'invalidazione svuotata. Se entrambe sono OK sopra,"
    "l'asimmetria e' stabilita e un fallimento della fase 5 la conferma solo."
    "Se invece e' rossa la 3 o la 4, la #40 non e' provata."
    exit 1
}
