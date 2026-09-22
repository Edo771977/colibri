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

function Esegui([string]$Comando) {
    # Il comando viene SCRITTO IN UN .cmd ed eseguito, invece di essere
    # passato a `cmd /c "..."`. Due ragioni, entrambe gia' costate tempo:
    # catturare la pipeline restituiva output vuoto proprio quando il
    # comando falliva, cioe' quando serve leggerlo; e annidare le virgolette
    # (`cmd /c "x > ""y"" 2>&1"`) mette PowerShell, cmd e il comando stesso
    # a litigare sul riquotaggio. Un file non ha nessuno dei due problemi e
    # regge qualunque riga di comando, comprese quelle che nvcc produce.
    $base = [System.IO.Path]::GetTempFileName()
    $bat  = "$base.cmd"
    $log  = "$base.log"
    Set-Content -LiteralPath $bat -Encoding ASCII -Value @(
        "@echo off",
        "$Comando > ""$log"" 2>&1",
        "exit /b %ERRORLEVEL%")
    & cmd /c $bat
    $codice = $LASTEXITCODE
    $testo = if (Test-Path $log) { Get-Content -Raw -LiteralPath $log } else { "" }
    foreach ($f in @($base, $bat, $log)) {
        Remove-Item -Force -LiteralPath $f -ErrorAction SilentlyContinue
    }
    return @{ Testo = [string]$testo; Codice = $codice }
}

# Stampa le prime righe dell'output di un comando fallito. Una fase rossa
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

"=== FASE 1: il test Python passa sul file intatto ==========================="
$r = Esegui "python -m unittest tests.test_graph_reserve_wrappers"
Verifica ($r.Codice -eq 0) "test_graph_reserve_wrappers verde sul sorgente intatto"
if ($r.Codice -ne 0) { Mostra $r }

""
"=== FASE 2: MUTAZIONE A -- reserve nuda su un buffer del grafo =============="
"Sostituisco reserve_graph(ctx,&ctx->gate,...) con reserve(&ctx->gate,...) in"
"expert_group_impl. E' esattamente il difetto che la PR esiste per impedire."
Muta "ob=(size_t)S*proj->O*sizeof(float);if(!reserve_graph(dc,&dc->y,&dc->y_cap,ob))return 0;" `
     "ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;" `
     "A (reserve nuda su dc->y in attention_absorb_batch_run)"
$r = Esegui "python -m unittest tests.test_graph_reserve_wrappers"
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
$r = Esegui "python -m unittest tests.test_graph_reserve_wrappers"
Verifica ($r.Codice -eq 0) "il test Python NON vede la funzione svuotata (buco dichiarato, confermato)"

"-- il test g4 deve diventare ROSSO: e' l'unico che la vede"
$riga = (& cmd /c "make -n cuda-test 2>&1") | Where-Object { $_ -match 'grouped_g4_test' -and $_ -match 'test_grouped_g4_cuda\.cu' } | Select-Object -First 1
if (-not $riga) { throw "non riesco a ricavare da 'make -n cuda-test' la riga che compila test_grouped_g4_cuda.cu. Compila a mano e rilancia con -SkipFullCudaTest." }
"       compilo con: $riga"
$r = Esegui $riga
Verifica ($r.Codice -eq 0) "il g4 mutato compila"
if ($r.Codice -eq 0) {
    $r = Esegui ".\grouped_g4_test.exe"
    Verifica ($r.Codice -ne 0) "il g4 diventa ROSSO con la funzione svuotata"
    Verifica ($r.Testo -match 'did not re-capture') "e il messaggio e' quello giusto (did not re-capture)"
    ($r.Testo -split "`n" | Select-String -Pattern 'FAIL|re-capture|REPLAYED') | ForEach-Object { "       $_" }
}

Ripristina
"  sorgente ripristinato"

""
"=== FASE 5: il g4 torna verde sul sorgente intatto =========================="
$r = Esegui $riga
Verifica ($r.Codice -eq 0) "il g4 ricompila"
if ($r.Codice -eq 0) {
    $r = Esegui ".\grouped_g4_test.exe"
    Verifica ($r.Codice -eq 0) "il g4 e' di nuovo VERDE una volta ripristinato"
    ($r.Testo -split "`n" | Select-String -Pattern 'grouped-g4|^OK') | ForEach-Object { "       $_" }
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
    "NON PROVATO. Punti falliti:"
    $fallite | ForEach-Object { "  - $_" }
    "Nessuna conclusione: finche' una di queste e' rossa la #40 non e' provata."
    exit 1
}
