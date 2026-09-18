# Istruzioni per il mio PC

Note di configurazione di questo fork per la macchina su cui gira Colibri.
Non sono documentazione del progetto originale: i riferimenti `file:riga` puntano
al codice di questo fork e vanno ricontrollati dopo ogni aggiornamento da upstream.

## Hardware

| Componente | Valore |
|---|---|
| CPU | AMD Ryzen 9 7950X, 16 core / 32 thread, AVX-512 (VNNI, BF16) |
| RAM | **64 GB DDR5 a 5200 MT/s**: 2×16 Corsair CMK32GX5M2E6000Z36 + 2×16 CMG32GX5M2E6000Z36 (4 banchi: con EXPO restano a 5200, non 6000) |
| Scheda madre | ASUS TUF GAMING X670E-PLUS, BIOS **3881** (4 slot RAM, 4 slot M.2) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER 16 GB (Ada, `sm_89`), driver 616.64 |
| Disco | Silicon Power UD90 4 TB in **M.2_3** → `Gen4 x4`. In M.2_2 girava a `Gen3 x2`, cioè un quarto della banda |
| In arrivo | Crucial T705 4 TB PCIe 5.0 in M.2_1, dedicato ai modelli |
| Sistema | Windows 11, build MinGW-w64 (MSYS2 UCRT64) + DLL CUDA con MSVC 2022 e CUDA 13.4 |

## Misure (17 settembre 2026, Qwen3.6-35B-A3B int4 gs64)

| | 32 GB a 4800, SSD Gen3 x2 | **64 GB a 5200, SSD Gen4 x4** |
|---|---|---|
| Generazione (GPU, cache calda) | 12,1 tok/s (82,9 ms/token) | **17,7 tok/s (56,4 ms/token)** |
| Prefill, 56 token | ~4,2 s | **~2,3 s** |
| Picco RAM | 25,3 GB | 30,6 GB |
| Disco, blocchi da 19 MB, 16 thread | 1,6 GB/s | **5–6 GB/s** |
| Qwen3.6 su CPU | 1,00 tok/s | 1,24 tok/s |

Dove vanno i 56 ms per token (cache calda): DeltaNet 27,1 (proiezioni su GPU 9,8;
normalizzazione e uscita su CPU 9,0), MoE 15,0 (shared expert su CPU 9,3),
attention su CPU 10,5, lm_head su GPU 3,2. **Circa 34 ms su 56 sono ancora su CPU.**

## 1. Cose da fare subito

| Cosa | Perché | Riferimento |
|---|---|---|
| Compilare con `ARCH=native` (e `CUDA_ARCH=sm_89` per la DLL) | Su Windows il default è `x86-64-v3`, solo AVX2: i kernel int8/int4 AVX-512/VNNI del 7950X vengono esclusi. Non usare i binari già pronti. | `c/Makefile:105` |
| Impostare sempre `COLI_GPU=0` | Senza, su Windows la GPU poteva non essere trovata e il motore restava sulla CPU senza avvisi. Corretto da #1542 (incluso in questo fork), ma impostarla esplicitamente resta la scelta più sicura. | JustVugg/colibri#1542 |
| `OMP_WAIT_POLICY=active` e `GOMP_SPINCOUNT=200000` **solo senza GPU** | Tengono "caldi" i thread OpenMP tra le tante piccole regioni parallele per esperto. Da questo fork `coli` le imposta su Windows per i motori che girano solo su CPU. **Con CUDA no**: l'attesa attiva compete con la GPU e ha dato forti peggioramenti misurati; con la GPU si prova solo misurando con e senza. | `c/coli` (`env_for_engine`), `docs/tuning.md` (Hybrid CUDA/CPU) |
| Con 4 banchi, EXPO più `DRAM Frequency = DDR5-5200` | Il solo EXPO lascia la RAM a 3600 MT/s. Impostata a mano regge 5200; con 2 banchi si arrivava a 6000. BIOS (Canc) → Ai Tweaker. | — |
| `--auto-tier` accende il tier VRAM anche sui motori non-GLM | Prima lo lasciava spento in silenzio: chi l'ha segnalato è passato da 11,8 a 21 tok/s. | JustVugg/colibri#1582 |
| `RAM_GB=<n>` se vuoi un tetto alla RAM | Senza, il warmstart carica **tutti** gli esperti in RAM: il picco misurato qui è 30,6 GB. Con `RAM_GB` impostata la cache viene limitata. | cherry-pick di JustVugg/colibri#1564 |
| Non usare `DSV4_CUDA_TC=1` | Usa l'FP8 a microscaling delle Blackwell (RTX 50): sulla 4070 Ti SUPER fallisce a ogni chiamata e rimanda il lavoro alla CPU. | `c/backend_cuda_dsv4.cu:1704` |

## 2. Compilazione

Da un prompt `cmd` (non PowerShell), dopo `vcvars64.bat` di Visual Studio 2022 e con
MSYS2 nel `PATH` (dettagli in `docs/windows.md`), dentro la cartella `c\`:

```bat
set PATH=%PATH%;C:\msys64\usr\bin
make cuda-dll CUDA_ARCH=sm_89
make colibri.exe CUDA_DLL=1 ARCH=native
make qwen36.exe CUDA_DLL=1 ARCH=native
make qwen38.exe CUDA_DLL=1 ARCH=native
make iobench.exe
```

Controllo: all'avvio `colibri.exe` (motore GLM) stampa `idot: <kernel>` nel banner
(`c/colibri.c`, riga `== GLM C engine`). Se il kernel è solo AVX2, la build non ha usato `ARCH=native`.

## 3. Impostazioni per modello

### Qwen3.6-35B-A3B (container int4, ~20 GB) — il più adatto a questa GPU

Tramite `coli` bastano `COLI_GPU=0` e le impostazioni del tier. Lancio diretto:

```bat
set COLI_CUDA=1
set COLI_GPU=0
set CUDA_EXPERT_GB=auto
set COLI_PLACE=auto
set HEAT_FILE=heat.bin
set OMP_NUM_THREADS=16
set SNAP=D:\modelli\qwen36_int4
set N_NEW=200
qwen36.exe 256 4 prompt.txt
```

- `HEAT_FILE` salva gli esperti più usati: dalla seconda esecuzione la VRAM parte già riempita bene.
- `OMP_WAIT_POLICY=active` qui non va messo di default (vedi sezione 1): provarlo solo con una misura A/B.
- Picco di RAM documentato ~29 GB (misurato con due GPU da 8 GB): con 32 GB chiudere i programmi pesanti.
- Riferimento: `docs/qwen36-cuda-tier.md`.

### DeepSeek V4 Flash (~167 GB)

`CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1` (riga "16 GB" della tabella in
`docs/deepseek-v4.md`). Senza DeepGEMM (solo RTX 50) il prefill è più lento dei numeri del documento.

### Qwen3.8-Flash-Next (~185 GB)

`COLI_GPU=0 CUDA_EXPERT_GB=auto CUDA_DENSE=1` (tier CUDA di #1424). Disk-bound: meno di 1 tok/s.

### GLM-5.2 / 5.3 (~195–372 GB)

La GPU conta poco; contano RAM e disco. `DIRECT=1` (già default di `coli` su Windows),
da provare `COLI_CUDA_MTP=1` su GLM-5.2. Con 32 GB aspettarsi circa 0,1–0,3 tok/s.

## 4. Misurare prima di cambiare

1. Velocità del disco: `iobench.exe <file-shard-grande> 64 64 16 1` (confrontare `direct` 0 e 1).
2. Collo di bottiglia: `PROF=1 coli run --model <dir> "prompt" --ngen 64` → riga finale `[PROF] verdict`.
3. Tuning automatico: `coli tune --model <dir>` (salva il profilo migliore).
4. Scaldare la cache (PowerShell, dalla cartella `c\`): `.\warmup.ps1 -Model <dir> -Rounds 10 -Ngen 32`.

### A/B dell'I/O parallelo su Windows

Dopo aver ricompilato `iobench.exe` e il motore con questo fork, da `cmd` (sostituire il file con
uno shard grande del modello, meglio se non letto di recente):

```bat
rem 1) come il vecchio motore: un fd condiviso, handle sincrono
set IOBENCH_SHARED=1
set COLI_WIN_SYNC_DIRECT=1
iobench.exe D:\modelli\<shard>.safetensors 19 256 16 1
rem 2) come il motore nuovo: un fd condiviso, handle OVERLAPPED
set COLI_WIN_SYNC_DIRECT=
iobench.exe D:\modelli\<shard>.safetensors 19 256 16 1
rem 3) riferimento: un fd per thread
set IOBENCH_SHARED=
iobench.exe D:\modelli\<shard>.safetensors 19 256 16 1
```

Atteso: la 1) molto più lenta, la 2) vicina alla 3). Se è così, ripetere sul motore con un modello
grande (`DIRECT=1`), confrontando `COLI_WIN_SYNC_DIRECT=1` e senza, a parità di prompt e cache
(stesso ordine alternato, almeno 3 giri): tok/s e tempo disco nella riga `[PROF]`.

## 5. Aggiornamenti hardware

**Fatti:** BIOS 1813 → 3881; RAM 32 → 64 GB a 5200 (4 banchi, due kit diversi: EXPO da solo
tornava a 3600, la frequenza va messa a mano); SSD da M.2_2 (`Gen3 x2`) a M.2_3 (`Gen4 x4`),
cioè da 1,6 a 5–6 GB/s.

**Da fare:**
- **MemTest86**, almeno un giro completo: 4 banchi di due kit diversi, entro il periodo di reso.
- **Crucial T705 4 TB in M.2_1** (PCIe 5.0, unico slot Gen5, con dissipatore della scheda madre):
  solo per i modelli. Serve per GLM-5.2, DeepSeek V4 e Qwen3.8, che leggono gli esperti dal disco.
- **Oltre i 64 GB**: ha senso solo per i modelli grandi, e ai prezzi attuali della DDR5 conviene
  aspettare. Con 4 banchi non si va oltre ~5200.

## 6. Miglioramenti aperti (da misurare su questo PC)

| Tema | Stato |
|---|---|
| I/O parallelo su Windows (handle diretto `OVERLAPPED`) | **fatto e misurato.** Con l'SSD a Gen3 x2 valeva +8%; ora che il disco fa 5–6 GB/s le tre configurazioni si equivalgono. Tornerà utile col T705 |
| RAM di Qwen3.6: quantizzazione al caricamento + embedding int8 | **fatto.** RSS dopo il caricamento 9,23 → 4,82 GB. Il picco complessivo resta ~30 GB, perché arriva dal warmstart degli esperti: per quello serve `RAM_GB` |
| **Kernel int8 AVX-512** | **fatto, da misurare.** I kernel caldi (proiezioni dense e esperti) avevano solo la versione AVX2: metà registro e metà FMA sul 7950X. `QWEN36_AVX512=0` torna ad AVX2 per il confronto |
| **Uscita DeltaNet e shared expert sulla GPU** | da fare, ma **dopo** aver ridotto le chiamate GPU: da sole aggiungerebbero ~30 viaggi sincroni per token, più o meno quanto risparmiano |
| Copia int8 degli esperti non residenti in VRAM (~9 GB di RAM) e riserva KV contata su 40 layer invece di 10 | da fare |
| Copie CPU↔GPU sincrone della parte densa (`c/backend_cuda.cu`, `coli_cuda_matmul`) | da fare, collegato a #431. Con 64 GB l'attesa GPU è scesa da 11,6 a 2,5 ms/token, quindi ora vale meno di prima |
| Testa MTP di Qwen3.6 (il "secondo motore" già dentro il checkpoint) | da valutare. Il convertitore la salta apposta (#1326); servono conversione, caricamento e salvataggio dello stato DeltaNet |
| Due motori con modello esterno | da valutare, issue #494: serve un modello piccolo con lo stesso vocabolario da 248.320 token |
