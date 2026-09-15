# Istruzioni per il mio PC

Note di configurazione di questo fork per la macchina su cui gira Colibri.
Non sono documentazione del progetto originale: i riferimenti `file:riga` puntano
al codice di questo fork e vanno ricontrollati dopo ogni aggiornamento da upstream.

## Hardware

| Componente | Valore |
|---|---|
| CPU | AMD Ryzen 9 7950X, 16 core / 32 thread, AVX-512 (VNNI, BF16) |
| RAM | 32 GB DDR5 (kit Corsair CMK32GX5M2E6000Z36, 2×16 GB) |
| Scheda madre | ASUS TUF GAMING X670E-PLUS (4 slot) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER 16 GB (Ada, `sm_89`) |
| Disco | NVMe PCIe 4 TB |
| Sistema | Windows 11, build MinGW-w64 (MSYS2 UCRT64) + DLL CUDA con MSVC |

## 1. Cose da fare subito

| Cosa | Perché | Riferimento |
|---|---|---|
| Compilare con `ARCH=native` (e `CUDA_ARCH=sm_89` per la DLL) | Su Windows il default è `x86-64-v3`, solo AVX2: i kernel int8/int4 AVX-512/VNNI del 7950X vengono esclusi. Non usare i binari già pronti. | `c/Makefile:105` |
| Impostare sempre `COLI_GPU=0` | Senza, su Windows la GPU poteva non essere trovata e il motore restava sulla CPU senza avvisi. Corretto da #1542 (incluso in questo fork), ma impostarla esplicitamente resta la scelta più sicura. | JustVugg/colibri#1542 |
| `OMP_WAIT_POLICY=active` e `GOMP_SPINCOUNT=200000` **solo senza GPU** | Tengono "caldi" i thread OpenMP tra le tante piccole regioni parallele per esperto. Da questo fork `coli` le imposta su Windows per i motori che girano solo su CPU. **Con CUDA no**: l'attesa attiva compete con la GPU e ha dato forti peggioramenti misurati; con la GPU si prova solo misurando con e senza. | `c/coli` (`env_for_engine`), `docs/tuning.md` (Hybrid CUDA/CPU) |
| Attivare EXPO nel BIOS | La RAM oggi lavora a 4800 MT/s invece di 6000: +25% di banda di memoria. BIOS (Canc) → Ai Tweaker → Ai Overclock Tuner → EXPO I. Se il PC è instabile tornare su Auto o provare 5600. | — |
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

## 5. RAM: piano di aggiornamento

- Il salto utile è da 32 a 64–96 GB (GLM-5.2 e DeepSeek V4 trovano molti più esperti in RAM).
- Scelta consigliata: **2×48 GB DDR5-6000 EXPO**, sostituendo il kit attuale (con 4 banchi DDR5 la velocità scende spesso a 4800–5200).
- Prima di comprare: aggiornare il BIOS e controllare la QVL sulla pagina di supporto ASUS della TUF GAMING X670E-PLUS.

## 6. Miglioramenti aperti (da misurare su questo PC)

| Tema | Stato |
|---|---|
| I/O parallelo su Windows: un solo handle sincrono per file, Windows serializza le `ReadFile` (vedi commento in `c/iobench.c`) | in lavorazione in una PR separata del fork |
| RAM di Qwen3.6: quantizzazione incrementale al caricamento (idea di #1218) + embedding in int8 | da fare |
| Copie CPU↔GPU sincrone della parte densa in VRAM (`c/backend_cuda.cu`, `coli_cuda_matmul`) | da fare, collegato a #431 |
| Due motori (draft esterno) per Qwen3.6: serve un modello piccolo con lo stesso vocabolario da 248.320 token | da valutare, issue #494 |
