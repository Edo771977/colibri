# qwen36: CUDA VRAM expert tier

Applies colibri's placement concept ("route -> place -> overlap -> learn") to
Qwen3.6-35B-A3B one level up from the GLM disk tier: all 10,240 experts live
in RAM, the **hot** ones are promoted into DEVICE_LOCAL VRAM across one or
more GPUs and computed there through the existing shared CUDA backend
(`backend_cuda.cu` expert-group API — no new backend).

> **Two engines.** The tier also serves Qwen3.8-Flash-Next (`c/qwen38.c`,
> [qwen38.md](qwen38.md#gpu-cuda-vram-expert-tier)) in its *fp8 streaming
> mode* (`qt_init_fp8`): experts do not all live in RAM there, `cap` may be
> smaller than the expert count, the tier copies each expert's e4m3 slab and
> block scales when the engine reports it and keeps no pointer into the
> engine's slot, and promotion happens at report time instead of a warmstart.
> Everything below describes the Qwen3.6 modes unless it says otherwise.


## How it works

- **Home device:** expert `eid` lives on GPU `eid % n_gpus`; no duplicates.
- **Placement:** routing heat decides who earns VRAM (LFRU semantics from
  `tier.h`, 25%+4 hysteresis). Runtime heat is halved every 1024 decode ticks,
  so a long-lived process can replace experts from an old workload instead of
  permanently freezing its initial hot set. A parallel **warmstart** fills the per-device
  budget before the first token — ordered by a persisted heat table
  (`HEAT_FILE`) when present, so a second run starts fully placed.
- **Decode:** per (token, layer) the resident experts are issued as async
  groups on all devices (`coli_cuda_expert_group_issue/take`); VRAM misses
  fall back to the CPU int8 path and overlap with the in-flight groups, as
  does the shared expert. Placement never changes routing or precision.
- **Memory:** on an **int4 container** the warmstart frees the RAM int8 copies
  of VRAM-resident experts (rematerialized from the packed int4 copy on LFRU
  eviction; no container access). Peak RSS for the 35B int4 container: ~29 GB
  with two 8 GB GPUs. The RSS saving is a property of packed containers only:
  on an **int8 container** there is no second copy to rematerialize from, so
  since #1341 nothing is freed and every resident keeps its full RAM weights —
  such a run gets the tier's VRAM speed at full residency cost, and the 29 GB
  figure below does not apply to it.

## Usage

```bash
make -C c qwen36 CUDA=1 CUDA_ARCH=native   # NVCC=/usr/bin/nvcc on distro CUDA
COLI_CUDA=1 COLI_GPUS=0,1 HEAT_FILE=heat.bin CUDA_EXPERT_GB=auto \
OMP_NUM_THREADS=<physical cores> OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close \
SNAP=<container> N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

### Windows (CUDA_DLL=1)

MinGW cannot link CUDA directly, so the backend is built into `coli_cuda.dll`
with nvcc + MSVC and `qwen36.exe` reaches it through `backend_loader.c`.
`CUDA=1` is rejected on Windows by design. From an *x64 Native Tools* prompt
with MSYS2's `mingw64\bin` and `usr\bin` on `PATH`:

```cmd
cd c
make cuda-dll CUDA_ARCH=sm_89
make qwen36.exe CUDA_DLL=1 ARCH=native
set COLI_CUDA=1
set COLI_GPUS=0
set CUDA_EXPERT_GB=auto
qwen36.exe <same arguments as the CPU build>
```

**Build the engine with clang, not gcc.** Measured on this host, same commit,
same container, same placement: **57.70 ms/token under MinGW libgomp against
32.73 under LLVM libomp** — 1.8×, and it is the OpenMP runtime, not the
compiler version (`qwen36-toolchain-2x-2026-09-19-raw.txt`). `deltanet`,
`shared` and `router` all roughly halve. The engine builds and answers
correctly either way, which is exactly why this is easy to lose:

```cmd
pacman -S mingw-w64-clang-x86_64-clang mingw-w64-clang-x86_64-openmp
make qwen36.exe CC=C:/msys64/clang64/bin/clang.exe CUDA_DLL=1 ARCH=native
```

The resulting `qwen36.exe` links `libomp.dll` dynamically, so `clang64\bin`
must be reachable at run time (on `PATH`, or copy the DLL next to the exe).
`make clean` deletes `coli_cuda.dll` (`tools/clean.py`); use `make -B` when
rebuilding the engine with a different compiler, or rebuild the DLL after.

Keep `coli_cuda.dll` next to `qwen36.exe`, built from the same checkout, and
the CUDA toolkit's `bin` directory on `PATH` for `cudart`. A startup line
`[gpu] MoE experts -> CUDA VRAM tier` confirms the tier is active; without it
the run is CPU-only.

`cap` (argv[1]) must equal `n_experts` (full RAM residency). int4 containers
only (the int8 container keeps the CPU path). `COLI_TIMERS=1` prints
per-phase timings and tier telemetry.

## Placement: where the dense trunk goes (`COLI_PLACE`)

The tier moves the routed experts. On this hybrid model that is the *small*
part of a token: 40 layers × top-8 experts, ~190 MB of int8 per token at an
81 % hit rate, against a dense trunk -- attention, DeltaNet projections,
shared expert, lm_head -- of **1.8 GB of int8 read on every token**. Leaving
the trunk on the CPU is why a 6 GB card sees the hit rate stop mattering
(#1040): the GPU is doing the cheap job.

By default the engine now places the trunk itself. Before the tier decides its
budget, the engine offers each trunk component with its size (lm_head once, the
fused DeltaNet projection of every DeltaNet layer), and the tier prices them
against the experts they would displace, in **bytes saved on the memory bus per
token, per byte of VRAM**:

- a dense component is read every token: 1.0 per byte;
- a routed expert is read with the probability a token routes to it -- its heat
  share when `HEAT_FILE` exists, `topk / n_experts` otherwise -- and the CPU
  fallback reads the int8 slot, twice the VRAM bytes of an int4 expert: 2·p per
  byte.

A component goes to the device with the most room if its value beats that of
the coldest experts it pushes out. Without heat that tail is worth 0.06 per byte
on the 35B and the trunk always wins; with heat, a card whose marginal expert is
routed on more than every second token keeps its experts. Placed bytes come out
of that device's expert budget, and each decision prints as a `[place]` line.

| `COLI_PLACE` | behaviour |
|---|---|
| unset or `auto` | automatic, as above |
| `off` | nothing placed: experts only (the behaviour before this) |
| `lmhead=0,dnproj=0:20+1:20,experts=0` | hand-written list (the measurement tool); obeyed as written, trunk bytes still charged to the budget |
| `lmhead=0,dnproj=0:20+1:20,dnout=0,attnout=0,experts=0` | the same, now including the two **output** projections |

### The output projections (`dnout`, `attnout`)

Of the 1.8 GB the trunk reads per token, `auto` used to move `lm_head` and the
fused DeltaNet input projection only — 1.17 GB on the calibration above. The
rest is attention, the shared expert, the router and the two out_proj
matrices, and none of it had a knob: `qt_place_of()` has always answered for
any name, but the engine asked about `lmhead` and `dnproj` only, so
`COLI_PLACE="dnout=0"` parsed, stored a device, and moved nothing.

`dnout` (DeltaNet `out_proj`) and `attnout` (attention `o_proj`) are now
offered to the placer like everything else, so `auto` takes them when they
beat the experts they displace, and their bytes come out of that device's
expert budget. `COLI_PLACE="off"` is the way back to experts only.

**What the measurement below does not cover**: it is a 16 GB card with a heat
table warmed on the prompt, i.e. the favourable end. On a small card with no
`HEAT_FILE`, `auto_displaced_value` prices every expert at the flat
`topk / n_experts` (0.06 per byte on the 35B) against the trunk's 1.0, so the
trunk wins under pressure and now takes 0.31 GB more than it did. The `off`
/ `auto` calibration further down is a precedent for `lm_head` and `dnproj`
holding up on an 8 GB card, not a measurement for these two.

They landed explicit-only first, because nothing had measured whether a placed
matrix pays for the driver round-trip it adds to the serial layer chain. Then
it was measured.

#### Measurement, 19 September 2026

RTX 4070 Ti SUPER (16 GB) + Ryzen 9 7950X, 64 GB at 5200 MT/s,
Qwen3.6-35B-A3B int4 gs64, 25-token prompt, 128 generated, **four runs a
side, alternated A B A B**, heat table warmed on this prompt and then frozen
so every run starts from the same residency. Both arms at **100 % VRAM hit
rate, `miss(CPU)` 0**. Both arms on a **gcc/MinGW-libgomp binary** — see the
note on the absolute level below, which is why every ms here is about 1.8×
what the same code does under clang.

```
A  COLI_PLACE="experts=0,lmhead=0,dnproj=0"                     (what auto did before)
B  COLI_PLACE="experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0"
```

>  **RETRACTED AS A CITABLE FIGURE, 22 September 2026.** The −7.77 below is
>  the number every later record in this directory cites as the standard of
>  proof — and it is the worst-documented number of the set. **There is no raw
>  record for it**: no file in `docs/experiments/`, no per-run data, no
>  manifest, only this table. The build is an *inference from the timings*
>  (see "Inferred from the timings" below), not a recorded command. It has
>  been superseded twice on the build that actually ships.
>
>  **Cite −5.05 ms/token (−15.3 %) instead** — clang, both arms quiet,
>  per-run data in `qwen36-place-clang-clean-2026-09-20-raw.txt`. The table
>  below is kept as the historical record of a measurement whose *direction*
>  survives and whose *magnitude* does not.

| ms/token | A (min–max) | B (min–max) | Δ |
|---|---:|---:|---:|
| **`step() total`** | **59.75** (58.7–60.8) | **51.98** (51.4–53.0) | **−7.77** |
| `norm+out` → `dnout` | 9.45 | 5.43 | −4.02 |
| `dn-sub proj` | 11.95 | 10.75 | −1.20 |
| attention → `attnout` | 11.02 | 10.09 | −0.93 |
| lm_head | 4.18 | 3.50 | −0.68 |
| take | 0.93 | 0.36 | −0.57 |
| issue | 2.15 | 1.83 | −0.32 |
| shared / router / cpu-miss | 10.18 / 2.88 / 10.19 | 10.09 / 2.89 / 10.10 | ≈0 |

**16.74 → 19.24 tok/s**: −13.0 % of the time per token, which is +14.9 % of
throughput — **on a build nobody should ship** (59.75 ms/token; see below).
THOSE TWO PERCENTAGES ARE THE WITHDRAWN MAGNITUDE RESTATED: −13.0 % is
7.77/59.75 and +14.9 % is 59.75/51.98, i.e. the same 59.75 → 51.98 pair the
box above withdraws. They carry the withdrawal with them and must not be
cited as an effect size either. An earlier version of this page withdrew the
ms figure and left its percentage form standing as a live comparator, which
is the citation the withdrawal forbids, performed on the same page that
forbids it. Repaired 22 September 2026.
On clang the same change is −15.3 %. Both percentages come from
`step() total`, i.e. decode only; a percentage computed from the wall-clock
`Speed:` line is not comparable to either, because that one carries the
prefill. For 0.31 GB of VRAM and 177 experts (7136 → 6959), hit rate
unchanged at 100 %. This paragraph used to size the effect as "7.77 ms
against a worst-case within-arm spread of 2.1 ms": the 2.1 is right (60.8 −
58.7), the 7.77 is the figure withdrawn above, and quoting it as an effect
size is the same citation the withdrawal forbids. On the build that ships the
effect is **−5.05** against that spread.

The table is **not a partition and does not sum to the total**: `norm+out` and
`dn-sub proj` are sub-timers inside `deltanet`, and `shared`, `router`,
`issue` and `take` sit inside `moe_total`, which is not listed. Only
`step() total` is the whole token.

The rows that should not move do not: `shared` −0.09, `router` +0.01,
`cpu-miss` −0.09. That is the strongest part of the result — the saving is
where the change is and nowhere else.

**Two thirds of it are the bytes; one third is not.** The matrices are
`dn_out` 2048×4096 over 30 layers plus `o_proj` 2048×4096 over 10, i.e.
335.5 MB of int8 weights read per token — what the engine's `[place]` line
counts as `0.31 GB`; the offer charges 336.2 MB because it includes the
per-row scales — worth 6.4 ms of CPU bus at the 52.5 GB/s this machine was
actually measured at (see the note on 55.03 GB/s below). The direct rows
account for −4.95, the shortfall being GPU time that is not free. But
`dn-sub proj`, `lm_head`, `take` and `issue` improve by a further −2.77 ms
while **already residing on the GPU in both arms**. The likely mechanism is
contention: 335 MB/token less CPU traffic leaves the staging copies and
driver calls of the already-placed matrices more bus to work with. That is a hypothesis, not a measurement.

**The absolute level: this arm was a gcc build.** Arm A reads 59.75 ms/token
where an `auto` run on 18 September read 32.1, both at 100 % hit rate, and the
gap was left open here as "not established", with the prompt (25 tokens vs 56)
as the leading suspect. It was measured: a 2×2 of {gcc, clang} × {25, 56
tokens}, one session, four runs an arm, same commit, same placement, same
frozen residency — `qwen36-toolchain-2x-2026-09-19-raw.txt`.

```
toolchain (gcc − clang, 25-token prompt):  −24.98 ms/token   [spread max 2.80]
prompt    (56 − 25 tokens, clang):          −0.13 ms/token
```

The prompt does nothing. The toolchain is nine times the noise, and the clang
arm reads 32.73 — the 18 September number, on the 25-token prompt. Row for row
the two arms land on the two bands the repo's own OpenMP record already
measured (`qwen36-openmp-runtime-2026-09-18.md`): `router` 0.61 against 2.74,
`shared` 3.94 against 9.49. So the 2× is **MinGW libgomp against LLVM libomp**,
not the prompt and not the `trunk_out_matmul` branches.

Inferred from the timings, not from a recorded build command: the 19 September
session did not log which compiler produced its binary. Every row of arm A
sits on the gcc band and none on the clang band, which is as close to a
recorded fact as this gets.

**What survives and what does not.** The A/B is a within-session ratio — same
binary in both arms, alternated, frozen residency, spread 2.1 ms — so `dnout`
and `attnout` are still worth placing. What does not survive is the absolute
level: 59.75 ms/token is a build nobody should ship. The prediction that
followed — that −7.77 ms would shrink on clang, because placement pays by
removing CPU reads and libgomp makes those reads cost twice what libomp
charges — was then measured, and was **wrong**. See below.

>  **The 55.03 GB/s has no source, 22 September 2026.** It appears three
>  times on this page, twice to turn bytes into milliseconds to four
>  significant figures and once to defend those conversions, and **no
>  measurement anywhere in this repository produces it**. What was measured
>  on this machine is **52.2 / 52.5 GB/s** streaming read
>  (`docs/qwen36.md:138`, `docs/experiments/qwen36-openmp-runtime.json:10`,
>  from `c/tests/bench_qwen36_decode_omp`). Until someone produces the run,
>  the byte-to-time arithmetic on this page should be redone at that rate:
>  335.5 MB is **6.4 ms**, not the 6.1 this page printed, and 189 MB is
>  **3.6 ms**, not 3.4. The caveat that was already here stands on top of
>  that: whatever the number is, it is this machine under libgomp, not this
>  machine.
>
>  There is one **55,03** in the records, and it is not a bandwidth:
>  `qwen36-toolchain-2x-2026-09-19-raw.txt:56` reads
>  `step  …  G56  55,03 (53,1-56,5)`, i.e. **ms/token** on the gcc arm of the
>  toolchain 2×2. A comma decimal separator, a different unit and a different
>  quantity. That is the most likely origin, and it is a warning about how
>  this page has been assembled, not a source.

#### Re-measured on clang — `qwen36-place-clang-2026-09-19-raw.txt`

Same 2×2 discipline, {gcc, clang} × {A, B}, arms alternated inside every
repetition so both deltas are born in one session rather than compared across
days. Unlike 19 September this session settles at a **98.1–98.2 % hit rate**
rather than 100 %, so arm B pays the expert-displacement cost a fully resident
session hides — which makes this the more honest number of the two.

```
clang   A 33.43 (33.1-34.3)   B 28.71 (28.2-30.2)   paired median  −4.90
gcc     A see below           B see below           paired median  −5.20
```

**The gcc arm is not measurable by its mean here.** Four of eight repetitions
are disturbed, and in two of them the B arm is *slower* than A — while clang,
running minutes apart inside the same repetition, stays inside 1.2 ms across
all eight. That binary is unstable on this host, so more repetitions do not
average it away. What works is the **paired** delta: `gA` and `gB` run back to
back, so their difference absorbs whatever both are suffering.

**The absolute gain shrinks modestly on clang, and the relative gain grows.**
A later session with *both* arms quiet — 6 % spread on gcc, 1 % on clang, no
arm flagged — prices them properly
(`qwen36-place-clang-clean-2026-09-20-raw.txt`):

```
clang  paired deltas  −5.10 −5.00 −5.00 −5.30    median −5.05   (−15.3 % of 33.05)
gcc    paired deltas  −6.20 −6.70 −5.60 −6.00    median −6.10   (−11.0 % of 55.60)
```

The two sets do not overlap, so the difference is real: clang keeps **83 %** of
gcc's absolute saving while that saving is worth **half again as much** of its
token. The −5.20 above was biased low by the disturbed arm; clang has held at
−4.97, −4.71, −4.90 and −5.05 across four sessions. So the original prediction
was right about the direction and wrong about the size — the gain does not halve
with the CPU read cost, it gives up about a sixth.

One row carries most of the reason (clean session):

```
norm+out -> dnout    clang  5.55 -> 2.20   (−3.35)
                     gcc    8.70 -> 5.03   (−3.68)
```

Baselines 57 % apart, savings 9 % apart. What placement removes there is one
large streaming GEMV, and that is DRAM-bandwidth bound — DRAM does not know
which compiler built the host. libgomp's price is per-parallel-region overhead
(54 µs a region, from the OpenMP record), which falls on the many small
operations — router, shared expert, the gated norm — not on the single large
read being displaced.

That is also where the missing sixth is. The biggest row keeps 91 % of its
saving across the toolchains; the small ones keep less (`attention` 76 %,
`dn-sub proj` and `lm_head` 83 %), and the total lands at 83 %. Most of the
gain is bandwidth, which the compiler cannot touch; the remainder is the
per-region overhead it can.

**The control rows follow the build, not the session.** They move on clang —
`shared` +0.19, `router` +0.28, `cpu-miss` +0.27 — and are flat on gcc
(−0.01, −0.01, +0.06) in the **same session at the same residency**. That
disposes of the reason offered here first, that 98 % residency makes arm B pay
for displaced experts: residency is identical in all four arms and only clang
moves. It is a property of the binary, it is small against −5.05, and it is
not explained.

#### What the placed GEMVs actually spend — `COLI_CUDA_PROFILE`

The dense counters exist to tell two costs apart: kernels that leave the card
idle, and a call structure that never lets it start. One profiled run per arm,
outside the A/B because four `cudaEventRecord` on calls this short are overhead
on exactly the calls in question:

| | calls | GB | kernel | kernel GB/s | wall GB/s | round-trip |
|---|---:|---:|---:|---:|---:|---:|
| clang, arm A | 2704 | 92.37 | **475 ms** | **194** | 146 | 4 % |
| gcc, arm A | 2704 | 92.37 | **750 ms** | **123** | 98 | 7 % |
| clang, arm B | 5224 | 112.07 | 593 ms | 189 | 128 | 8 % |
| gcc, arm B | 5224 | 112.07 | 852 ms | 132 | 95 | 9 % |

**The kernel window is not all kernel.** The two arm-A rows issue the *same*
2704 calls over the *same* 92.37 GB to the *same* card, and the interval
between the events is 475 ms against 750 — 1.58×, from changing the **host**
compiler. A GPU does not know what compiled its host; what it can know is when
the launch arrives. `ev[1]` is recorded right after the H2D copy and `ev[2]`
after the launch, both on stream 0, so any delay in the CPU *issuing* the
launch is GPU idle time sitting inside the measured interval. About 275 ms of
gcc's "kernel" time is the host failing to feed the card — the spinning
16-thread libgomp team against the driver's own threads, which is the
contention the OpenMP record hypothesised and never measured. Read every
`kernel GB/s` in this document as a lower bound.

**The round-trip is not the problem.** It is 4–9 % of wall. `coli_cuda_matmul`
really is synchronous three times per call, and that really is not where these
GEMVs spend their time. The cost is inside the kernel window: partly host
contention, and the rest a kernel running at **194 GB/s against the card's
672 — 29 % of peak** on the toolchain that feeds it properly.

So the claim that motivated these counters — placed GEMVs running at 10–20 % of
the card — survives with better numbers. The 137 GB/s behind it was a
gcc-contaminated floor; 194 is the figure on a sane build, and it is still 3.5×
short of the hardware. That gap is now the largest single unexplained cost in
the decode path, and it is a kernel question, not a scheduling one.

One structural detail worth keeping: arm B issues **93 % more calls for 21 %
more bytes**, because `dnout` and `attnout` are smaller matrices than `dnproj`,
and its round-trip share doubles from 4 % to 8 %. Placing many small matrices
costs more overhead per byte than placing few large ones — which bounds how far
the `COLI_PLACE` allowlist is worth extending downward.

**A row that should not have moved.** `lm_head` is on the GPU in every arm of
the 2×2, and still goes 2.65 → 3.92 ms when only the *host* compiler changes.
A timer meant to price a GPU GEMV is carrying 1.27 ms of toolchain-sensitive
CPU time — consistent with the contention hypothesis in the OpenMP record (a
spinning 16-thread team against the driver's own threads) and a reason to read
every bytes-over-timer bandwidth in this document as a floor. `COLI_CUDA_PROFILE`
times the GEMV itself and is the way to settle it.

### The attention input projections (`attnproj`)

`q`, `k` and `v` read the same `x`, so concatenating them along `O` makes one
GEMV out of three — the trick `dnproj` already plays with qkv ++ z. On this
model that is 8192 + 512 + 512 rows of 2048, **18.9 MB a layer over 10
attention layers, 189 MB/token**: more than twice what `attnout` moves, for
**10 driver calls a token instead of 30**.

Unlike `dnproj`, the engine's `q`/`k`/`vv` scratch slices are `DN_PAD`'d
individually and so are not contiguous. The fused result therefore lands in a
scratch row and is split with three `memcpy` — about 36 KB a layer against the
18.9 MB the GEMV reads.

**Measured on 22 September 2026**, and offered to the automatic placer since.
These are the two arms this section wrote before anyone had run them:

```
COLI_PLACE="experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0"              # A
COLI_PLACE="experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0,attnproj=0"   # B
```

Six repetitions, order counterbalanced inside each repetition, heat table
frozen, RTX 4070 Ti SUPER 16 GB, read from `step() total` so that it is the
same metric the withdrawn 19 September figure was quoted on: **25.20 → 20.02 ms/token, median of
the paired deltas −5.1, worst of the six −4.0, 6/6 negative, +25.9 % decode
throughput**. (The wall-clock `Speed:` line gives −4.80 and +19.8 %, but it
carries the prefill, which this change does not touch — `S == 1` only — so a
percentage computed on it is not comparable to the decode percentages quoted
earlier on this page.) The
attention row goes 5.76 → 1.94 ms/token, which is 73 % of it; the rest lands on dense GEMVs that were
already on the GPU, and the raw record keeps the explanation for that labelled
as an untested hypothesis. `cpu-miss` stayed at 0.00 with the same VRAM hit
rate in both arms, so on this card the displaced experts are ones the run does
not ask for — a property of this card and this workload, not of the change.
Full run: `docs/experiments/qwen36-attnproj-place-2026-09-22-raw.txt`.

The byte model predicted 189 MB at **52.5 GB/s ≈ 3.6 ms** of CPU bus; the
measurement came in above that, and the record says where the surplus lands
without claiming to have proved why. (That prediction was written as 3.4 ms at
55.03 GB/s. The 55.03 has no source anywhere in this repository — see the note
further up — so the denominator is the 52.5 GB/s this machine was actually
measured at, and the predicted figure moves with it.)

One thing the measurement of `dnout`/`attnout` already settled, and that
bounds what is left: the **shared expert was not worth moving on the build
that measurement ran on**. It was the largest CPU item left at 10.09 ms/token,
and it is computed deliberately between `issue` and `take` so it overlaps the
GPU's expert groups.

The evidence is **`take` = 0.36 ms/token, from a run with profiling off**:
across 40 layers that is 9 us a layer, so the GPU finishes essentially the
moment the CPU arrives. The two halves are balanced, and moving the shared
expert onto that GPU would turn a 0.36 ms wait into the whole of its work.

> **Both numbers are arm B of the 19 September gcc session** — the build this
> page calls one nobody should ship — and the conclusion is written above in
> the present tense as though they were properties of the engine. They are
> not. On the clang build that ships, `shared` reads **4.0–4.3 ms/token**
> (`qwen36-place-clang-clean-2026-09-20-raw.txt`). And `take` has no value
> that can be written in the present tense at all. Every reading in the
> records of 19–21 September, at GRAPH=0 or on a build with no graph:
>
> | value | record | arm |
> |---|---|---|
> | 0.27, 0.69 | `qwen36-place-clang-2026-09-19-raw.txt:66` | B, A (clang, session 2) |
> | 0.28, 0.33, 0.33, 0.53 | `qwen36-expert-down-rows-2026-09-20-raw.txt:46` | the four DOWN_ROWS cells |
> | 0.31, 0.31, 0.43, 0.70 | `qwen36-place-clang-clean-2026-09-20-raw.txt:44` | cB, gB, gA, cA |
> | 0.76, 0.57 | `qwen36-graph-downrows-2x2-2026-09-21-raw.txt:42,44` | g0d0, g0d4 |
> | 1.14 | `qwen36-expert-graph-2026-09-20-raw.txt:35` | graph=0 |
>
> Thirteen readings from 0.27 to 1.14, eleven of them distinct, across arms
> that differ in placement, in `DOWN_ROWS` and in host compiler, and two of
> the records (`expert-graph`, `place-clang-clean`) carry no `Placement` line
> of their own. **No spread over the whole table means anything**, because it
> is not one configuration.
>
> Two of these records come closer than the rest —
> `qwen36-expert-down-rows-2026-09-20-raw.txt` and
> `qwen36-graph-downrows-2x2-2026-09-21-raw.txt` declare the same box, engine,
> model, prompt, `COLI_PLACE` and `COLI_CUDA_I8_ROWS=2`, with the graph off in
> both — and reading `take` across them gives +0.23 at `DOWN_ROWS=0` (0.53 →
> 0.76) and +0.24 at `DOWN_ROWS=4` (0.33 → 0.57).
>
> **That pair is not a drift measurement either, and a version of this note
> published it as one. Withdrawn 22 September 2026**, for three reasons that
> are worth keeping because each one is a way to fool yourself:
>
> - **It quotes the one axis that flatters it.** The same two arms move
>   +0.71 and +0.63 on `issue` (1.76 → 2.47, 1.84 → 2.47) and **+3.35 and
>   +3.05 on `step()`** (24.05 → 27.40, 25.20 → 28.25). The whole token is
>   14 % slower on the later day. +0.23 is not "the drift of `take`"; it is
>   `take`'s share of a level shift neither header explains. And the drift on
>   `issue` is as large as the entire effect this page attributes to the
>   graph on `issue` (−0.68), which the note used the drift to qualify.
> - **The two sides are different kinds of number.** The 20 September values
>   come from a section headed `2. WHERE THE TIME WENT (rip 1, ms/token)` —
>   **one repetition**. The 21 September values are cell aggregates over six.
>   Three paragraphs below, this page disqualifies another figure for being
>   exactly that: "one repetition, labelled rip 1 in that record".
> - **They are not the same binary.** The 20 September record was committed
>   at 14:55; `COLI_CUDA_X_BCAST` — which stops the activation vector being
>   copied once per expert — landed at 15:22 and shipped on by default. The
>   later session has it, the earlier one does not, and it sits on
>   `coli_cuda_expert_group_issue*`, the call whose completion `take` waits
>   for. No header declares the DLL, so "match on every axis their headers
>   declare" was true and useless.
>
> What survives is the original finding: **no run in these records pins one
> configuration and reads `take` twice**, and the near-match above is not a
> substitute for one. With `shared` at 4 ms instead of 10 and `take`
> unquotable, **this argument has to be re-run on clang before it can be
> believed**. It has not been.
>
> (Four corrections inside this note itself, all from review. It first listed
> five values "across six sessions", including a **1.30** that exists nowhere
> in this repository as a reading of `take` — the only 1.30 in the records is
> a paired `step()` delta in the 2x2's DOWN_ROWS list. Inventing a number
> inside a retraction of unsourced numbers is the defect this file exists to
> correct. It also listed **1.50**, which is not a session of its own but the
> GRAPH=1 cell of the same session as the 0.76. The second version then
> listed six values and split them into "placed" and "unplaced" arms: false —
> the 0.53 and the 0.76 come from records whose header pins the same full
> `COLI_PLACE` as the 0.31, and the 0.43 and the 0.70 are the same arm of one
> session differing only in host compiler. It also derived a spread of 0.61
> and called 0.61/0.36 "1.7×"; the arithmetic was right and the subset was
> not.) Noted 22 September 2026.

> **The two figures that used to be here are withdrawn, 22 September 2026.**
> The paragraph read that `COLI_CUDA_PROFILE` corroborates with 9.91 ms/token
> of expert-kernel time (1506 ms over 152 positions) against the CPU's 10.09,
> and that a group call takes 247.7 us. **Neither number has any provenance in
> this repository** — no raw record, no log, no date, no manifest; they are
> the only figures in this area with none. They also compared a *profiled*
> value against an *unprofiled* one from a different build. Either the run is
> produced, or they stay withdrawn.

> **And the conclusion that rested on them has to go too, 22 September 2026.**
> This paragraph used to read that the unprofiled `take` is the number the
> conclusion rests on, and that the way to unlock the shared expert is to make
> the expert kernels faster rather than add work to them. The second half was
> carried by the 247.7 us just withdrawn -- that figure was what said the card
> had headroom on this path. Withdrawing the evidence and keeping the
> recommendation, in the present tense, is not a retraction. The first half
> does not survive either: the note above establishes that no value of `take`
> can be written in the present tense at all. What remains is a question for
> the re-run on clang, not an answer.

First calibration, one Quadro RTX 4000 (8 GB), per-row int4 container, 200-token
decode, same prompt, output bit-identical in all four runs:

| | `off` | `auto` |
|---|---|---|
| trunk in VRAM | -- | lm_head 0.47 GB + 30 dnproj 0.70 GB |
| experts resident | 4,391 | 3,595 |
| cold: hit rate / tok/s | 44 % / 8.64 | 36 % / **9.62** |
| warm: hit rate / tok/s | 95 % / 9.63 | 90.6 % / **12.92** |
| same card, budget capped at 5 GB (a 6 GB card's share), warm | 88.9 % / 9.50 | 81.2 % / **13.15** |
| RTX 3070 (8 GB) alone, warm | 93.6 % / 11.09 | 88.4 % / **16.55** |
| both cards, experts on both, warm | 100 % / 10.79 | 100 % / **14.80** |
| reference: Ollama 0.32.5, same model Q4_K_M, same prompt, both cards (57 % CPU / 43 % GPU, 11.8 GB VRAM) | 20.3 warm (21.3 cold) | |

The warm row is the one that matters: at a 95 % hit rate the marginal expert
is as valuable as it gets on this card, and the trunk still wins by a third.
The hit rate drops only 4.4 points for 796 fewer residents because the
displaced experts are the coldest of the heat order -- exactly the ones the
placer priced as cheap. The relative win grows with the trunk's share of the
token: +34 % on the Quadro, +38 % at a 5 GB budget, +49 % on the 3070. (An
earlier version of this table had the two card names swapped: CUDA orders
devices fastest-first, `nvidia-smi` by bus, and I had read the wrong one.)
All sixteen runs of this calibration produced bit-identical text. Against
Ollama on the same box the gap closes from 1.5× (14.97 vs 22.4 in August, two
cards, hand-placed) to **1.23× on a single 8 GB card** (16.55 vs 20.3) --
with Ollama holding its dense weights at ~0.56 bytes per weight (Q4_K_M)
against this engine's 1.0 (int8), and using both cards.

**Two cards are the open case.** With experts on both cards the second card
paces every layer (the slower `take()` gates the chain), so `off` on two cards
is barely ahead of the 3070 alone (10.79 vs 11.09), and `auto` -- which in
this version spreads the trunk by free room and leaves the experts on both --
reaches 14.80 where the hand-written R4 split (`experts=0,lmhead=0,
dnproj=0:20+1:20`: experts on ONE card, trunk across both) reaches 17.11. On
two unequal cards the list still wins; the next version of the placer has to
learn that lesson (experts on one card, the trunk on the other) rather than
have it written for it.

Peak RSS is ~2 GB higher under `auto`: the host-side int8 copies stay as the
CPU fallback. Known, not yet addressed.

**Any dense matrix, by name.** The offer table is not limited to `lmhead` and
`dnproj`: an engine offers whatever it wants placed with `qt_trunk_offer(name,
layer, bytes)` before `qt_init`, asks `qt_place_of(name, layer)` afterwards,
and hands the placed matrices over as int8 rows with
`qt_dense_init(q, scales, I, O, device, gs)` (gs 0: one scale per row; gs > 0: scales `[O][ceil(I/gs)]`, the backend's grouped fmt 1), which returns a handle;
`qt_dense_matmul(handle, y, x, I, O)` answers one GEMV from VRAM and returns 0
(CPU from here on) if the backend fails. The qwen36 calls remain thin
wrappers over the same mechanism. Qwen3.8 uses it for its whole trunk -- 553
matrices, 4.0 GiB int8 on one card ([qwen38.md](qwen38.md), "GPU") -- and
that is also where the backend's resident dense matvec got its own staging
buffers: `coli_cuda_matmul` used to share the `x`/`y` device buffers with the
expert group, which runs asynchronously on its own stream between `qt_issue`
and `qt_take`. A dense GEMV issued in that window (Qwen3.8's shared expert)
overwrote the group's input and output mid-flight -- no CUDA error, only
wrong numbers. qwen36 never called the dense path inside that window, so its
outputs were unaffected.

## The per-row int8 dense GEMV: R output rows per block (`COLI_CUDA_I8_ROWS`)

The generic `quant_matmul` branch gives one block to one output row, so every
block reads the **whole** activation vector: a call moves `I*O` bytes of
weights and `I*O*4` of activations, and four fifths of the traffic is `x`.
`dense_stats` counts only the weights, which is why its reported GB/s sat at
29 % of the card while the kernel was in fact moving five times that.

`quant_matmul_i8r<R,U>` gives a block R rows. `x` is then read once per R
rows and the traffic falls from 5x the weight bytes to `(1 + 4/R)x`. The hot
path also issues `R*U` independent weight loads -- 16 in every instantiation
-- before consuming any, because Nsight Compute measures this kernel as
**memory-latency-bound, not bandwidth-bound**: occupancy 91.9 %, DRAM
throughput 34.6 %, 0.64 eligible warps out of 11 active, and 60 % of a
27-cycle issue gap stalled on an L1TEX scoreboard. Those two changes went in
together and the measurements below cannot separate them.

The summation order per row is untouched -- `i = t, t+256, ...` ascending,
the same 256-wide reduction tree, the same trailing f32 scale -- so the
output is **bitwise identical** to the original kernel's. That is the
contract `tests/test_int8_rows_cuda.cu` asserts with `memcmp`, on trunk
geometry, on `S > 1`, and on the short-block shapes (`O` = 13, 5, 17, 3, 1)
where the tail path runs.

### Measured (RTX 4070 Ti SUPER, sm_89, qwen36 i4 gs64, clang build)

Two sessions, each 4 arms x 4 alternated repetitions for `step()` plus one
profiled pass per arm for the kernel time. The repetitions generate 128
tokens from a 25-token prompt; the profiled pass generates 64. The two
columns below therefore come from different run lengths -- compare within a
column, never across. `dense_stats` weight bytes are
identical across arms (112.07 GB over 5224 calls), so the GB/s column is a
like-for-like ratio of kernel times.

| R | kernel GB/s (s1 / s2) | `step()` ms/token (s1 / s2) |
|---|---|---|
| 0 (original) | 187 / 191 | 27.25 / 27.40 |
| **2** | **381 / 372** | **24.35 / 23.45** |
| 4 | 389 / 342 | 23.75 / 23.90 |
| 8 | 318 / 299 | 23.70 / 23.80 |

>  **The `kernel GB/s` column is SUSPENDED, 22 September 2026**, along with
>  everything derived from it below. The same 5224 calls over the same 112.07
>  GB read 175 GB/s in `qwen36-dense-pinned-2026-09-21-raw.txt` the next day,
>  a factor of 2.1 from the 372 here; see the note further down and the
>  `COLI_CUDA_I8_ROWS` row of `docs/ENVIRONMENT.md`. Note that
>  `qwen36-dense-pinned` does not pin `COLI_CUDA_I8_ROWS` in its header
>  either, so the objection raised against the third reading applies to it
>  too: of the five records that pin the variable, neither of the two low
>  readings is among them. Do not cite a GB/s figure
>  for this kernel until a session measures R=0 and R=2 back to back with the
>  variable explicitly pinned. The `step()` column is unaffected.

**The step gain reproduces and is large**: every `R >= 2` arm lands below
every `R = 0` arm -- about **-13 % on the whole token**, from 36.6 to 42.5
tok/s. (This paragraph used to claim "no overlap between the original and any
of them in either session". Only session 2 retains per-rep lines; session 1
says "per-rep lines not retained" at `qwen36-i8-rows-2026-09-20-raw.txt:32`,
so there the check is median-to-median. Qualified 22 September 2026.)

**The ranking among R does not reproduce** -- and the paragraph that used to
stand here got the direction wrong. **Withdrawn 22 September 2026.** It read
that R=4 won session 1 on both metrics, that R=8 was the slowest of the three
in both sessions on both metrics, and that R=2 is "no worse than R=4 on
anything measured". The table three lines above says otherwise on `step()`:
session 1 is R=2 24.35 / R=4 23.75 / **R=8 23.70**, so R=8 is the FASTEST and
R=2 the slowest; session 2 is R=2 23.45 / **R=4 23.90** / R=8 23.80, so R=4 is
the slowest. R=8 is consistently last only on the GB/s column, which is
suspended above. On `step()` the three sit within 0.65 ms of each other in
session 1 and 0.45 in session 2, and no ordering survives both sessions. R=2
ships as the cheaper of the widths in registers and in shared memory
(`partial[R][256]`), which is the side to err on for an architecture nobody
has run this on -- not because it was measured best.

The traffic model was compared against the GB/s column: measured over
predicted 1.17-1.22 at R=2, 0.72-0.83 at R=4, 0.47-0.51 at R=8. **That
comparison is suspended with the column**, and so are the two conclusions
drawn from it -- that past R=2 the `x` re-read is no longer what limits the
kernel, and that R=8 gives back in occupancy and registers more than it saves
in traffic. A ratio whose denominator is a contaminated kernel time says
nothing, and `step()` does not stand in for it: see the withdrawal above.

### A measurement caveat that cost a day

Under `CUDA_DLL=1` (the Windows path) `CUDA_OBJ = backend_loader.o`: the
executable links only the loader, and **every kernel in this file lives in
`coli_cuda.dll`**, which only `make cuda-dll` rebuilds. A measurement script
that rebuilds the engine and not the DLL measures the DLL it started with, in
every arm, and reports a clean flat result with a straight face. Three
kernel measurements were lost that way before anyone noticed; the numbers
above are from runs where the DLL is newer than `backend_cuda.cu`, and the
measurement script now refuses to start otherwise. Any future kernel A/B on
Windows has to prove the same thing before its numbers mean anything.

## The pair: expert graph + expert down-rows (`COLI_CUDA_GRAPH`, `COLI_CUDA_DOWN_ROWS`)

**Both on by default. Neither is worth anything alone; one of them is a
regression alone. They ship together or not at all.**

Three optimisations in this area measured correct and worth exactly zero on
the token, and one fact explains all three. The MoE phase is

```
issue + max(CPU window, GPU group) + residue
```

and `take` is what is left of the GPU group after the CPU stops covering it.
**`take` > 0 in every measurement taken on this path**, which means the phase
is GPU-bound *with idle CPU inside it*. A CPU-side saving there converts into
waiting; a GPU-side saving is capped by how far `take` can fall.

That model made a prediction, and the prediction is why these two are on:

- `COLI_CUDA_GRAPH` is a **CPU-side** saving. It should widen `take`.
- `COLI_CUDA_DOWN_ROWS` is a **GPU-side** saving. It needs a wide `take` to
  have anywhere to go.
- So they should **interact**, negatively. If the interaction came out zero or
  positive, the model was wrong and had to be thrown away.

### The 2×2

Four cells, six alternated repetitions each, `N_NEW=128`, heat table frozen,
hit rate 98.1 % and `miss(CPU)` 901 identical in every cell.

| cell | `step()` | `issue` | `take` | `moe` |
|---|---|---|---|---|
| graph 0, rows 0 | 27.40 | 2.47 | 0.76 | 9.92 |
| graph 1, rows 0 | 27.35 | 1.79 | 1.50 | 9.60 |
| graph 0, rows 4 | 28.25 | 2.47 | 0.57 | 10.21 |
| **graph 1, rows 4** | **27.05** | 1.83 | 0.73 | **9.09** |

```
GRAPH at ROWS=0    -0.20 ms/token   signs disagree
GRAPH at ROWS=4    -1.50 ms/token
ROWS at GRAPH=0    +1.15 ms/token
ROWS at GRAPH=1    -0.25 ms/token   signs disagree

INTERACTION        -1.40 ms/token   <- the model holds
```

And the diagonal, which is what a default change actually rests on —
everything off against everything on:

```
rep      off     on    delta
 1     27.90  27.60   -0.30
 2     27.20  27.10   -0.10
 3     27.10  26.70   -0.40
 4     27.60  27.20   -0.40
 5     27.00  26.70   -0.30
 6     28.10  27.00   -1.10
             median  -0.35
             mean    -0.43     6 of 6 repetitions negative
```

**−0.43 ms/token, 1.6 %.** Small, and the first thing to move on this path.

> **Qualified 22 September 2026.** −0.43 is the MEAN. The paired median in the
> same record is **−0.35** (`qwen36-graph-downrows-2x2-2026-09-21-raw.txt:69`,
> printed there ABOVE the mean), and this transcription had dropped that line
> while `docs/ENVIRONMENT.md` quoted it; it is restored above, in the record's
> own order. Every other record of
> that week quotes paired medians, so publishing the mean was choosing the
> more favourable of the two. The within-cell spread (0.70–1.90) is 1.6–4.4×
> the mean and 2.0–5.4× the median. **What the pair rests on is the 6/6 sign
> agreement, not the distance between the two cells.**

The mechanism is confirmed to within 0.06 ms: the graph takes 0.68 ms out of
`issue` and 0.74 ms reappears in `take`. A transfer, not a saving — until the
down-rows kernel is there to spend it.

### Why the pair, and not two switches

`COLI_CUDA_DOWN_ROWS=4` **alone costs 1.15 ms/token**. The kernel is 18 %
faster on the GPU (expert-group kernel time 697 → 572 ms over 64 tokens) and
the token gets *worse*, because 91 % of the regression lands on `dn-sub proj`,
`norm+out` and `lm_head` — the placed dense GEMVs, which run on stream 0
concurrently with the expert group. 4,096 long-lived blocks holding 4 KB of
shared memory each leave fewer scheduling slots than 16,384 that retire at
once.

So turning the graph off while leaving down-rows on is **the one combination
known to be worse than shipping neither**. It is not silently corrected — a
flag that means one thing is worth more than a flag that quietly rewrites
another — but `down_g4_launch` prints a warning when it sees it, and
`COLI_DOWN_ROWS_DEFAULT` is compiled to 0 wherever `COLI_GPU_HAS_GRAPH` is 0
(HIP has no `cudaStreamBeginCapture`), because there the arm that pays for the
kernel does not exist.

### The cap has moved, not gone

`DOWN_ROWS=4` removes ~1.9 ms/token of expert-group GPU work and the token
gains 0.43 — a ratio this page used to quote as "22 % of the GPU saving
reaches the token". **Withdrawn 22 September 2026**: the ratio is 0.43 / 1.9,
and its two terms come from different worlds. The **denominator** (1.9
ms/token, from 697 → 572 ms over 64 tokens) is the difference of two
*profiled* passes, taken under `COLI_CUDA_PROFILE`, where by design the graph
does not run at all; the **numerator** (0.43) is an *unprofiled* A/B over 128
tokens. They come from two different worlds and their ratio means nothing.
The noise is visible in the record itself: g0d4 572 against g1d4 587, two
cells that should be identical, 15 ms apart.

What each of the two does support, on its own terms, is unaffected: the
profiled pair says the kernel really is ~1.9 ms/token cheaper on the GPU, and
the unprofiled A/B says the token really does gain 0.43. What cannot be said
is what fraction of one is the other. A third figure from the same record
makes the same point without a ratio: in the best cell `take` is still 0.73
ms/token (`qwen36-graph-downrows-2x2-2026-09-21-raw.txt:45`), so the phase
remains GPU-bound with idle CPU inside it and there is
still headroom that this kernel did not convert.

### What this is not

One box, one model, one placement, six repetitions. Half the pair is a
measured regression on its own, and the pair is justified only by the
interaction above. On different silicon the SM-contention term could scale
differently from the `take` headroom the graph opens, and the pair could go
the other way. **Measure the 2×2 before trusting these defaults on other
hardware.** The within-cell spread (0.70–1.90 ms) is larger than the −0.43 the
diagonal claims; the claim rests on the pairing and on 6/6 sign agreement, not
on the medians being far apart.

Both changes are **bitwise identical** to the paths they replace — `memcmp` in
`tests/test_grouped_down_rows_cuda.cu` and `tests/test_grouped_g4_cuda.cu` —
so no logit and no token can move, and the tiny-model oracle is untouched.

Full record: `docs/experiments/qwen36-graph-downrows-2x2-2026-09-21-raw.txt`.

### A correction that shipped with them

`tests/test_grouped_g4_cuda.cu` holds the **only** assertions that ever
*execute* the expert graph — capture, two replays, and the per-call path after
the graph is switched off. It was compile-only: it used `cudaMallocManaged`,
which `backend_gpu_compat.h` does not map for HIP, so it could not join
`make cuda-test`, which also serves `make hip-test`. **The graph had shipped
with its bitwise assertions never run anywhere.** The managed allocations are
gone, the file is in `cuda-test` and `gpu-compile`, and its reference arm now
pins `COLI_CUDA_GRAPH=0` explicitly — without that, the new default would have
made every `memcmp` in it compare the graph against itself.

## The dense round-trip: pinned host staging (`COLI_CUDA_DENSE_PINNED`)

`coli_cuda_matmul` is the placed dense GEMV — `dnproj`, `dnout`, `attnout`,
`lm_head`, the shared expert. It runs **81.6 times a token** on qwen36, and
each call does exactly this:

```
cudaMemcpy(ctx->dx, x, xb, H2D)      synchronous, from PAGEABLE host memory
quant_matmul_launch(...)
cudaMemcpy(y, ctx->dy, yb, D2H)      synchronous, to PAGEABLE host memory
```

`COLI_CUDA_PROFILE=1` prices it. From the R=2 arm of the `COLI_CUDA_I8_ROWS`
session (5224 calls over 64 tokens, `docs/experiments/qwen36-i8-rows-2026-09-20-raw.txt`):

| | total | per token | per call |
|---|---|---|---|
| kernel | 301 ms | 4.70 ms | 57.6 µs |
| h2d | 96 ms | 1.50 ms | 18.4 µs |
| d2h | 120 ms | 1.88 ms | 23.0 µs |
| residue (launch + sync) | 71 ms | 1.11 ms | 13.6 µs |
| **wall** | **588 ms** | **9.19 ms** | **112.6 µs** |

>  **This table and the `COLI_CUDA_DENSE_PINNED` one further down are
>  incompatible, and nothing
>  on this page said so until 22 September 2026.** Same path, same
>  configuration, same card: here the kernel column reads **301 ms** over
>  5224 calls and 112.07 GB (372 GB/s); in the `COLI_CUDA_DENSE_PINNED`
>  table further down, arm m0 — the *unmodified* arm — reads **640 ms** for
>  the same 5224 calls and the same 112.07 GB (175 GB/s). A factor of 2.1,
>  one day apart. A THIRD reading exists —
>  `qwen36-place-clang-clean-2026-09-20-raw.txt:60-61` gives **593 ms / 189
>  GB/s** on arm cB for the same 5224 calls and the same 112.07 GB — and it
>  neither widens nor closes the gap: 189 falls inside [175, 372] and sits
>  within 8 % of the low reading. What it does do is make the low band look
>  less like an outlier: 189 is also 1 % from the 191 GB/s that
>  `qwen36-i8-rows-2026-09-20-raw.txt:106-107` reports for the R=0 arm, so
>  this third reading may simply be an R=0 measurement — that record pins
>  `COLI_CUDA_I8_ROWS` nowhere. **And neither does `qwen36-dense-pinned`**,
>  the source of the 175: the same objection applies to it, and an earlier
>  version of this note raised it against one reading only. Five records in
>  `docs/experiments/` pin the variable; neither low reading comes from one. (An earlier version of this note called it a
>  third reading of the same configuration that "widens the spread". Neither
>  half survives. Retracted 22 September 2026.)
>
>  "Never explained" overstates it. `docs/ENVIRONMENT.md:228` already
>  documents a mechanism of this family and prescribes how to read it: the
>  same 2704 calls over the same 92.37 GB read 475 ms under clang and 750
>  under gcc — **purely from the host compiler**, because a CPU slow to issue
>  leaves the card idle inside the interval the kernel column measures. 1.58×
>  there, 2.1× here, same axis. The better-sourced suspect is on that axis
>  too: of the three records carrying `dense_stats`, the pinned one is the
>  only one whose header declares `OMP_NUM_THREADS=16 (7950X physical cores),
>  WAIT_POLICY=ACTIVE` — sixteen spinning threads against the submit thread.
>  (An earlier version of this note named a different suspect, that the
>  pinned record was the only one of the week not pinning
>  `COLI_CUDA_I8_ROWS`. That is false: `place-clang-clean` and
>  `xoff-fix-noop` do not pin it either. Retracted 22 September 2026.) Until
>  this is resolved,
>  **neither table's absolute figures should be cited**, and the derived
>  "4.48 ms/token" below inherits the doubt — it also compares a profiled
>  value against an unprofiled 25 ms token, which this page warns against
>  elsewhere.

**Half the wall clock of this path is not the kernel.** 4.48 ms/token of it
is transfer and driver overhead, against a 25 ms token — and unlike the expert
group, *none of it is hidden*: these copies are synchronous, so the engine
thread is stopped inside them. There is no CPU window covering this the way
`qt_issue`/`qt_take` covers the expert group, which is precisely why the three
optimisations that preceded this one were each correct and each worth zero.

A pageable `cudaMemcpy` is not one copy. The driver cannot DMA out of memory
the OS may page, so it stages through a pinned buffer of its own — two copies,
one of them serialised against the caller. `COLI_CUDA_DENSE_PINNED` gives that
buffer to the backend instead:

| value | behaviour |
|---|---|
| `0` / unset | the original path: two synchronous pageable copies |
| `1` | pinned staging, both copies still synchronous |
| `2` | pinned staging, both copies async on stream 0, **one** synchronize |

Three values rather than two on purpose. There are two separate mechanisms in
here — the driver's hidden staging, and the two blocking round-trips — and a
single flag that changed both would produce a number nobody could attribute.
`1` prices the staging alone; `2` adds the collapse to one wait.

The staging cannot change a value, only where the bytes are read from, so the
contract is **bitwise identity** across all three modes and
`tests/test_dense_pinned_cuda.cu` asserts it with `memcmp`. Mode 2 is the one
that needs a test: async copies into a reused host buffer fail *silently* when
the synchronize is missing — the caller reads the previous call's answer, which
for a repeated shape is usually almost right. The test therefore interleaves
mode 2 with the pageable path on a **new input every iteration**, where a stale
buffer shows up as the previous answer instead of passing unnoticed.

### The prediction, written down before the run -- and wrong

*Kept as written, because the failure is the useful part. It is answered below.*

Every dense call here is `S=1`: `xb` is `I*4` — **8 KiB** for the qwen36 trunk —
and `yb` is smaller still. At that size pinned memory buys essentially nothing
in *bandwidth*; the driver's hidden staging copy it removes is itself an 8 KiB
memcpy. So:

- **mode 1 should be worth roughly zero.** It removes a small copy and nothing
  else. If it shows a large gain, the explanation is not the one written here
  and the result should be distrusted until it is.
- **mode 2 is the actual bet.** It collapses two blocking waits into one, and
  the residue line already prices those at 13.6 µs/call on top of the 41.4 µs
  the two transfers cost. Pinned memory is only the precondition: a
  `cudaMemcpyAsync` out of pageable memory is synchronous anyway.

If mode 2 is also flat, the conclusion is not "try harder here" — it is that
81.6 driver round-trips a token is the floor of this design, and the next lever
is **fewer calls**, not faster ones: batching the placed GEMVs, or a graph over
the dense trunk the way `COLI_CUDA_GRAPH` does for the expert group.

### Measured (RTX 4070 Ti SUPER, sm_89, qwen36 i4 gs64, clang, 3 x 6 alternated)

**Both pinned arms make the token slower, 6/6 repetitions each. Leave it off.**

| arm | `step()` | deltanet | dn-sub proj | attention | lm_head |
|---|---|---|---|---|---|
| m0 pageable | **27.20** | 10.25 | 5.40 | 5.77 | 2.30 |
| m1 pinned sync | 28.75 | 11.53 | 6.05 | 5.97 | 2.34 |
| m2 pinned async | 29.20 | 11.93 | 6.25 | 6.07 | 2.35 |

```
m1 - m0   +1.60 ms/token   [1.90 1.50 1.80 0.90 1.60 1.60]   6/6 positive
m2 - m0   +2.00 ms/token   [2.00 1.90 1.90 2.00 2.00 4.50]   6/6 positive
m2 - m1   +0.40 ms/token   [0.10 0.40 0.10 1.10 0.40 2.90]   6/6 positive
```

Every phase moves the same way, and they are exactly the phases holding the
placed dense GEMVs, so the damage is where the change is. Hit rate 100 % and
`miss(CPU)` 0 in all three arms: the arms generated the same thing.

`COLI_CUDA_PROFILE`, 5224 dense calls over 64 tokens, one run per arm
(residue = `wall - h2d - kernel - d2h`, i.e. launch and synchronisation):

| arm | h2d | kernel | d2h | residue | wall |
|---|---|---|---|---|---|
| m0 | 140 ms | 640 ms | 171 ms | 39 ms | 990 ms |
| m1 | **186 ms** | 604 ms | 140 ms | 83 ms | 1013 ms |
| m2 | 118 ms | 607 ms | **34 ms** | **160 ms** | 919 ms |

Mode 2 did exactly what it was built to do: the D2H collapses by 80 %, and the
wait does not vanish, it **moves** into the residue -- the one
`cudaStreamSynchronize`. Transfers plus residue, 350 -> 312 ms. And the token
is 2.00 ms worse. (The profile pass and the A/B are different runs -- 64 vs 128
tokens, events on vs off -- so each block is read on its own, with no
arithmetic across them.)

### Why the premise was wrong

`h2d` rose from 140 to 186 ms in the arm that is supposed to be doing strictly
less work. That is the answer.

CUDA's documented behaviour for a **pageable** host-to-device `cudaMemcpy` is
that it returns once the bytes are in the driver's staging buffer -- the DMA to
the device need not have completed. Every dense call on this path is `S=1`, so
`xb` is 8 KiB, well inside that window.

So the two "synchronous pageable copies" this toggle set out to remove **were
not both synchronous**. The H2D was effectively fire-and-forget, and the
driver's hidden staging copy was not a cost: on an 8 KiB payload it was buying
asynchrony for free, because the copy is cheaper than the wait. Pinning the
buffer removed the staging copy and, with it, the early return.

Mode 2 restores asynchrony explicitly and still loses, by a further 0.40 ms. A
blocking `cudaStreamSynchronize` per call, 81.6 times a token, against 16
OpenMP threads spinning under `OMP_WAIT_POLICY=ACTIVE`, is not obviously
cheaper than the driver's own early return. This run does not separate that
from the other candidates and does not pretend to.

**Kept, off by default, as the B arm for a fact worth not re-deriving: on this
path the pageable copy is the fast one.** It also retires the hypothesis that
opened the day -- the 4.48 ms/token outside the dense kernels is real, but it
is not two removable copies. Full record:
`docs/experiments/qwen36-dense-pinned-2026-09-21-raw.txt`.

## Measured (Threadripper 3945WX 12C, RTX 3070 8 GB + Quadro RTX 4000 8 GB, Qwen3.6-35B-A3B int4, 200-token decode)

| | 1 GPU (8 GB) | 2 GPUs (16 GB) |
|---|---|---|
| decode tok/s (cold / warm heat) | 9.2 / 9.9 | 10.6 / **11.3** |
| VRAM-resident experts | 4,391 (43 %) | 8,532 (83 %) |
| VRAM hit rate (cold / warm) | 44 % / 95 % | 85 % / 100 % |
| peak RSS | 40 GB | **29 GB** |
| reference: Ollama q4_K_M, same box | 7.5 | 10.5 |

All figures above are int4-container measurements; the peak-RSS row in
particular has no int8 analogue (see **Memory**).

CPU-only baseline of this engine before the tier: 0.35 tok/s.
Numerics: logits cosine vs the f32 CPU reference 0.9992 (dense int8 on),
bit-identical GPU-vs-CPU on the same container (cosine 1.0000001).
