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
throughput. For 0.31 GB of VRAM and 177 experts (7136 → 6959), hit rate
unchanged at 100 %. The effect is 7.77 ms against a worst-case within-arm
spread of 2.1 ms.

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
per-row scales — worth 6.1 ms at this machine's measured 55.03 GB/s. The direct rows account for −4.95, the
shortfall being GPU time that is not free. But `dn-sub proj`, `lm_head`,
`take` and `issue` improve by a further −2.77 ms while **already residing on
the GPU in both arms**. The likely mechanism is contention: 335 MB/token less
CPU traffic leaves the staging copies and driver calls of the already-placed
matrices more bus to work with. That is a hypothesis, not a measurement.

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
level: 59.75 ms/token is a build nobody should ship, and **−7.77 ms is very
likely an overestimate for a clang build**. Placement pays by removing CPU
reads, and libgomp makes those reads cost about twice what libomp charges, so
the same 335 MB/token buys less on a sane build. Re-measuring on clang is
outstanding.

The same caveat reaches the bytes-to-time arithmetic two paragraphs up: the
55.03 GB/s that turns 335.5 MB into 6.1 ms is this machine under libgomp, not
this machine.

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

**Explicit only**, and that is the point: `dnout` and `attnout` are offered to
the automatic placer because an A/B measured them, and this has not been
measured. Name it or it stays on the CPU:

```
COLI_PLACE="experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0"              # A
COLI_PLACE="experts=0,lmhead=0,dnproj=0,dnout=0,attnout=0,attnproj=0"   # B
```

**No speed number is claimed here.** The fused block is proven to return
exactly what the three separate CPU matmuls return, in order, against the fake
CUDA backend — not that it returns them sooner. The byte model says 189 MB at
55.03 GB/s is 3.4 ms of CPU bus, of which the GPU will take some back.

One thing the measurement of `dnout`/`attnout` already settled, and that
bounds what is left: the **shared expert is not worth moving**. It is the
largest CPU item left at 10.09 ms/token, and it is computed deliberately
between `issue` and `take` so it overlaps the GPU's expert groups.

The evidence is **`take` = 0.36 ms/token, from a run with profiling off**:
across 40 layers that is 9 us a layer, so the GPU finishes essentially the
moment the CPU arrives. The two halves are balanced, and moving the shared
expert onto that GPU would turn a 0.36 ms wait into the whole of its work.

`COLI_CUDA_PROFILE` corroborates with 9.91 ms/token of expert-kernel time
(1506 ms over 152 token positions) against the CPU's 10.09 — but that figure
comes from a profiled run, and `backend_cuda.cu` says in as many words that
four `cudaEventRecord` per call at 40 calls a token add launch overhead to the
path whose launch overhead is the question. The event interval is genuine GPU
timeline, yet a slower dispatch can widen the gaps inside it, so treat 9.91 as
an **upper bound** on how busy the card really is. The unprofiled `take` is
the number the conclusion rests on.

The way to unlock the shared expert is to make the expert kernels faster —
247.7 us per group call for 25M MACs is a small fraction of what the card can
do — not to add work to them.

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
