# Qwen3.6-35B-A3B decode: Colibri against llama.cpp, one host

**18 September 2026** · Colibri `8e58948` · llama.cpp `ec9281505` (b11042)
· raw output: [`qwen36-vs-llamacpp-2026-09-18-raw.txt`](qwen36-vs-llamacpp-2026-09-18-raw.txt)
(sha256 `45219424a3900db55d372d8efd4e09f577fa3477816cfd01dc7a3cda358d7196`)

## Why this is not a validated manifest

`c/experiment_manifest.py` checks a record with **one** declared variable whose
config diff matches. This is not that: engine, quantisation and codebase all
change together. Forcing an `ENGINE` field into the schema would make the
validator pass while the experiment stayed uncontrolled, which is worse than an
honest narrative. So this file is a lab record, and it states its confounds
instead of hiding them behind a green validator.

## Result: llama.cpp is faster here, by 18% at equal bytes

| | container | generation tok/s | ms/token |
|---|---|---|---|
| llama.cpp `-ncmoe 99`, Q8_0 | 34.36 GiB | 26.27 ± 1.63 | 38.07 |
| **Colibri**, int4 gs64 experts + int8 trunk | **21.40 GiB** | **31.3** | **32.0** |
| llama.cpp `-ncmoe 99`, Q4_K_M | 20.60 GiB | 37.97 ± 0.13 | 26.34 |

Same file, other placements: `-ngl 99` alone 11.46 ± 0.07, `-ngl 0` 10.62 ± 0.19.

The containers land within 4% of each other, so the Q4_K_M row is close to a
direct comparison and needs no model to read: **+3.9% of bytes for +21.5% of
time**. Normalising that 4% through llama.cpp's own two points,

```
ms/token = 8.78 + 0.8524 x GiB
           ^^^^   ^^^^^^
           fixed  per GiB of container
```

their curve predicts 27.02 ms/token at 21.4 GiB against 32.0 measured here:
**4.98 ms/token behind, 18.4%.**

### How this record reached the opposite conclusion twice

It is worth writing down, because both wrong answers looked finished.

1. **"llama.cpp is 12x slower."** First run used a 27 B *dense* GGUF. Different
   model class; see the last section.
2. **"Colibri is 19% faster."** Q8_0 only, which carries 60% more bytes than our
   container. The precision confound was noted as a caveat under the result —
   the caveat *was* the result.
3. **"Inconclusive, 19% ahead of Q8_0 and 18% behind Q4_K_M."** True but
   incomplete: it treated the bracket as unresolvable when one `Get-ChildItem`
   resolved it.

The container is **21.4 GiB**, below the 27.2 GiB threshold this record set for
itself before measuring, which puts the engine below llama.cpp's curve. The
prediction written down beforehand was 19-22 GiB, i.e. below. That is the only
reason this third answer deserves more trust than the first two: the falsifiable
form was committed to first.

The fit still uses container size as a stand-in for bytes touched per token,
which for a sparse MoE is a proxy — it holds across two quantisations that scale
every tensor, less well for one that treats experts and trunk differently, which
is what Colibri's container does. But at a 4% size difference the normalisation
barely matters: the Q4_K_M row alone carries the conclusion.

## Comparing the right quantities

`llama-bench`'s `tg128` is generation only; prompt processing is the separate
`pp512` row. Colibri's `Speed:` line is **not** the same thing: its `dt` wraps
`generate()`, which prefills the prompt and then generates, while its numerator
counts only generated tokens. On this run that is 1.92 s of prefill charged to
128 decode tokens, reporting 21.36 tok/s for an engine generating at 31.2.

The comparable Colibri number is `[timers] step() total`, printed under
`COLI_TIMERS=1`. Quoting `Speed:` against `tg128` would have made Colibri look
27% slower instead of 19% faster — a 46-point swing from reading the wrong line.

For the record, the three numbers this engine prints:

| | cold | warm |
|---|---|---|
| phase sum | 30.54 ms/token | 30.64 |
| `step() total` | 32.1 | 31.9 |
| `outside the phases` | 1.5 | 1.3 |
| `Speed:` (prefill included) | 21.36 tok/s | 21.11 |

`outside the phases` at 1.3-1.5 ms settles a suspicion raised while setting this
up: the 16 ms/token that seemed missing between the phase sum and the reported
tok/s was prefill in its entirety, not an untimed hole in decode.

## The confound, which turned out to be the whole story

Colibri ran **int4 gs64 routed experts with an int8 dense trunk**. llama.cpp ran
Q8_0 throughout in one run and Q4_K_M in the other. Neither matches: Q8_0 is
heavier than Colibri everywhere, Q4_K_M is lighter on the trunk — and the trunk
is what this engine spends most of a token in (DeltaNet in_proj and out_proj,
attention, lm_head). So Colibri is bracketed but not matched, and the bracket is
±19%.

A matched run would need llama.cpp quantised the way this container is: experts
around 4 bits, trunk around 8. `--tensor-type` overrides can express that; nobody
has built it. Note also that the Q4_K_M file served was `UD-Q4_K_M`, an unsloth
dynamic quant that upcasts selected tensors, so it is not a plain Q4_K_M either.

Also not equalised: Colibri's expert tier had a warm `heat-qwen36.bin`, and its
container lives on the same NVMe as the GGUFs but is read through a different
path.

Also not equalised: Colibri's expert tier had a warm `heat-qwen36.bin`, and its
container lives on the same NVMe as the GGUF but is read through a different
path.

## What replicated independently: the split

`-ncmoe` keeps MoE expert weights on CPU and leaves the dense trunk on the GPU —
the division Colibri is built around. In llama.cpp it is worth **26.27 against
11.46**, a 2.3x gain over that engine's own default, which fills VRAM with whole
layers, experts included, and pays PCIe for the remainder every token. Its
default barely beats all-CPU (11.46 against 10.62): without `-ncmoe` the GPU is
buying it almost nothing here.

That is a second engine, written by other people, agreeing with the architecture
on the same hardware. It is the most transferable thing in this record.

## What it pointed at next

Two lines from the same run, previously unread:

```
[timers]   qtier: issue 1.95 | cpu-miss 3.69 | take 1.37 ms/token
```

### Correction: `cpu-miss` is not a miss counter

The first version of this section read that line as two targets worth 7 ms and
said the deficit was "smaller than the sum of two named causes". One of the two
does not exist. `g_qt_cpu` is not the cost of experts missing the VRAM tier; it
is the whole CPU-side overlap window, and `c/qwen36.c` puts the shared expert
inside it on purpose, to cover the GPU's latency:

```c
double _q1 = tm_now();
for (kk...) { if (qmask & (1u<<kk)) continue;   /* the experts the GPU declined */
              ... }
/* Compute the shared expert NOW so it overlaps with the GPU groups */
{ double _ts2 = tm_now(); ... shared expert ... tm_add(S, 3, tm_now()-_ts2); }
double _q2 = tm_now();
g_qt_cpu += _q2 - _q1;
```

The `(shared)` row of the same report already counts that work, so subtracting
gives what the misses actually cost:

| | `cpu-miss` | `(shared)` | difference |
|---|---:|---:|---:|
| cold | 3.69 | 3.69 | **0.00** |
| warm | 3.21 | 3.20 | **0.01** |

Zero. The VRAM tier hits essentially every time on this configuration, and the
3.7 ms is the shared expert doing useful work in the shadow of the GPU.

What survives:

- **`issue`, 1.95 ms.** CPU time spent launching. Per MoE layer, two
  `cudaMemcpyAsync` plus two or three kernels — about five driver calls, ~200
  per token over 40 layers, `1.95 / 200 ≈ 9.8 us` each, which is WDDM's price on
  Windows against 2-3 us on Linux. This is the part CUDA graphs collect
  (`ggml-cuda.cu:2558-2660` is llama.cpp's version), and the shape suits them:
  the chosen experts travel as data in `GroupDesc[]`, read by the kernel from a
  device buffer, so graph topology does not follow routing.
- **`take`, 1.37 ms.** `coli_cuda_expert_group_take` is `cudaStreamSynchronize`.
  It is the CPU waiting on the GPU, not a launch cost, and **no graph removes
  it.**

So 3.3 ms, of which 1.9 is graph-shaped — against a 4.98 ms deficit. The gap is
no longer smaller than its named causes, which is the honest position: part of
it is still unaccounted for.

Two things left unread in the same output, both free: `[qtier] VRAM hit rate`,
which states the hit rate directly instead of inferring it, and
`[qtier] group_stats`, which splits the GPU's time into h2d / kernel / d2h and
so says whether the `take` wait is transfer or arithmetic.

## A discarded measurement, kept

The first llama.cpp run used `ggml-org/Qwen3.6-27B-GGUF:Q8_0` and returned
**1.78 tok/s**. That is a 27 B **dense** model: it reads all 26.62 GiB per token
against roughly 3 B active parameters for a 35B-A3B, and it does not fit in
16 GiB of VRAM. Per-token bytes explain it to within the measurement:

```
~16 GB in VRAM at ~670 GB/s     ->  ~24 ms
~12 GB in RAM over PCIe 4.0 x16 -> ~375 ms
                                    ~400 ms  ->  ~2.5 tok/s   (measured 1.78)
```

It is in the record because for about twenty minutes it read as "llama.cpp is
12x slower", which is the same error as the confound above with the sign
reversed. Benchmarking a different model class and calling it an engine
comparison is easy to do twice.
