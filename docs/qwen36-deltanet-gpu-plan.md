# Qwen3.6 DeltaNet decode on the GPU — plan

Status: **proposal, no code yet.** Written 28 September 2026 from the
measurements in `docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt`
(sections 8-10). Each stage below is a separate PR with its own tests and,
where it touches ms/token, its own A/B.

## Why

On the reference machine (RTX 4070 Ti SUPER, 7950X, Windows, `qwen36_i4_gs64`,
cap 256, heat table, keep-alive on), a decode token costs **22.93 ms**
(record section 10). DeltaNet is 7.83 of it, split by the `dn-sub` and
`dn-split` timers (ms/token, 30 DeltaNet layers):

| part | ms/token | where it runs |
|---|---|---|
| `qkvz` projection | 2.81 | GPU call: upload x, int8 GEMV, download 12288 floats, sync |
| `a`+`b` matmuls | 0.66 | CPU, one OpenMP region, 64 x 2048 f32 |
| conv + SiLU | 0.43 | CPU |
| l2norm + recurrence | 2.00 | CPU, 32 heads x 128 x 128 state |
| gated RMSNorm | 0.13 | CPU |
| `out` projection | 1.79 | GPU call: upload 4096 floats, int8 GEMV, download, sync |

The two GPU calls are ~94 and ~60 us each. The kernels inside them are
~42-55 and ~15-27 us by the Nsight durations of record section 5, so each
call carries an estimated ~30-60 us that is not kernel: two copies, a launch
and a blocking wait. Pinned staging does not reduce it (record section 9b,
and the 21 September record). The CPU work between the two calls, ~3.2 ms,
runs at one layer's worth of parallelism at a time.

Every DeltaNet layer today does GPU -> CPU -> GPU. The point of this plan is
to make it **one GPU call per layer**: upload the normed hidden vector, run
the whole layer on the device, download the 2048-float result.

## Model facts the design rests on

From `c/qwen36.c` (`deltanet()`, the config checks) and the 35B snapshot:

- `vh` = 32 value heads, `vk` = 16 key heads (`rep` = 2), `kdim` = `vdim` = 128,
  `convk` = 4, hidden = 2048, `conv_dim` = 8192, `value_dim` = 4096.
- Recurrent state per layer: 32 x 128 x 128 f32 = 2 MiB; conv ring 8192 x 3
  f32 = 96 KiB. 30 layers: ~63 MB. Today in `m->DN_rec` / `m->DN_conv`.
- The state is read or replaced outside `deltanet()` in four places: the
  snapshot save and restore (`c/qwen36.c` ~3487, ~3524), the reset (~3548)
  and the segment sessions, which swap `m->DN_rec`/`m->DN_conv` pointers
  (~4926). Prefill (S > 1) runs `deltanet()` on the CPU and must keep doing
  so in this plan.
- `dnproj` and `dnout` are already device-resident int8 tensors
  (`G_dnp[layer]`, `G_dense[h]` in `c/qwen36_tier.c`), placed by auto-place.

## Expected gain, and how sure it is

Per layer on the GPU path, estimated: one upload + one download + one
sync (~30-60 us, the same overhead one call pays today), the two GEMVs
(~57-82 us), and five to eight small kernels (a/b GEMV, gates, conv,
l2norm, recurrence, gated norm) of a few us each; the recurrence reads and
writes 2 MiB of state per layer, ~8 us at the card's bandwidth. That is
~110-170 us per layer, **~3.3-5.1 ms per token against 7.8 today**, so a
gain of roughly **2.7-4.5 ms/token (12-20 %)**.

What would make it smaller, in the order I would expect it:

1. Launch cost on Windows (WDDM). Eight launches per layer are 240 per
   token; if each costs the host 5-10 us and the GPU runs dry between
   them, the small kernels cost more than their arithmetic. Mitigated by
   fusing (stage 2 targets five kernels per layer) and, if needed, one
   CUDA graph per layer (the backend already uses graphs for the experts,
   `COLI_CUDA_GRAPH`).
2. The first token after a prefill pays a full state upload (~63 MB, a
   few ms), and a prefill after decode pays a download. Per turn, not
   per token.
3. VRAM: ~63 MB of state plus ~16 MB of f32 a/b weights come out of the
   expert tier's budget: at ~2.3 MB per resident expert (6859 in the
   tier's ~16 GB), about 30-40 fewer resident experts out of ~6860.
   That changes the hit rate slightly, and **it changes residency between
   the arms of an A/B**, which `misure-envab.ps1` refuses by design. Stage 3
   has to reserve the same VRAM in both arms (see there).

## Numerics

The GPU kernels will not be bit-identical to the CPU loops: different
reduction order in the l2norm, the recurrence dot products and the norm,
and the device `expf`. The recurrence carries its state across tokens, so
differences can grow over a generation. Consequences:

- Every stage keeps the CPU path as the default and puts the GPU path
  behind an opt-in variable (`COLI_DN_GPU=1`, name to confirm) until it is
  validated.
- Kernel tests compare against the CPU code with stated tolerances, and
  include a multi-token recurrence run (e.g. 512 steps on random data)
  that reports how the relative error grows, not just one step.
- On the real model the generated text will probably differ from
  `4a000cc01f310b64` at some point. Acceptance cannot be "text identical";
  it needs a quality check: `PPL=1` with a reference file against the CPU
  path (same prompt, same tokens), plus the tiny-model token check the CI
  already runs (`tools/make_qwen36_tiny.py`, 16/16 on the CPU path).

## Stages

### Stage 0 — confirm the budget (measurement only, no code)

One profiled run (`COLI_CUDA_PROFILE=1`, keep-alive on, heat table) to read
the h2d / kernel / d2h / residue split of the `dnproj` and `dnout` calls at
full clocks. The 21 September split was taken without the keep-alive. If
the non-kernel share per call turns out well below ~30 us, the gain above
shrinks and this plan should be reconsidered before stage 1.

### Stage 1 — kernels and their tests (no engine change)

In `c/backend_cuda.cu`, CUDA only:

- `dn_ab_gates`: the f32 a/b GEMV (64 x 2048) fused with `beta = sigmoid(b)`
  and `g = -exp(A_log) * softplus(a + dt_bias)`.
- `dn_conv_silu`: depthwise conv (k = 4) against the device ring, SiLU, ring
  advance; one thread per channel.
- `dn_l2norm_rep`: repeat-interleave q/k from 16 to 32 heads, l2norm with
  eps inside the sqrt, q scaled by `kdim^-0.5`.
- `dn_recurrence_norm`: one block per value head: decay, `kv = k S`,
  `delta = (v - kv) * beta`, `S += k delta^T`, `o = q S`, then the gated
  RMSNorm of that head (it is per head over `vdim`, so it fuses).

A new `tests/test_deltanet_cuda.cu` runs each kernel and the whole chain
against the CPU loops (copied from `deltanet()` or shared through a header),
one step and 512 steps, with tolerances written in the test. Wired into
`make cuda-test` and `gpu-compile`. Measurable in isolation: a small bench
target for the per-layer chain, the way `bench_dense_gemv_cuda.cu` did it
for the GEMV.

### Stage 2 — one backend entry point per layer

`coli_cuda_deltanet_decode(handle, x_host, out_host)`: upload x, `dnproj`
GEMV into device qkvz, the stage 1 kernels, `dnout` GEMV, download out,
one sync. The handle owns the device state and the f32 a/b weights and
refers to the already-placed `dnproj`/`dnout` tensors, which must be on the
same device (refuse otherwise). Plus `coli_cuda_deltanet_state_upload` /
`_download` / `_zero`. Exported through `coli_cuda.dll` and the loader
table like the other entry points. Test: stage 1's test extended to the
entry point, including upload/download round trips of the state.

### Stage 3 — engine integration, opt-in

In `c/qwen36.c` / `c/qwen36_tier.c`, behind `COLI_DN_GPU=1`, decode only
(S == 1):

- One flag says who holds the current state. Every CPU reader or writer of
  `DN_rec`/`DN_conv` (prefill, snapshot save/restore, reset, segment session
  swap) calls a `dn_state_to_host()` first, and the next GPU decode uploads
  again. A missed site is a silent wrong answer, so this list gets a test
  that runs prefill -> decode -> snapshot -> restore -> decode on the tiny
  model with the flag on and compares against the flag off within tolerance.
- Any backend failure downloads the state and falls back to the CPU for
  good, with one line on stderr, like `[dnp] ... CPU from here on`.
- VRAM: when the variable is set to `0` the tier must still reserve the
  same bytes, so that an A/B keeps residency identical (`COLI_DN_GPU=0`
  with the reservation vs `=1`); unset means no reservation.
- A marker line, `[qwen36] DeltaNet decode on the GPU`, for
  `misure-envab.ps1`.
- `dn-split` keeps working: on the GPU path it reports the single call.

Validation before any merge: `make check`, the new CUDA tests on the card,
the tiny-model check with the flag on, and on the real model `PPL=1` flag
on vs off plus one `misure-envab.ps1` A/B (six pairs, heat table,
keep-alive on in both arms via `-AllowEnv`).

### Stage 4 — only if stage 3's numbers say so

Launch overhead: a CUDA graph per layer, or fewer kernels. Default on:
only after stage 3 is validated and measured, and as its own PR.

## What this plan does not cover

- Prefill on the GPU. It stays on the CPU; only the state crosses.
- HIP. The kernels are plain CUDA C and could be ported, but nothing here
  is tested on AMD.
- More than one GPU: `dnproj` and `dnout` on different devices are refused.
- Attention layers (10 of 40) and their `attnproj`/`attnout` round trips,
  which have the same shape of cost. If stage 3 pays off, the same pattern
  applies there next.
- The shared expert (3.6 ms/token on the CPU) and the CPU misses
  (4.1 ms/token), which are separate, larger items.
