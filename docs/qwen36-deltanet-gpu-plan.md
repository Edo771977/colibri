# Qwen3.6 DeltaNet decode on the GPU — plan

Status: **stages 0-2 done; stage 3a (engine, opt-in `COLI_DN_GPU`) in review, not yet measured on the card.** Written 28 September 2026 from the
measurements in `docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt`
(sections 5 and 8-10). Each stage below is a separate PR with its own tests
and, where it touches ms/token, its own A/B. Estimates are marked as such.

## Why

On the reference machine (RTX 4070 Ti SUPER, 7950X, Windows, `qwen36_i4_gs64`,
cap 256, heat table, keep-alive on) a decode token costs **22.93 ms**
(record section 10, ON arm). DeltaNet is 7.83 of it (ms/token, 30 DeltaNet
layers; `dn-split` and `dn-sub` of that arm):

| part | ms/token | where it runs |
|---|---|---|
| `qkvz` projection | 2.81 | GPU call: upload x, int8 GEMV, download 12288 floats, sync |
| `a`+`b` matmuls | 0.66 | CPU, one OpenMP region, 64 x 2048 f32 |
| gates + conv + SiLU | 0.43 | CPU (`dn-sub` conv: the beta/g loop and the conv) |
| l2norm + recurrence | 2.00 | CPU, 32 heads x 128 x 128 state |
| gated RMSNorm | 0.13 | CPU |
| `out` projection | 1.79 | GPU call: upload 4096 floats, int8 GEMV, download, sync |

That is ~94 and ~60 us per GPU call. Section 5's Nsight run put the kernels
alone at a 42 and 15 us minimum (benchmark 44-55 and 17-27 us); the same
run's engine medians were 84 and 30 us, without the keep-alive and with
no clock log (P3 is inferred from sections 5-6), and the 15/30 us row is
the 1024-row grid, dnout mixed with attnout. So each call carries an **estimated** ~30-60 us that is not
kernel: two copies, a launch and a blocking wait. Pinned staging showed no
measured gain (record section 9b, interval [-1.44, +2.34]; the 21 September
record measured it slower). The CPU work between the two calls is ~3.2 ms.

Every DeltaNet layer today does GPU -> CPU -> GPU. The aim is **one GPU call
per DeltaNet layer** on the decode path: upload the normed hidden vector,
run the whole layer on the device, download the 2048-float result.

This is feasible in the engine's structure: `deltanet()` is called from
`layers_forward_range()`; after it the residual, the pilot prefetch,
`post_ln` and the router all need the result on the host, and the MoE's own
GPU work (`qt_issue`/`qt_take`) starts and finishes inside `moe()`. No GPU
work is pending when the next layer's call starts.

## Model and engine facts the design rests on

- Dimensions (`c/qwen36.c`, `deltanet()` and the config checks; the
  record's 12288-row dnproj = 8192 + 4096, its binaries section): `vh` = 32 value heads, `vk` = 16 key
  heads (`rep` = 2), `kdim` = `vdim` = 128, `convk` = 4, hidden 2048,
  `conv_dim` = 8192, `value_dim` = 4096.
- Per layer: recurrent state 32 x 128 x 128 f32 = 2 MiB, conv ring 8192 x 3
  f32 = 96 KiB; 30 layers ~63 MiB, in `m->DN_rec` / `m->DN_conv`.
- Weights the device would need per layer besides the placed GEMVs:
  `in_proj_b`+`in_proj_a` (f32, 64 x 2048, one buffer since #59), the conv
  weights (8192 x 4 f32), `A_log`, `dt_bias` (32 each) and the norm weight
  (128).
- **Placement.** `dnproj` is placed through `qt_dnproj_init` into
  `G_dnp[layer]`; `dnout` through `qt_dense_init` into `G_dense[h]`, reached
  by `l->h_dnout` and `trunk_out_matmul`. They are separate offers, placed
  independently: each goes to the device with the most room when it is
  offered, heat pricing can place one and not the other, and an upload can
  fail. Nothing is placed unless the tier starts (`COLI_CUDA=1`, cap equal
  to the expert count). On the measured setup (one card, `COLI_GPUS=0`)
  all 30 of each are on the GPU (`dn-gpu` 3810/3810); on two cards many
  layers would be split. **So eligibility is per layer**: a layer takes the
  new path only if both its projections and its new state are on the same
  device; every other layer runs as it would without the flag (see stage
  3 for what a unit refused on price costs).
- **Prefill.** `qt_dnproj_matmul` already runs on the GPU for every prompt
  row (S > 1); only `dnout` is decode-only. The conv, recurrence and norm of
  prefill stay on the CPU in this plan.
- **Every place the recurrent state is read or replaced outside the
  decode step**, in `c/qwen36.c`: the pin state save and restore
  (`q36_pin_state_save` ~3487, `pin_restore` ~3524, used by serve at
  ~4118-4135; `pin_save` can run right after a one-row prefill),
  `reset_recurrent` (~3540, reached from `generate`, `consist_reset`,
  `tf_nll`/PPL and serve), and any S > 1 call of `deltanet()` itself (a
  prefill, including serve's prefix-reuse prefill that skips the reset,
  ~4096-4122). Inside `deltanet()`, `DN_DBG` dumps layer-0 intermediates.
  The segment adapter (`qwen36_segment_*`,
  ~4727-5019) swaps in per-session buffers and reads/writes them directly;
  it is CPU-only today (it refuses a non-CPU backend mask and advertises a
  CPU numeric class), and **this plan keeps it CPU-only**, with an assert.
  `qwen36_tier.c`, the tests and `kv_prefix` do not touch `DN_*`.

## Expected gain, and what could shrink it

Per eligible layer on the new path, **estimated**: one upload + one download
+ one sync (~30-60 us, what a call pays today), the two GEMVs (~57-82 us by
section 5's minima and benchmark), a handful of small kernels, and the
recurrence reading and writing 2 MiB of state. With 32 head-blocks the
recurrence does not use the whole card; ~5-15 us is a guess until stage 1
measures it. Total ~110-170 us per layer: **~3.3-5.1 ms/token against 7.83,
a gain of roughly 2.7-4.5 ms/token (12-20 %)**, if the round-trip estimate
holds (stage 0 checks it).

What could make it smaller:

1. **Launch cost on Windows.** Five to eight launches per layer are 150-240
   per token. If the GPU runs dry between them, the small kernels cost more
   than their arithmetic. Mitigations: fusing (stage 1 targets four kernels
   plus the two GEMVs), then graphs in stage 4 (see the stream note there).
2. **VRAM taken from the experts.** By allocator footprint: state 30 x (2 MiB
   + 104 KiB) = 63 MiB, a/b 30 x 544 KiB = 16 MiB, conv weights 30 x 136 KiB
   = 4 MiB, small vectors ~1 MiB: **~84 MiB, about 46 experts** at the tier's
   1.80 MiB per int4 gs64 expert, out of ~6860 resident in a ~12.1 GB expert
   budget. Expect a slightly lower hit rate and slightly more `cpu-miss`
   (roughly +0.1 ms/token, an estimate).
3. **State transfers per turn.** The first decode token after a prefill
   uploads the eligible layers' state (~63 MiB, a few ms); anything that
   reads it on the host downloads it. Per turn, not per token.

## Numerics

The GPU path will not be bit-identical to the CPU loops, and the recurrence
carries its state across tokens. Rules for every stage:

- The CPU path stays the default; the GPU path is opt-in (`COLI_DN_GPU=1`,
  name to confirm) until validated.
- Match the CPU where it is cheap: the l2norm computes its sum, sqrt and
  division in double with eps 1e-6 added to the sum; the gated RMSNorm
  accumulates in double with `c->eps` added to the mean; `softplus_f` has a
  threshold at 20 and uses `log1pf`; the a/b buffer is b rows then a rows.
  Double accumulation over 128 terms is cheap even at consumer FP64 rates.
- nvcc contracts FMAs by default (`--fmad=true`); state it in the tests'
  tolerances rather than fight it.
- **Deterministic kernels only** (no float atomics): `misure-envab.ps1`
  refuses a run whose text differs within an arm.
- The conv ring on the device has the host's exact layout (channel-major
  `[conv_dim][convk-1]`, oldest first, the raw projected qkv, `w[convk-1]`
  on the current token), so an ownership transfer is a plain copy.
- The decay `exp(g)` with g < 0 makes the recurrence contractive; the
  multi-step test should expect a bounded error and fail if it grows.
- Acceptance on the real model cannot be "same text". It is `PPL=1` with a
  reference file, flag on vs off, within a stated margin, plus the checks
  in stage 3.

## Stages

### Stage 0 — confirm the round-trip budget (measurement only)

`COLI_CUDA_PROFILE` is not the right tool: it pools all 81 dense calls per
device in one accumulator, turns off the expert CUDA graph and adds an event
sync per call. Instead, one run under Nsight Systems with the keep-alive on
and the heat table (`nsys profile -t cuda`, as in section 5), reading
`cuda_gpu_trace` for the `dnproj`/`dnout` kernel grids and the memcpy rows
by size (8 KiB up, 48 KiB down for dnproj; 16 KiB up, 8 KiB down for dnout),
and the host-side gaps between them. If the non-kernel share per call is
well below ~30 us, the gain above shrinks and this plan is reconsidered
before stage 1.

**Result (28 September 2026, record section 11).** Wall minus kernel per
call is 34.6 us for dnproj and 29.6 for dnout (unprofiled wall, profiled
kernel), 1.9 ms/token over the 30 layers: at the low end of the range,
not well below it, so stage 1 goes ahead. At least 90 % of the dnproj
kernels took within 1.2 us of their minimum (median 42.8 us). Between
the two calls of a layer the stream has no operation for a median 127.5
us (profiled), which holds the CPU part of deltanet, ~114 us a layer by
the timers of one run, plus the host ends of the two calls. Most of the
gain is that CPU work moving to the device, and most of that work is
l2norm + recurrence (2.0 of 3.43 ms/token, one timer for both): the
recurrence and l2norm kernels matter most. Re-estimated with that run's deltanet, 7.32
ms/token (~244 us a layer): ~100-125 us a layer, a gain of roughly
3.5-4.3 ms/token (an estimate; against section 10's 7.83 it would be
~4.1-4.8). On Windows the nsys log does not carry the engine's output;
the per-call wall comes from an unprofiled run of the same
configuration.

### Stage 1 — kernels and their tests (no engine change)

In `c/backend_cuda.cu`, compiled out under HIP (guard like the keep-alive's,
with HIP stubs that return 0):

- `dn_ab_gates`: the f32 a/b GEMV (64 x hidden) fused with `beta` and `g`.
- `dn_conv_silu`: depthwise conv against the device ring, SiLU, ring advance.
- `dn_l2norm_rep`: repeat-interleave q/k (`h / rep`), l2norm, q scaled.
- `dn_recurrence_norm`: per value head: decay, `kv = k S`, `delta`,
  `S += k delta^T`, `o = q S`, then that head's gated RMSNorm (per head over
  `vdim`, so it fuses). Splitting v across blocks is an option if 32 blocks
  underuse the card; it costs the norm fusion.

Kernels take `kdim`, `vdim`, `convk`, `rep` as parameters (or refuse
unsupported values with a message), because the CI tiny fixture has
`kdim` = `vdim` = 8 and `conv_dim` 128. Test:
`tests/test_qwen36_deltanet_cuda.cu`, following the `test_*_cuda.cu`
convention, against a CPU reference copied from `deltanet()` into the test
(an nvcc test cannot include `qwen36.c`; the copy is compiled by nvcc's host
compiler, which is part of what the tolerances cover). One step, and 512
steps on random data with a bound on the error. Both the 35B shapes and the
tiny ones. Wired into `make cuda-test` and `gpu-compile`. A small bench of
the per-layer chain, like `bench_dense_gemv_cuda.cu`, gives stage 1's own
numbers.

**First run (28 September 2026, RTX 4070 Ti SUPER, #64).** The test
passed on all five shapes: errors 1e-7 to 2.5e-6 against the CPU copy, the
repeat byte for byte, no growth over 512 steps. The bench, 35B shape, 30
layers of state: 38.2 us per layer back to back, 48.7 with a synchronize
after each layer; alone, dn_ab_gates 7.35, dn_conv_silu 6.26 and
dn_rec_norm 29.0 us. The two small kernels do almost no work, so their
time is most likely the launch cost on that machine (not measured; the
bench now has an empty kernel for it), and the recurrence, one thread per
state column, used 32 blocks of 128 threads. So the gates moved into the
recurrence kernel (one launch less per layer) and each column's key rows
were split across four threads (dn_head, #65: the plan's four kernels are
now dn_conv_silu and dn_head). The tolerances were tightened to 2e-6 -
5e-5, about 4-5 times the worst case of a host emulation of dn_head over 40
seeds (the first run's errors are the old kernels'); dn_head is checked by
its own first run.

### Stage 2 — one backend entry point per layer

`coli_cuda_deltanet_decode(handle, x_host, out_host)`: upload x, `dnproj`
GEMV into device qkvz, the stage 1 kernels, `dnout` GEMV, download out, one
sync. The handle owns the device state, the ring, the a/b and conv weights
and the small vectors, all allocated once (a `cudaFree` from growth waits
for the keep-alive's spin), and refers to the already-placed `dnproj`/`dnout`
tensors, refusing if they are on different devices. Destroying a handle
yields to the keep-alive like `coli_cuda_tensor_free`. Plus
`_state_upload` / `_download` / `_zero`. The existing device pipe API
(`coli_cuda_pipe_*`, including a double-accumulating `pipe_rmsnorm_rows`) is
a precedent to reuse where it fits.

Exports: the new entries are resolved with `RESOLVE_OPT` in
`c/backend_loader.c`, so an older `coli_cuda.dll` still loads (a missing
mandatory symbol unloads the backend), and `qwen36_tier.h` gets the
`!COLI_CUDA` stubs. Test: stage 1's test extended to the entry point, with
state upload/download round trips, and a failure injected through the
backend's `fault_injected()` hook at each step.

**First run (28 September 2026, RTX 4070 Ti SUPER, #66).** Both tests
passed (one call: output 2e-5 tolerance; 64 calls: worst output 4.2e-6,
final state 2.6e-7, tolerance 5e-5). Bench, 35B shape, 30 layers, 200
tokens, keep-alive on: one decode call 120.4 us per layer, against 139.7
us for today's two `coli_cuda_matmul` calls alone -- without the CPU work
between them, about 114 us per layer by stage 0. Estimate, not measured in
the engine: 30 x (139.7 + 114 - 120.4) us, about 3.7-4 ms per token of the
21.2 ms step stage 0 measured (22.93 in section 10), which stage 3's A/B
has to confirm.

### Stage 3 — engine integration, opt-in

In `c/qwen36.c` / `c/qwen36_tier.c`, behind `COLI_DN_GPU=1`, decode only:

- **VRAM is charged, and co-located.** The state and weights are offered
  with `qt_trunk_offer` before `qt_init`, so they come out of the expert
  budget instead of the 1 GiB headroom. Because auto-place prices each
  offer on its own and puts it on the device with the most room, `dnproj`,
  `dnout` and the state of a layer are offered **as one unit**: one offer
  under a new component name replacing today's separate dnproj and dnout
  offers, with both placements looked up by that name (`qt_place_of`
  matches any offered name and `qt_init` subtracts every offer, so the
  tier API allows it). Consequences to accept or solve: an explicit
  `COLI_PLACE` has no word for the unit yet; the `[place] auto:` line would
  report the unit instead of dnproj; a unit refused on price loses today's
  two-call GPU path for that layer too (not only the new one); and qwen38
  shares the tier code. Only with `COLI_DN_GPU` set: unset, the offers stay
  as they are today.
- **Ownership per layer.** A flag per eligible layer says whether the
  device or the host holds its state. Every site listed above calls
  `dn_state_to_host()` for all layers first, and the S > 1 path of
  `deltanet()` itself requires host ownership (it does not rely on
  `pin_restore` having run first); the next GPU decode uploads again.
  `DN_DBG` forces the CPU path. The segment adapter asserts the flag is
  never set.
- **Failures.** A failure before the layer's state kernels have run (the
  first of them is `dn_conv_silu`, which advances the ring) leaves the
  device state at token t-1: download it, and that layer uses the CPU from
  here on, with one stderr line. A failure after them (the state has
  already advanced to t) or a failing download (sticky CUDA error) cannot
  be recovered: `reset_recurrent` (which also clears the prefix cache) and
  fail the request. Never run the CPU on top of a half-advanced state.
  "Fail the request" needs plumbing that does not exist: `deltanet()` and
  `layers_forward_range()` return void and `step()` has no error return, so
  a failure flag has to be added and checked by generate, serve, CONSIST
  and `tf_nll`. And `fault_injected()` gates only compute entry points
  today, not uploads or downloads, so testing the download branch means
  extending the hook.
- **A/B.** The OFF arm must carry the same VRAM offers, or residency and the
  `[place] auto:` line differ between arms and `misure-envab.ps1` refuses
  the run. So `COLI_DN_GPU=0` means "offer and reserve, run on the CPU",
  unset means neither, and the A/B is `=0` vs `=1`. `misure-envab.ps1`
  needs an `-OffValue` parameter for that (its OFF arm removes the variable
  today). A marker line, e.g. `[qwen36] DeltaNet decode on the GPU: N/30
  layers`, names how many layers took the path; the `=0` arm must not
  print it, since the script refuses an OFF log that contains the marker.
- `dn-split`/`dn-gpu` keep meaning something: on a GPU-path layer the
  whole call goes in one slot, and `dn-gpu` counts it.

Validation before merge:

- `make check`; the stage 1-2 CUDA tests on the card.
- **Tiny model, on the card, by hand** (the CI job has no GPU and builds
  without CUDA). The CI token check runs with `COLI_DENSE_I8=0`, where
  nothing is placed, so it could never reach the GPU path. The tiny
  acceptance is a CUDA build, `COLI_CUDA=1`, cap 8 (the tier starts only
  with cap equal to the fixture's 8 experts), `COLI_PLACE` unset,
  `COLI_DENSE_I8=1`, `CONSIST=1` (CPU prefill against S==1 decode, with
  resets), flag on vs flag off within a tolerance, and the marker must
  read 6/6 layers (the fixture's 6 DeltaNet layers of 8): without that
  check, "within tolerance" also passes when every layer stayed on the
  CPU. The fixture is `c/tools/make_qwen36_tiny.py`.
- A test that runs prefill -> decode -> `pin_save` -> `pin_restore` ->
  decode, and decode -> serve's prefix-reuse prefill -> decode, with the
  flag on, against the flag off.
- Real model: `PPL=1` flag on vs off, and one `misure-envab.ps1` A/B
  (`COLI_DN_GPU` 0 vs 1, six pairs, heat table, keep-alive on in both arms
  via `-AllowEnv`).

**Stage 3a as built (#67).** Where it departs from the text above:

- **Three offers, not one unit.** `COLI_DN_GPU` set (0 or 1) adds a
  `"dnstate"` offer per eligible layer (`qt_dn_state_bytes`), after the
  other offers, and leaves the dnproj/dnout offers as they are. A layer
  takes the new path only when all three landed on one device; otherwise
  it keeps the two-call path, with a line. This keeps the existing
  `COLI_PLACE` names and the `[place]` line unchanged and never costs a
  layer its existing GPU path. The prices:
  - an explicit `COLI_PLACE` must now also name `dnstate=<dev>`, or no
    layer takes the path (0/N, with a line per layer);
  - auto-place puts each offer on the device with the most room, so on
    two or more GPUs a layer's three usually split and most layers stay
    on the two-call path, their state still reserved (about 3 MB each);
    on one tight card the same waste happens when dnproj or dnout is
    refused on price and the state is not. Placing dnstate on its layer's
    projection device is left for stage 3b;
  - the offer charges the handle's payload (`qt_dn_state_bytes`, 2.83 MiB
    a layer on the 35B), like the other trunk offers, not the allocator's
    rounding (4 MiB for one such cudaMalloc): about 35 MiB over 30 layers
    comes out of the 1 GiB headroom. `qt_dn_init` warns if the backend's
    `coli_cuda_deltanet_bytes` ever differs from `qt_dn_state_bytes`.
- **A failed decode or download stops the run** (state zeroed, exit 3)
  instead of failing the request: the plumbing for per-request failure
  comes with the server in stage 3b. A failed state upload is still
  recoverable -- the host copy is authoritative -- and sends that layer
  back to the CPU with a line.
- **SERVE=1 and DN_DBG ignore `COLI_DN_GPU`** (0 or 1) with a line, and
  reserve nothing (stage 3b). When the tier does not start (no CUDA, or a
  cap below the expert count) the flag does nothing either, without a
  line; `misure-envab.ps1` notices through the missing marker.
  The segment adapter refuses to run with the path on.
- The marker `[qwen36] DeltaNet decode on the GPU: N/M layers` prints only
  with N > 0; with N = 0 a different line says the CPU path stays.
  `misure-envab.ps1` has `-OffValue`, so the A/B is
  `-Var COLI_DN_GPU -OffValue 0`.
- Test: `c/tests/test_qwen36_dn_gpu.c`, on the fake backend (no GPU): the
  fake device runs the CPU `deltanet()` on its own copy of the state, and a
  sequence of prompts, decodes, pin snapshot and restore, prompt without
  reset and resets must match the flag-off run byte for byte, with the
  expected numbers of uploads and downloads; plus the refusals and the
  failure paths (a failed decode, a failed upload; the failed-download
  branch goes through the same `dn_gpu_fatal` but has no fake switch yet).
  The tiny-model and real-model checks above are still to run on the card.
  The marker says N/M; the A/B script only checks it is there, so read N.

### Stage 4 — only if stage 3's numbers say so

Launch overhead. CUDA graphs cannot capture on the legacy stream the dense
calls use today, and pageable copies are not capture-safe, so this needs a
non-blocking stream and handle-owned pinned staging — the staging that
measured no gain on its own (section 9b). Measured as its own PR, default
on only after that.

## What this plan does not cover

- Prefill's conv/recurrence/norm on the GPU, and the segment adapter.
- HIP: compiled out, stubs only.
- Layers whose `dnproj`, `dnout` and state do not fit on one device as a
  unit: they do not take the new path (stage 3 says what they lose).
- Attention layers (10 of 40) and their `attnproj`/`attnout` round trips,
  which have the same shape of cost; the same pattern could follow.
- The shared expert (3.6 ms/token on the CPU) and the CPU misses (4.1
  ms/token), separate and larger items.
