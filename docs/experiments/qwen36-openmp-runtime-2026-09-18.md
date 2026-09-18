# The OpenMP runtime was the largest item in a Qwen3.6 decode token

A Qwen3.6 decode token on a CUDA-tier host crosses a few hundred `#pragma omp`
regions. On this Windows machine each one cost **54 us to enter and 70 us to
barrier**, on a team of 16. That is 13-16 ms of a 50 ms token spent inside the
threading runtime, before any arithmetic.

Nothing was failing. The engine produced correct tokens the whole time, and the
profile pointed at the kernels: shared expert 8.7 ms, DeltaNet `norm+out` 8.5,
router 2.7. Those numbers were real, but most of what they contained was not
arithmetic and not memory traffic.

Building the same source with a toolchain that links LLVM's `libomp` instead of
MinGW's `libgomp` took the token from **50.2 to 32.2 ms**, median **13.38 to
20.41 tok/s**. No line of engine code differs between the two arms.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 9 7950X, 16C/32T, 2 CCDs, AVX-512 |
| RAM | 64 GB DDR5-5200, 4 DIMMs — **52.5 GB/s** measured streaming read |
| Storage | Silicon Power UD90 4 TB, PCIe 4.0 x4, NTFS |
| GPU | RTX 4070 Ti SUPER 16 GB, driver 616.64, CUDA 13.4 |
| OS | Windows 11 |
| commit | `4bc62322b01cb0dcf33e1938701037858296decc` |
| baseline | MSYS2 UCRT64 gcc 16.2, `-static`, MinGW libgomp |
| trial | MSYS2 CLANG64 clang 22.1.8, LLVM libomp 22.1.8 |

`coli_cuda.dll` is the same nvcc+MSVC build in both arms and is loaded at
runtime, so the GPU side of the engine is literally identical code.

## The mechanism, measured on its own

`c/tests/bench_omp_sync.c` prices exactly two things and nothing else: entering
an empty parallel region, and a barrier inside a region already open. One source
file, four toolchains, same machine, `OMP_NUM_THREADS=16`:

| runtime | empty region | barrier in an open region |
|---|---|---|
| MinGW `libgomp` (gcc) | 54.0 us | 69.5 us |
| LLVM `libomp` (clang) | **3.7 us** | **1.4 us** |
| LLVM `libomp` (`cl /openmp:llvm`) | 8.1 us | 1.5 us |
| VCOMP (`cl /openmp`) | 1.9 us | 1.1 us |

Three independent runtimes land within an order of magnitude of each other and
`libgomp` sits 15-50x above all of them. Its price is also **linear in team
size** — 12.0 / 25.0 / 34.0 / 53.5 us for a region and 6.5 / 20.0 / 35.5 / 70.5
for a barrier at 2 / 4 / 8 / 16 threads, about 4.4 us per thread. That is the
shape of a runtime that wakes its team one thread at a time, and it explains why
`OMP_WAIT_POLICY=active GOMP_SPINCOUNT=200000` moves neither number: nothing is
sleeping, the signalling is serial.

The benchmark reports the quietest of seven samples beside the median precisely
because these two numbers are thread wake-up latency and a busy desktop inflates
them. Here they agree to within 0.8% on a machine with other applications open.

## What moved

Per-phase, cold run of each arm:

| ms/token | gcc + libgomp | clang + libomp |
|---|---|---|
| deltanet | 23.7 | **14.4** |
| &nbsp;&nbsp;in_proj (GPU) | 10.4 | 6.3 |
| &nbsp;&nbsp;causal conv | 1.9 | 0.5 |
| &nbsp;&nbsp;l2norm + recurrence | 2.9 | 2.0 |
| &nbsp;&nbsp;gated norm + out_proj | 8.5 | 5.5 |
| attention | 8.8 | **6.5** |
| MoE total | 13.9 | **8.2** |
| &nbsp;&nbsp;shared expert | 8.7 | 4.0 |
| &nbsp;&nbsp;router | 2.7 | 0.6 |
| lm_head (GPU) | 3.5 | 2.9 |
| **total** | **49.9** | **32.0** |

The three phases the hypothesis named — shared expert, router, gated norm +
out_proj — moved by the predicted amounts (predicted ~2.5 / ~0.5 / ~5 against
4.0 / 0.6 / 5.5 measured). The router is the clearest case: 20 MB of weights and
40 regions per token, so almost all of its 2.7 ms was the regions, and almost
all of it went away.

## The variable is one field and two things

Switching to clang changes the compiler **and** the OpenMP runtime. The manifest
declares that honestly as a single combined variable rather than pretending the
runtime moved alone.

What separates them is the per-phase table plus the standalone sync benchmark.
A compiler difference would show up as a broad shift across all CPU work; what
actually happened is that the phases made of many small regions collapsed while
the rest moved much less, and the direct measurement of region and barrier cost
accounts for the size of the collapse. A reader who wants the runtime isolated
from the compiler can get it on Linux, where `make OMP_RUNTIME=llvm` swaps the
library at link time with gcc still doing the compiling.

## Limitations, stated

- **`prompt_hash` is not a digest.** `misure.ps1` builds its own prompt and is
  not in the repository, so there is nothing to hash. Every sample here came
  from the same script, model and invocation, and the engine reports the same 56
  prompt tokens in all eight runs; that is the strongest cross-run pin
  available, and it is weaker than a digest.
- **The commit measured is not the PR head.** These runs are at `4bc6232`; the
  branch later gained review fixes (the scratch blocks moved from file statics
  into `Model`, and `q36_avx512_on()` moved out of the per-row hot path). Both
  are perf-neutral or marginally favourable by construction — same reuse
  pattern, strictly less work per row — but they were **not re-measured**.
- **The two GPU rows moved and the hypothesis did not predict it.** `in_proj`
  10.4 -> 6.3 and `lm_head` 3.5 -> 2.9. Those are ~30 synchronous round-trips
  per token whose measured time includes the CPU-side wait, so a team of 16
  churning at 54 us per region plausibly competed with the driver's own threads
  for cores. That is a hypothesis, not a measurement, and if it is right then
  part of what looked like a GPU ceiling was not.
- **No oracle exists for the 35B.** Cross-arm agreement is token-level, never
  byte-level: two compilers vectorise the scalar code outside the intrinsic
  kernels differently. The gates are the tiny-fixture oracle (16/16 for both
  toolchains on this commit), the byte-identity of both engine switches, ASan +
  UBSan, and 200 tokens of real output read by hand.
- **Four samples per arm, one machine, one prompt.** Enough to separate a 1.5x
  effect from this machine's drift — the spread within the gcc arm is 12.2-14.5
  and within the clang arm 19.0-21.1, and the arms do not overlap — but not a
  throughput characterisation.

## What this does not say

It does not say clang is faster than gcc. It says that on **this** host the
MinGW `libgomp` build charges tens of microseconds per synchronisation and the
LLVM `libomp` build charges single digits, and that a streamed-MoE decode token
crosses enough regions for that to dominate everything else. A host whose
`libgomp` behaves normally would see little or none of this. The way to find out
is `make -C c bench-omp-sync`, which needs no model and takes seconds.
