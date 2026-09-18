# Qwen3.6-35B-A3B on colibri

`c/qwen36.c` runs [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(35B total / ~3B active, Apache 2.0) — a hybrid architecture: 25% Gated
Attention layers, 75% Gated DeltaNet (linear attention) layers, each followed
by a streamed MoE block (256 experts, top-8 + 1 shared). Dense weights stay
resident; routed experts stream from the container through an LRU + pinned
cache. Development notes live in `docs/qwen36-phase01.md` /
`qwen36-phase02.md`.

Architecture-identical checkpoints (same config geometry, e.g.
KAT-Coder-V2.5-Dev) run on this engine unchanged.

## Quickstart

Pre-converted containers (int4 experts, self-contained, ~20 GB):

```sh
# group-scaled int4 (gs64) — recommended, see "Which container" below
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir ~/Models/qwen36_i4_gs64

# per-row int4
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4 --local-dir ~/Models/qwen36_i4
```

or convert the original bf16 checkpoint yourself (~70 GB download):

```sh
python3 c/tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out ~/Models/qwen36_i4_gs64 --gs 64
```

Build and chat:

```sh
make -C c qwen36
COLI_MODEL=~/Models/qwen36_i4_gs64 ./c/coli chat
```

`coli` reads the model's `config.json` and matches its `model_type` against the
family registry (`qwen3_5_moe` / `qwen3_5_moe_text` — an exact match, so other
Qwen architectures are not claimed by this engine), picks it, and drives it over the serve protocol — `coli web` and
`coli serve` (OpenAI-compatible API) work the same way.

Direct invocation without the gateway:

```sh
SNAP=~/Models/qwen36_i4_gs64 TOK=~/Models/qwen36_i4_gs64/tokenizer.json \
N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

## CPU prefill batching

On AVX2/FMA CPUs, dense int8 projections process two prompt rows per weight
decode.  The same kernel is used by attention, the router, DeltaNet projections,
and the CPU shared expert.  `S=1` decode keeps the original four-register GEMV,
and non-AVX2 builds retain the established per-row implementation.  The shared
expert additionally batches gate/up/down calls across prompt rows with at most
32 MiB of temporary activations; the CUDA expert tier keeps its per-token shared
work so it can continue to overlap the in-flight GPU groups.

Both optimizations are bit-exact and on by default.  For controlled A/Bs,
`QWEN_DENSE_BATCH=0` restores per-row dense-int8 GEMVs and
`QWEN_SHARED_BATCH=0` restores per-row shared-expert calls.  A positive
`QWEN_SHARED_BATCH=N` limits each shared chunk to `N` rows.

Requirements: ~30 GB RAM for comfortable expert caching and NVMe storage for
the container. The default build is CPU-only; `make -C c qwen36 CUDA=1` adds
the optional CUDA VRAM expert tier documented in
[`qwen36-cuda-tier.md`](qwen36-cuda-tier.md). The same tier builds for AMD
through ROCm with `make -C c qwen36 HIP=1 HIP_ARCH=<gfx>` (for example
`HIP_ARCH=gfx1151`, with `ROCM_HOME` and `HIPCC` pointing at the toolchain):
measured on a Ryzen AI MAX+ 395, output bit-identical to the CPU path and 2.4x
faster than CPU-only (#1502).

## The expert kernel

Routed experts run through `c/expert_ffn.h`, a header shared with the other
MoE engines rather than a set of GEMVs of this engine's own. Three things
changed with it, measured on the real gs64 container at full residency on an
8-core AVX-512 box (61 GB, DDR5-5600, 58 GB/s DRAM read):

- **The int4 stays int4.** The engine used to unpack every expert to int8 at
  load; the kernel keeps the container's nibbles, repacked once into a planar
  layout where `and 0x0F` yields 32 elements in order and `srli 4` the other
  32, so a block costs no unpack instruction. Half the bytes per token and half
  the expert-cache RSS: peak RSS at cap 256 went from 25 GB to 15 GB.
- **A layer is a unit of work.** gate+up share one pass over the activation,
  and threads split (expert, row-chunk) items: two OpenMP regions per layer
  instead of 3 x top-k. A prompt row routed to an expert another row already
  used reads that expert from cache, not DRAM.
- **Same tokens.** Activations stay f32 (the kernel also has an int8
  activation mode, `mode 1`, not wired in here: same policy as `IDOT`). The
  only difference from the old path is the accumulation order inside a dot;
  a 1024-token greedy decode on the real container is byte-identical, and CI
  pins old vs new on a tiny int4 fixture at caps 1, 2, 8 and 16.

Over a 1024-token greedy decode at cap 256 (same prompt, byte-identical
text): 12.8 -> 15.7 tok/s, MoE per token 34 -> 20 ms on average and 30 -> 17
ms in the last windows, peak RSS 29 -> 17 GB. Of the 20 ms, 11 are the kernel
(the DRAM floor for the int4 bytes is 9) and 6 are the residual misses of a
97.6% hit rate, fetched one at a time; that fetch is the next thing to
overlap, not this kernel. `tests/test_expert_ffn` holds the numerics.
`QWEN_EXPERT_KERNEL=0` restores the int8 path for A/Bs. The CUDA expert tier
keeps its own path: it uploads the pair-layout int4 and computes misses from
the int8 copy.

## Is the CPU half sync-bound or stream-bound?

On a CUDA-tier host most of a decode token is CPU, and the three biggest CPU
entries are the shared expert, the DeltaNet output projection and the router.
Measured on a Ryzen 9 7950X + RTX 4070 Ti SUPER at 56.4 ms/token: 9.3, 9.0 and
2.7 ms. Those kernels are 120, 30 and 40 OpenMP regions per token, and also
126 MB, 252 MB and 21 MB of int8 weights per token. A MAC count separates
neither: it does not model memory at all, which is how a "6-15 ms of headroom"
estimate gets written for a kernel that might be at its DRAM floor already.

`tests/bench_qwen36_decode_omp` settles it on the host in front of you, with no
model file and no GPU, in a few seconds:

```sh
make -C c tests/bench_qwen36_decode_omp ARCH=native
OMP_NUM_THREADS=<physical cores> ./c/tests/bench_qwen36_decode_omp
```

It runs the shipping shapes through the engine's own kernels, and prices the
three things a row can be spending time on: a parallel region entry, a barrier
inside a region already open, and a streaming read of every weight byte in ONE
region — the machine's real bandwidth. Each row is then charged its sync points
and its bytes, and what is left over is printed. A residual near zero on every
row means the model holds.

### What it found on the 7950X

| | team of 8 | team of 16 |
|---|---|---|
| empty parallel region | 32.7 us | 53.5 us |
| barrier inside a region | 35.2 us | 69.6 us |
| streaming read, one region | 52.2 GB/s | 54.6 GB/s |

The bandwidth is exactly what dual-channel DDR5-5200 should give, and every
kernel's residual lands on it. **The host is not stream-bound: it is paying tens
of microseconds per synchronisation.** A healthy OpenMP runtime charges single
digits — the same benchmark on a 4-core Linux box measures 2.8 us and 0.8 us. A
decode token crosses roughly 250-300 regions, so at this floor the *runtime*
spends 13-16 ms of the 56.4. That is the single largest addressable item in the
token, larger than anything SIMD width can reach, and the reason
[PR #8](https://github.com/Edo771977/colibri/pull/8)'s AVX-512 kernels measured
no gain at all.

It is a Windows/MinGW `libgomp` property, not a CPU one: the same build also
reports `libgomp: Affinity not supported on this configuration`, so
`OMP_PROC_BIND`/`OMP_PLACES` cannot even be tried, and `OMP_WAIT_POLICY=active
GOMP_SPINCOUNT=200000` moves neither number. **Before reshaping any kernel on
such a host, try another OpenMP runtime** (clang with LLVM's `libomp`, which has
a real Windows implementation) and re-run the benchmark. No kernel change
recovers what the runtime is spending.

### Two things the benchmark had to be fixed for

- **A "streaming ceiling" measured one region per matrix is not a ceiling.** The
  first version read the reference bytes in 190 separate regions; at a 54 us
  floor that is 10 ms of pure fork/join, and the reference came out *slower*
  than the kernels it was meant to bound — so it printed "stream-bound" for a
  host that is nothing of the sort.
- **Fusing regions does not remove synchronisation, it converts it.** Three
  `parallel for` regions become one region plus barriers, and on this host a
  barrier costs MORE than a whole region entry. The benchmark prices both and
  charges each row for what it actually spends.

### What that means for `QWEN36_SHARED_FUSE`

The fused shared expert is one region and one barrier per layer against three
regions: worth having where a barrier is the cheaper price, close to a wash
where it is not. It is opt-in for exactly that reason. Two things it must not
do, both learned the hard way:

- **Keep one weight stream per thread.** A fused kernel that computes gate row
  *i* and up row *i* together keeps two streams live per thread; on the 7950X at
  16 threads that took the same 120 MB from 71 GB/s to 15 and lost 1.9 ms while
  removing 80 regions worth 4.3. `qwen_shared_fused_row` hands each thread a
  contiguous range of hidden units and walks the two matrices in turn — which
  also lets the thread that produced `g[i]` and `u[i]` combine them, with no
  barrier.
- **Do not pay for a barrier the region exit already provides.** The final
  worksharing loop carries `nowait`; at 70 us a redundant barrier is 2.8 ms per
  token on a 40-layer model.

## Which container?

The gs64 container carries one scale per 64-weight group instead of one per
row. On GLM, per-row int4 was the root cause of think-mode loops and
never-terminating generations (#455), and group scales fixed them in
controlled A/Bs — with `moe_intermediate_size=512`, Qwen's rows are short, so
per-row quantization error concentrates the same way. The gs64 container costs
~1.7 GB more on disk and a few percent on cold-start; warm decode speed is the
same or slightly better.

## Which checkpoints, and what the banner calls them

Two Qwen checkpoints declare `model_type: qwen3_5_moe_text` and resolve to
this engine:

| checkpoint | layers | experts | hidden | banner |
|---|---|---|---|---|
| Qwen/Qwen3.6-35B-A3B | 40 (10 attention) | 256, top-8 | 2048 | `Qwen3.6-35B-A3B · 35B MoE` |
| Qwen/Qwen3.8-2.4T-A95B | 92 (23 attention) | 512, top-10 | 8192 | `Qwen3.8-2.4T-A95B · 2.4T MoE` |

The registry names a checkpoint by its geometry (`display_variants` on the
`qwen36` descriptor), so the banner says what is on disk. A config that
matches neither, a tiny fixture for instance, is named by its own
`model_type` and measured geometry rather than by a sibling's parameter
count (#1045).

The 2.4T checkpoint is **architecture-identical** to the 35B: same layer
pattern, every engine guard holds, and the registry's planner puts its KV
cache at 1.44 GiB for 8k context and 46 GiB at the 256k maximum, with a
context-free DeltaNet state of 0.55 GiB. What this engine cannot do for it
is hold the experts: the warmstart keeps every expert in RAM by design (see
`--ram` below), which is ~1.4 TB of int4 for 2.4T. Serving it needs the
disk-streaming design, not this one. The conversion and the geometry checks
are in place so that work starts from a verified shape, not from a guess.

### The converter's tensor contract

Both checkpoints ship experts **fused** per layer (`mlp.experts.gate_up_proj`,
`mlp.experts.down_proj`), a one-layer multi-token-prediction head (`mtp.*`,
`mtp_num_hidden_layers: 1`), and the 35B additionally a vision tower
(`visual.*`). `tools/qwen36_tensor_kinds.py` classifies every tensor name
before the first shard is read: layer tensors are converted, `mtp.*` and
`visual.*` are skipped **on purpose** and reported with a count, and a name
the contract does not know stops the conversion. A converter that silently
drops what it does not recognise produces a container that loads and is
quietly missing a tensor; this one refuses instead (the GLM-5.3 precedent).
`tests/test_qwen36_tensor_kinds.py` pins the contract to both real indexes.

### Validating the 2.4T shape without a single weight

`tools/make_qwen36_tiny.py --geometry qwen38-2p4t` builds a fixture with the
2.4T's structural numbers at toy widths -- 92 layers, interval 4, 512 experts
top-10, 16:1 attention heads, 8:1 DeltaNet heads -- and rewrites the shard
into the real layout: fused experts plus an `mtp.*` head. The converter must
split the one and skip the other, and the engine must match the transformers
reference token for token. CI runs it at cache capacities 1, 2 and 512, and
under ASan/UBSan. Locally:

```sh
cd c && make qwen36
python3 tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 \
        --out q24 --ref-mode full --emit-ref q24/ref_full.json
python3 tools/convert_qwen36.py --model q24 --out q24_c --ebits 8
COLI_DENSE_I8=0 SNAP=q24_c ./qwen36 512 8 q24/ref_full.json
```

(`--seed 3`: the default seed collapses this geometry's reference to one
repeated token, which a shape error could still reproduce; seed 3 yields
twelve distinct tokens over sixteen.)

## `--ram` is not honoured by this engine

The engine reads no `RAM_GB`: `grep -c RAM_GB c/qwen36.c` returns 0, and passing
`--ram` changes nothing. Said here rather than left to be discovered, because a
flag that appears to work and does not is worse than one documented as
unsupported.

What sizes the expert cache instead depends on where the experts live. On the
CPU path they stream from the container on demand, through the LRU described at
the top of this page, and `--cap N` is the direct lever on how many slots per
layer that cache holds. On the CUDA expert tier (#713) the hot set is resident
in VRAM and `CUDA_EXPERT_GB` decides its size. Neither path consults the RAM
budget.

(Earlier revisions of this section said qwen36 does not stream experts at all,
which contradicted the description at the top of the page and was wrong for the
CPU path: `c/qwen36.c` reads experts on demand with `pread` plus
`posix_fadvise(DONTNEED)` and caches them LRU. Reported in #1444.)
