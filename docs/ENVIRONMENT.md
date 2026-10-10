# Environment Variables

Reference for the environment variables read by the colibrì engine.

**Generated from `dev @ def8419`** by scanning every `getenv()` / `getenv_utf8()` site in `c/*.c`, `c/*.h`, `c/*.cu` and `c/*.mm`. Defaults and behavior are taken from the source; see [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate this after the code changes.

## Which program reads these?

**There are seven engine binaries, and they do not share a knob set.** The main
engine `c/colibri` (built from `c/colibri.c`, formerly `glm.c`) reads most of
what follows, but the sister engines read their own:

| Engine | Source | Its own variables |
|---|---|---|
| `colibri` | `c/colibri.c` | everything below except the three sections named for another engine |
| `kimi_k3` | `c/kimi_k3.c` | the `K3_*` family — see [Kimi K3 engine](#kimi-k3-engine-kimi_k3) |
| `inkling` | `c/inkling.c` | `INK_*`, plus `CTX_MAX`, `PIN_N`, `REP_PEN`, `GPU_DEV`, `NOGPU` — see [Inkling engine](#inkling-engine-inkling) |
| `qwen36` | `c/qwen36.c` | `QWEN_*`, `Q36_*`, its dense/CUDA-tier controls, and the `CACHE_ROUTE` family (VRAM tier over RAM cache) — see [Qwen3.6 engine](#qwen36-engine-qwen36) |
| `qwen38` | `c/qwen38.c` | `Q38_MAXT`, `Q38_EOS`, `Q38_NATIVE_FP8`, `Q38_NATIVE_BF16`, `Q38_PREFILL_BATCH`, `COLI_TIMERS` — see [Qwen3.8 engine](#qwen38-engine-qwen38) |
| `olmoe` | `c/olmoe.c` | `HOT`, `WIDE`, `SMOOTH`, `CONF_LIMIT`, `MAX_NEW`, `CHAT`, `EXPERT_DROP`, `WARMUP` — see [OLMoE engine](#olmoe-engine-olmoe) |
| `deepseek_v4` | `c/deepseek_v4.c` | `CTX`, the `V4_*` / `DSV4_*` families and the two `COLI_CUDA_*_BATCH` gates — see [DeepSeek V4 engine](#deepseek-v4-engine-deepseek_v4); note that the CUDA section below describes `colibri.c` knobs (`COLI_CUDA`, `CUDA_DENSE`, ...) which the V4 engine does not read — its GPU switch is `DSV4_CUDA` |

Setting an `INK_*` variable while running `colibri` does nothing, and vice
versa; nothing warns you about it. A few variables are genuinely shared because
they live in headers every engine includes (`COLI_USAGE`, `USAGE_SAVE`,
`COLI_USAGE_DECAY` in `route_trace.h`; `RANS_*` in `rans.h`;
`COLI_NO_OMP_TUNE` / `OMP_NUM_THREADS` in `omp_tune.h`).

You rarely export any of them by hand — the `coli` CLI and `openai_server.py`
translate most of their flags into these variables before launching the engine
(e.g. `--temp` → `TEMP`, `--ctx` → `CTX`). See [SETTINGS.md](SETTINGS.md) for
the flag → variable mapping. Export a variable directly only to reach a knob the
CLI doesn't surface, or to override what the CLI would set.

Format: `VAR` — default — effect.

---

## Common — everyday use

| Variable | Default | Effect |
|---|---|---|
| `RAM_GB` | `0` (auto ≈ 88% of free RAM) | RAM budget in GB for the resident/streamed expert working set. Higher → more experts stay hot → higher cache hit rate. Read by colibri, kimi_k3, glm53 and olmoe; on olmoe it sizes the expert cache once the dense weights are resident, and only when no `--cap` was given. |
| `CTX` | `4096` | Maximum context length (tokens) the KV cache is sized for. |
| `COLI_PREFILL_CHUNK` | `0` (off) | Run a long prompt through the layers in N-token slices instead of one pass. Every S-scaled activation buffer shrinks from prompt-sized to chunk-sized, which is the remedy when a long prompt exhausts CUDA scratch. Byte-identical output (verified at N=256). Skipped under an active MTP draft. **Cost:** a slice of 512 tokens already routes to essentially every expert of every layer (`P(miss) = (1-topk/n_experts)^N`), so each slice re-reads the whole non-resident expert set -- prefer the largest N that still fits your scratch. |
| `NGEN` | `256` (engine) | Max tokens to generate before stopping (stop tokens can end sooner). `coli --ngen` defaults to `1024`. |
| `COLI_TEMP` | `-1` (auto: `1.0` for chat/text, greedy elsewhere) | Sampling temperature. **`COLI_TEMP=0` = greedy/argmax = deterministic.** `TEMP` still works as a deprecated alias, but only if fully numeric: `$TEMP` is the temp-*directory* path on Windows and for the ROCm runtime (#509), so prefer `COLI_TEMP`. |
| `NUCLEUS` | `0.90` | Nucleus (top-p) mass kept when sampling. Slightly tighter than the official 0.95 because the int4 tail is noisy. |
| `TOPK` | `0` (off) | Top-k filter on the sampling distribution (`0` = no limit). |
| `TOPP` | `0` (off) | Top-p filter (`0` = use `NUCLEUS`). |
| `SEED` | unset → seeded from clock + PID | RNG seed for sampling. **Unset = different every run.** Set a fixed value for reproducible sampling. |
| `KVSAVE` | `1` (on) | Persist the KV cache to `<model>/.coli_kv` so a conversation reopens warm. `KVSAVE=0` disables save+load (lossless round-trip; does not change output). |
| `KV_SLOTS` | `1` | Number of independent KV conversation slots (1–16), used in serve mode. |
| `KV8` | `0` (off) | Store the MLA latent KV cache in fp8 e4m3 with a per-row scale: ~3.9× less KV RAM, and `.coli_kv` shrinks ~4× (saved as the v2 format; f32 v1 files are quantized on resume and rewritten). Adds DeepSeek-V3-class KV quantization noise to attention. CPU attention path only for now: the CUDA/Metal fused-attention fast paths read f32 KV rows, so under KV8 they fall back to the CPU consumer (native fp8 decode; a one-time notice is printed under `COLI_CUDA_ATTN=1`). Forces `COLI_CUDA_PIPE=0`. Native CUDA/Metal fp8-KV kernels are follow-up PRs. |
| `KV_TQ` | `0` (off) | Sub-byte MLA latent KV quantization, mutually exclusive with `KV8` (`KV_TQ` wins). `KV_TQ=4` is the recommended tier: rotated-int4 codec (randomized-Hadamard rotation + Lloyd codebook, per-row radius as the scale), ~7.6× less KV RAM than f32. `KV_TQ=2|3|5|6` selects the PolarQuant codec at that bit width (`KV_TQ_POLAR=1` forces PolarQuant at 4 bits too). Requires power-of-two row widths (`kv_lora`/`qk_rope`; the GLM MLA shapes 512/64 qualify) — on a model whose shapes don't, the engine refuses to start rather than silently zeroing the cache. A value below the 2–6 grid (e.g. `KV_TQ=1`) is treated as the recommended `4` with a notice, not as the most aggressive tier. `.coli_kv` is saved as the v3 format; a file saved under a different KV mode, codec, or bit width is refused with an explicit message and the cache restarts. Same CPU-only status as `KV8`: GPU fast paths fall back to the CPU consumer; native kernels are follow-up PRs. Forces `COLI_CUDA_PIPE=0`. |
| `THINK` | `0` (off) | Emit a `<think>` reasoning block. `THINK=1` turns on visible reasoning. |
| `MTP` | on | Multi-Token Prediction (speculative draft head). `MTP=0` disables it. |

---

## Performance / tuning

| Variable | Default | Effect |
|---|---|---|
| `COLI_METAL` | off | Enable the Apple-Silicon Metal GPU backend. Requires a `make METAL=1` build. |
| `COLI_METAL_GEMM_MIN` | `16` | Minimum matmul rows to dispatch a GEMM to the GPU (below this, stays on CPU). |
| `COLI_METAL_SPIN` | off | Keep a GPU keep-alive spinner running (reduces dispatch latency; costs power). |
| `COLI_METAL_PREFILL` | `0` (off) | `=1` runs S>4 (prefill) attention on the GPU. Off by default because the CPU path is bit-exact; this one is an opt-in speed/exactness trade. |
| `COLI_GEMM_CHUNK` | `1` (on) | Split a large GEMM dispatch into ≤2^25-thread chunks. `=0` restores the single full dispatch (the pre-fix behaviour), so the fix can be A/B'd on one binary. |
| `COLI_RTOP8` | `1` (on) | Parallel top-8 router kernel. `=0` falls back to the serial one. |
| `COLI_METAL_RESSET` | off | `=1` uses an `MTLResidencySet` (macOS 15+) for the resident buffers instead of per-dispatch `useResource` calls. |
| `PIPE` | `0` (off) | Overlap expert disk-load with matmul via I/O worker threads. Byte-identical output; reorders I/O. `PIPE=1` opts in. |
| `PIPE_WORKERS` | `8` | Number of pthread loaders when `PIPE=1`, or the io-wq worker maximum per ring when `URING=1` (capped at 64). Tune to SSD queue depth and available cores. |
| `COLI_PIPE_BLOCK` | `0` (spin) | `=1` makes `pipe_wait` block instead of spinning. Spinning wins on an idle box; blocking is better when the cores are contended. |
| `PILOT_WORKERS` | `1` | Pilot loader threads on the blocking (non-`URING`) `PILOT_REAL` path, via an SPMC ring. `>1` raises NVMe queue depth. Clamped to [1,16]; `1` is byte-identical to the historic behaviour. |
| `PILOT_EVICT_GUARD` | `1` (on) | Keep pilot-prefetched experts from being evicted before they are used. `=0` restores plain LRU eviction (A/B). Also read by `olmoe`. |
| `RSS_GUARD_GB` | the resolved RAM budget | Resident-set ceiling (GB) checked every 16 emitted tokens; the cache is trimmed when it is crossed. Set explicitly to guard tighter or looser than the RAM budget. |
| `XEXP` | `0` (off) | `=1` runs ONE OpenMP region across all experts of a batch-union block instead of ~2 fork/joins per expert. Engages only at S=1 with an all-resident int4 block, off the speculation window, and with the int4-IDOT S=1 family (`I4S<=1`); output is byte-identical to that family. Measured +11.6% on a 2-socket 48-core Ice Lake, but neutral-to-negative on a 24-core box — hence opt-in. Measure on your host. |
| `COLI_KV_SHARE` | `0` (off) | `=1` lets a new serve slot adopt an existing slot's KV prefix instead of re-prefilling it. Measured on 6x5090 with a 675-token shared prefix: slot TTFT 50.1s → 1.7s, generated tokens identical. |
| `KVB_FLASH_MB` | `2048` | Ceiling (MB) for the one-shot `kvb_all` k/v reconstruction buffer in prefill attention (#768 — 30.1 GB at ctx 262144, and `cap_for_ram` reserved it permanently). Above the ceiling the reconstruction is tiled with an online (flash-style) softmax: same rebuild total, ~tile-sized transient, output may differ from one-shot by rounding (same divergence class as the CUDA/Metal attention arms). `=0` disables tiling (always one-shot). DSA-selected rows always take the one-shot path. |
| `KVB_TILE_MB` | `512` | Tile size (MB) for the tiled reconstruction above. |
| `KVB_FLASH` | unset | `=1` forces the tiled path at any size, `=0` forces one-shot — overrides the `KVB_FLASH_MB` trigger (A/B switch). |
| `COLI_GROUP_ASYNC` | `0` (off) | `=1` issues and collects CUDA expert groups asynchronously so CPU and GPU overlap at decode (S≤4). |
| `COLI_DISKCLASS_WINDOW` | see source | Recency window (in ticks) for the DISK-CLASS heat statistic. |
| `URING` | `0` (off) | Linux-only queued expert I/O. `URING=1` implies `PIPE=1`, forces cold reads through io-wq (`IOSQE_ASYNC`), replaces blocking loader pthreads and spin waits with batched SQEs/CQEs, and batches `PILOT_REAL` loads on a separate ring. Use `DIRECT=1` for cold NVMe to avoid page-cache copy/readahead limits. Fails clearly if the kernel denies io_uring; incompatible with `COLI_MMAP=1`. |
| `DIRECT` | `0` (off) | Use `O_DIRECT`/unbuffered reads for expert slabs. **Drive-dependent — measure it on your hardware.** On real NVMe with DRAM cache and headroom it is often a large win (measured +34% decode with `PIPE=1` on a Blackwell/Windows box, and 4.25→9.69 GB/s in iobench on a GB10); on QLC/DRAM-less drives or slow/virtualised disks it can be neutral to negative. Helps sustained NVMe; keeps the zero-copy GPU path. |
| `COLI_WIN_SYNC_DIRECT` | off | **Windows only, A/B switch.** The direct (`DIRECT=1`) handle is opened with `FILE_FLAG_OVERLAPPED` so concurrent `pread`s on the one fd per shard reach the drive in parallel; a synchronous handle serializes them on the file-object lock (queue depth 1). `=1` reopens the old synchronous handle for comparison. Pair with `IOBENCH_SHARED=1 iobench` (all threads on one fd, as the engine does). |
| `COLI_NO_OMP_TUNE` | off | **Kill-switch** for the OpenMP hot-thread tuning (`OMP_WAIT_POLICY=active` spin + proc-bind). Set `=1` when the CPU is mostly waiting on the GPU (Metal) so spin doesn't steal the shared power budget. Hybrid CUDA/CPU hosts may test an explicit user-owned policy only with controlled profiling; see [tuning.md](tuning.md#hybrid-cudacpu-openmp-override). |
| `COLI_NUMA` | auto in generated plans on multi-socket Linux; otherwise off | `COLI_NUMA=1` selectively interleaves large expert and dense slabs across NUMA nodes via `mbind` (raw syscall, no libnuma). Helps multi-socket hosts (+7–40% expert matmul); silent no-op on single-node or non-Linux. Explicit `COLI_NUMA=0` overrides the generated plan. |
| `MLOCK` | `-1` (auto: on for macOS) | Wire the streamed expert cache into physical RAM (`mlock`) to dodge the memory compressor. `0` off, `1` force. |
| `CAP` | unset | Expert-cache cap (slots/layer) when no CLI positional was given. Precedence: explicit `--cap`/positional > `CAP` > platform default > historic default (#379). Mainly for direct `./glm` use — `coli` users should prefer `--cap`. |
| `CAP_RAISE` | `1` (on); `0` on Metal + macOS + fast model volume (#379) | Let the engine raise the expert-cache cap above `topk` when RAM allows (bigger batches). `0` fixes the cap. When the platform-aware Metal cache default engages (F_NOCACHE probe measured the model volume fast), the *default* flips to `0` — auto-raise re-creates the Metal residency churn the minimal cache avoids. An explicit `CAP_RAISE` always wins. |
| `COLI_SSD_FAST_GBS` | `4.0` | Threshold (GB/s, measured F_NOCACHE, cached in `<model>/.coli_ssd` — see [The `.coli_ssd` probe cache](#the-coli_ssd-probe-cache) below) at or above which the model volume counts as "fast" for the platform-aware Metal cache defaults (#379). |
| `PREFETCH` | `0` | Prefetch depth for streamed experts. |
| `COLI_MMAP` | `0` | `mmap` the weights instead of read()-ing into slabs. |
| `PIN` | unset | Path to a `.coli_usage`/stats file; pins the hottest experts into a resident "hot store" at startup. **`PIN=auto`** seeds from the model dir's live `.coli_usage` (appended after every turn, so each restart's pin placement follows the accumulated real workload) with `stats.txt` as the fallback for a virgin model dir; neither present → no pin this run. |
| `PIN_GB` | `10.0` | Size budget (GB) for the pinned hot store when `PIN` is set. |
| `AUTOPIN` | `1` (on) | Auto-pin the hot store from usage history once ≥5000 selections are recorded. Automatic pinning is capped so it cannot reduce the adaptive LRU capacity that fits before pinning; explicit `PIN`/`PIN_GB` settings remain authoritative. |
| `REPIN` | `0` (off) | Live re-pin the hot store every N emitted tokens (RFC). |
| `PILOT` | `0` (off) | Router-piloted cross-layer expert prefetch. |
| `PILOT_REAL` | `0` (off) | Value-preserving real cross-layer prefetch loads (`PILOT_REAL=1` opts in). |
| `PILOT_K` | `6` if `PILOT_REAL` else `8` | Number of experts the pilot prefetches per step. |
| `PILOT_TWO` | `0` (off) | Two-step shared-expert-corrected router prediction for the pilot. |
| `COUPLE` | unset | Path to a coupling-score file driving cross-layer expert prefetch (#176). When set, `couple_load` reads it. |
| `COUPLE_K` | `8` | Top-K coupled experts per layer when `COUPLE` is set. |
| `COUPLE_D` | `1` | Coupling lookahead depth (`1` or `2`) when `COUPLE` is set. |
| `CACHE_ROUTE` | `0` (off) | Opt-in max-rank cache-aware MoE routing (pin∪LRU prefer within top-M). Also read by `qwen36`, where the VRAM tier outranks the RAM cache. See [CACHE_ROUTE.md](CACHE_ROUTE.md). |
| `ROUTE_J` | `2` | Sacred top ranks always taken when `CACHE_ROUTE=1`. |
| `ROUTE_M` | `12` | Max-rank window for resident preference when `CACHE_ROUTE=1`. |
| `ROUTE_P` | `0` | Cumulative mass window for CACHE_ROUTE (`0` = fixed M). |
| `ROUTE_ALPHA` | `1` | Scale gate mass of substituted experts before renorm (`1` = off). |
| `ROUTE_AGREE` | auto | Overlap% + KL vs true top-K; auto-on when `CACHE_ROUTE=1`. Alone it changes nothing and prints the meters (always 100% / 0). |
| `ROUTE_TRACE` | unset | If set to a path, logs every routing decision there (testing/analysis). |
| `ABSORB` | `-1` (auto: absorbed for S≤4) | MLA attention absorption mode. |
| `IDOT` | `1` | Integer dot-product kernel. `IDOT=0` uses exact f32 kernels (for A/B numerical checks). |
| `COLI_POLICY` | `quality` | Resource policy: `quality`, `balanced`, or `experimental-fast`. |
| `PROF` | `0` (off) | Performance profile: a startup header (machine + effective config), then per run — or per turn in serve mode, on stderr — forward-latency percentiles (p50/p90/p99/max), expert-I/O totals and cache-tier fill, phase shares of wall time, and a verdict naming the knob most likely to help on this machine. Output is additive; `PROF` unset changes nothing. |
| `COLI_NO_FUSED_PAIR` | `0` (off) | `=1` disables the fused-pair matmul kernel. |
| `DISK_SPLIT` | `0` (off) | `=1` splits the reported disk-load time across the draft/absorb/forward phases in stats. |
| `I4S` | per-ISA (`1` on AVX-512-VNNI / NEON-dotprod, `2` elsewhere) | Engage the int4 `IDOT` kernel for batch `S>=<n>`. `I4S=1` turns IDOT on at decode too: int8-quantized activations on expert matmuls — **not bit-identical** to the f32 decode path (measured 0.39% of scale on the gate output; the same numerics prefill already uses at `S>=2`, and the shipped default on AVX-512-VNNI, measured +5.5% end-to-end there). Attention projections always stay exact regardless. A default flip on AVX-VNNI awaits the quality ablation. |
| `IDOT_GS` | `0` (off) | **Opt-in** grouped planar IDOT for `fmt=4` (gs64/gs128) tensors: int8 activations with the K1 plane layout, one integer dot per scale group. Same numerics family as `I4S=1` — not bit-identical to the f32 grouped kernel, hence off until the ablation. Requires the planar family (AVX2 build, no GPU backend, no `XEXP`). Activation prints `[K1b]` once. |
| `SPEC_PIN` | `1` (on) | Speculation gate mode. `0` reverts to the legacy S-dependent speculation gates (#163). |
| `COLI_RAM_OVERCOMMIT` | off | `=1` overrides the "projected peak > MemAvailable → exit(2)" guard so a run that risks kernel OOM-kill is allowed to proceed. |

## The `.coli_ssd` probe cache

On Metal + macOS the engine's first startup measures the model volume with an
honest F_NOCACHE random-read probe (#379) and caches the result in
`<model>/.coli_ssd`, so every later startup reads a file instead of
re-measuring. Details that matter when you meet this file in the wild:

- **Cold-range steering.** `F_NOCACHE` bypasses the page cache only for pages
  that are not already resident, so probing a freshly-read (warm) shard would
  measure RAM, not the disk. The probe snapshots residency with `mincore` and
  reads only 4 MB windows that are entirely cold.
- **Contamination veto.** If the shard offers fewer than 64 MB of such cold
  windows, the measurement is refused: nothing is cached, one stderr line
  explains the deferral, the conservative (slow-storage) defaults hold, and
  the probe simply retries on the next, colder, startup. The same veto (with
  its own honest message) fires for an under-allocated shard — a sparse or
  still-downloading file whose "cold" pages are holes that would measure as
  RAM-speed zero-fill — and for a shard too small to ever offer 64 MB of
  probe windows. The probe measures the largest `.safetensors` in the dir.
- **Format (v2).** One line, `v2 <gbs> <st_dev>` — the measured GB/s and the
  `st_dev` of the model dir's volume at measurement time. The grammar is
  strict (plain digits, `0 < gbs < 1000`; no inf/nan/hex/exponents) and both
  readers — the C engine and `coli doctor`/`coli plan` — accept exactly the
  same bytes; anything else is ignored and re-probed, never trusted.
- **Volume identity (best-effort).** The cache is honored only while its
  recorded `st_dev` matches the model dir's current volume, so copying or
  rsyncing the model dir (including this hidden file) to another drive
  normally triggers a re-probe there instead of inheriting the old drive's
  number; doctor/plan likewise stop showing the stale value. This is
  best-effort, not an identity guarantee: macOS recycles `st_dev` values, so
  a cache carried to an external volume that happens to be assigned the old
  device id (e.g. drives attached one after another in the same slot) will be
  wrongly trusted until deleted. When in doubt after moving a model dir,
  delete `.coli_ssd`. True volume-UUID identity is a named follow-up.
- **Legacy upgrade.** A pre-v2 bare-number cache (written before steering
  existed, so possibly warm-contaminated) is re-measured once on the next
  startup and rewritten as v2.
- **Deleting the file is always safe** — the only cost is one ~0.35 s re-probe.
- **Split/mirror layouts:** the probe measures the **primary** model dir only
  (`COLI_MODEL`), and its verdict sets the cache defaults for the whole run.
  With `COLI_MODEL_DIRS`/`COLI_MODEL_MIRROR` spreading shards across drives of
  different speeds, that single-drive verdict is an approximation; revisit if
  mixed-speed split setups become common (the `COLI_DISK_WEIGHTS` startup
  probe already measures every drive, but feeds the split ratio, not the
  cache defaults).

---

## Dual-SSD streaming

| Variable | Default | Effect |
|---|---|---|
| `COLI_MODEL_DIRS` | unset | SPLIT the model across 2+ drives: a `;`/`,`-separated list of extra directories, each holding a **distinct** subset of the `.safetensors` shards (no duplication). Shards act as a search path — every shard is read from whichever drive holds it, so concurrent expert loads parallelise across drives and combined capacity is used. Scales to N drives. Metadata (config/tokenizer/`.coli_usage`) stays in the primary `COLI_MODEL` dir. Pairs well with `PIPE=1` (concurrent loaders) + `DIRECT=1`. Distinct from — and composable with — `COLI_MODEL_MIRROR`: the mirror is matched per-shard by basename against the merged (split) index, so a mirror dir may hold a copy of any subset of the split's shards. |
| `COLI_MODEL_MIRROR` | unset | `;`/`,`-separated list of directories, each a byte-identical (read-only) copy of the model on another drive; expert reads split across the primary and every mirror. Partial mirrors work (only the shards present are used). |
| `COLI_DISK_WEIGHTS` | unset (startup bandwidth probe) | Split ratio `<primary>,<mirror>[,<mirror2>...]` — one positive weight per drive (e.g. `1,1` for 50/50, `9,3` for a fast+slow pair, `1,1,1` for a 3-way mirror). Unset = probe every drive with the engine's own access pattern at startup. |
| `SNAP_MIRROR` | unset | Legacy alias for `COLI_MODEL_MIRROR`, consulted only when that is unset or empty. |
| `COLI_MIR_STRIPE` | see source | Stripe granularity for splitting a single expert read across mirror replicas. |

Per-drive byte counts are reported in a `MIRROR:` stats line. Combine with `DIRECT=1` so the two copies never compete for page cache.

## Vulkan (any GPU with a Vulkan 1.2 driver)

| Variable | Default | Effect |
|---|---|---|
| `COLI_VULKAN` | off | Enable the Vulkan backend. Requires a `make VK=1` build; fails at startup (no silent fallback) if libvulkan or the compiled shaders are missing. |
| `COLI_VK_DEV` | unset | Select the primary Vulkan physical-device enumeration index. Without it, the backend prefers a discrete GPU, then integrated/virtual devices. |
| `COLI_VK_SHADERS` | auto | Path to the compiled `qmatmul.spv` **or** the directory holding the `.spv` set; the other shaders are found next to it. Unset: `shaders/` next to the binary, then CWD-relative `shaders/qmatmul.spv`. |
| `COLI_VK_EXPERTS` | `320` | Pinned VRAM expert tier size: top-N experts by `.coli_usage` heat uploaded once at startup and served from VRAM with no RAM slot or disk read. `0` disables the tier (experts stay on the CPU path). ~19 MB VRAM per int4 expert. |
| `COLI_VK_DENSE` | `0` | Run the resident dense matmuls (attention projections, shared expert) on the GPU. |
| `COLI_VK_ATTN` | `0` | Run the S≤4 MLA absorb attention core (+ fused o-projection) on the GPU, with a persistent device-side KV mirror. |
| `COLI_VK_QPREP` | `1` (on) | Fuse the Q-prep step (RMSNorm + rope + compress) into one GPU dispatch instead of splitting it, which cost three fences where one suffices. `0` restores the split path; `2` additionally keeps CPU reference copies of Q and comp for A/B comparison. |
| `COLI_VK_RESERVE_GB` | `3.0` | VRAM (GB) held back from the expert tier for the lazily-allocated dense weights, KV mirror and staging buffers (measured ~1.7 GB at 4k ctx, growing with `max_t`). Only meaningful when the driver reports `VK_EXT_memory_budget`; without it the `COLI_VK_EXPERTS` count cap applies alone. |
| `COLI_VK_SPIN_US` | `300` | Microseconds to spin-poll a fence before blocking. `0` always blocks — lower latency at idle, at the cost of a core spinning. |

### Second Vulkan device (opt-in)

A second GPU can hold the *next* heat-ranked experts after dev0's budget stops. Deliberately separate from the first device so the dev0 hot path is untouched and both groups can be in flight at once.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_DEV2` | unset (off) | Enable the second device tier. A number selects that physical device index; `auto` picks a distinct real GPU (a second *logical* device on the same physical GPU is accepted only when forced by index — that is the pre-hardware test mode). |
| `COLI_VK_EXPERTS2` | `512` | Expert count cap for the dev2 tier (only read when `COLI_VK_DEV2` brought a device up). |
| `COLI_VK_RESERVE2_GB` | `0.5` | VRAM (GB) held back on dev2, as `COLI_VK_RESERVE_GB` is for dev0. |

See [docs/vulkan.md](vulkan.md). On multi-core boxes also set `COLI_NO_OMP_TUNE=1` (see that doc for why).

## CUDA (NVIDIA)

| Variable | Default | Effect |
|---|---|---|
| `COLI_CUDA` | off | Enable the CUDA backend. Requires a CUDA build. An explicit `COLI_CUDA=0` disables it **and suppresses the Windows bare-run auto-enable** (before this, Windows "CPU" runs with `COLI_CUDA=0` silently got a VRAM expert tier). The CLI flag `--gpu none` is the canonical hard off-switch on every platform. |
| `COLI_GPU` / `COLI_GPUS` | unset | Device selection (`auto`, `none`, or a list like `0,1`). Requires `COLI_CUDA=1`. |
| `CUDA_DENSE` | `0` | Place dense (non-expert) matmuls on the GPU. Off by default the engine reports `routed experts only (resident dense on CPU)`: on a host where the CPU is the limiter this leaves the dense path of every layer on the CPU while the VRAM tier serves experts only. Measured x2.8 on a 4x A6000 / 24-core host (1.53 -> 4.26 tok/s). |
| `CUDA_EXPERT_GB` | `0` | VRAM budget (GB) for caching experts on the GPU. Also accepts `auto`. |
| `CUDA_RESERVE_GB` | `2.0` | VRAM (GB) held back from the expert tier for activations, scratch and the KV cache. |
| `CUDA_EXPERT_LOAD_BALANCE` | `0` (off) | Experimental multi-GPU expert assignment: keep the same frequency-ranked GPU prefix, but greedily distribute it by accumulated profile weight instead of resident bytes alone. On one 6×RTX 5090 fixed replay its three-run median was +2.9%, with large variance; leave off unless validated on the target workload. |
| `CUDA_RELEASE_HOST` | auto (`1` if >1 device) | Release host-side copies after upload. |
| `COLI_CUDA_ROUTER` | `0` (off) | `=1` runs the MoE router (logits + top-k select) on the GPU at S=1. Skipped while a routing trace is being recorded, under `CACHE_ROUTE`, and above 4096 experts / topk 64. |
| `COLI_CUDA_RESID` | `0` (off) | `=1` keeps the residual stream on the device between layers instead of copying it back to the host each time. |
| `COLI_DSA_GATHER` | `0` (off) | `=1` gathers the DSA-selected KV rows on the GPU. With `DSA_FORCE=1` (identity selection) the output is byte-identical to the dense CUDA path, which is how the gather is validated. |
| `COLI_CUDA_ATTN` | off | Run S≤4 attention on the GPU. |
| `COLI_CUDA_ATTN_PREFIX` | off | Reuse one uploaded decode activation across `q_a` and `kv_a` while preserving the stock CPU RMSNorm path. |
| `COLI_CUDA_ATTN_SHARD` | off | `=1` splits KV-b heads across devices during attention load (multi-GPU). |
| `COLI_CUDA_PROFILE` | off | Emit CUDA timing. For the expert groups: both the synchronous dispatch and the async issue/take path decode uses, measuring the same three phases (`x` upload, kernels, result download) into the same counters. For the **resident dense GEMVs** (`lm_head`, `dnproj`, `dnout`, `attnout`, `attnproj`): the same three phases plus CPU **wall** time over the H2D→D2H window (the tensor upload and the buffer reserve sit before it, so a first call does not poison the average), reported by `[qtier] dense_stats` as effective bandwidth over weights **and** scales, broken out per card as `[qtier]   dev N dense` when more than one GPU is configured (`auto_place` splits the trunk by budget, and cards need not match) — bytes over kernel time is what the kernel achieves, bytes over wall time is what the caller gets, and the gap between them is the round-trip. Off, nothing is recorded and `[qtier] group_stats` says so rather than printing zeros. Measuring costs: the events are launch overhead on the very paths whose launch overhead is in question, so compare absolute times only against other profiled runs. **The kernel window is not all kernel**: `ev[1]` is recorded after the H2D copy and `ev[2]` after the launch, both on stream 0, so a CPU slow to *issue* the launch leaves the card idle inside the measured interval — the same 2704 calls over the same 92.37 GB read 475 ms under clang and 750 under gcc, purely from the host compiler (`docs/experiments/qwen36-place-clang-clean-2026-09-20-raw.txt`). Treat `kernel GB/s` as a lower bound, and a gap between two builds as host contention rather than a kernel difference. |
| `COLI_MTP_GUARD_PCT` | `70` | Pause MTP after the guard window when recent acceptance falls below this percentage. |
| `COLI_MTP_GUARD_WINDOW` | `24` | Number of MTP proposals used by the soft acceptance guard. |
| `COLI_CUDA_PIPE` | `0` (off) | `1` engages the multi-step attention pipeline; `2` enables the pipe2 path. |
| `COLI_CUDA_PIPE_SHARD` | off | `=1` runs the multi-device P2P head-shard attention path (opt-in for NVLink topologies; serializes ~95 MB/layer over a star PCIe topology). |
| `COLI_CUDA_PIPE_S_MIN` | `1` single-GPU, `8` multi-GPU | Minimum prefill batch S to engage the pipe2 CUDA path. |
| `COLI_CUDA_MTP` | `0` (off) | `=1` opts into MTP speculation under CUDA (off by default: cold streaming experts run on CPU where the fused-pair/IDOT kernels diverge in FP order, collapsing draft acceptance, #163/#292 — though #467 measured acceptance holding at 49% on sm_120). When set explicitly, the resource planner skips its `DRAFT=0` export so the engine's auto path can engage draft=3 — no need to also set `DRAFT`. Note the measured trade-off (#467): at ~85% hit the widened S=4 expert union costs more than speculation saves (−32%); the opt-in pays only near-full residency (~99% hit). |
| `COLI_CUDA_ASYNC` | on | `=0` forces synchronous `cudaMemcpy` instead of async + pinned host staging. |
| `COLI_CUDA_DUAL_PROJ` | on | `=0` issues gate+up as two separate launches instead of one fused `grouped_hidden_w4_dual`. |
| `COLI_CUDA_W4_PACKED` | on | `=0` disables the grouped packed-int4 path. |
| `COLI_CUDA_F8_WARP` | on (CUDA), off (HIP) | fmt=8 (fp8-e4m3) kernel selector. Default on CUDA: warp-per-row kernels with shared-memory LUT decode and reference-mirroring accumulation (f32 per 128-block, double across blocks, like the CPU `matmul_fp8`). `=0` restores the original fmt=8 kernels everywhere they run — grouped AND the dense `quant_matmul` branch. `=2` routes the warp kernels' decode through cuda_fp8.h: a real hardware `cvt` only on sm_89+, the header's bit-manip emulation below that, and plain `=1` behavior where cuda_fp8.h is absent (HIP); experimental until the 256-value sweep certifies it on the target silicon. Non-numeric values select the default. HIP defaults to `=0` because the warp kernels' wave64 width-32 shuffle sub-grouping is not yet validated on AMD silicon. |
| `COLI_CUDA_I8_ROWS` | `2` | Output rows per block for the fmt=1 **per-row** int8 dense GEMV (`quant_matmul_i8r`); `0` restores the original one-row-per-block kernel, `2`, `4` and `8` are the instantiated widths, anything else reads as the default. The generic branch gives one block to one row and every block reads the **whole** activation vector, so a call moves `I*O` bytes of weights and `I*O*4` of activations — four fifths of the traffic is `x`, and `dense_stats` counts only the weights. At R rows per block `x` is read once per R rows and the total falls from 5× the weight bytes to `(1 + 4/R)×`: **1.5× at R=8, a factor of 3.3**, against the 3.4 that the trunk's measured 29 % of peak leaves on the table. The hot path issues `R*U` independent weight loads (16 in every instantiation) before consuming any, because Nsight Compute measures this kernel as **memory-latency-bound, not bandwidth-bound** — occupancy 91.9 %, DRAM throughput 34.6 %, and 60 % of a 27-cycle issue gap stalled on an L1TEX scoreboard. The summation order per row is unchanged — still `i = t, t+256, …` ascending, still the same reduction tree and trailing f32 scale — so the result is **bitwise identical** and cannot move a logit, which is what `tests/test_int8_rows_cuda.cu` asserts with `memcmp`. **Measured** on an RTX 4070 Ti SUPER (sm_89, qwen36 i4 gs64, clang build, 4×4 alternated + a profiled pass per arm, two sessions; `step()` over 128 generated tokens, kernel GB/s over 64 — compare within a column, not across): `step()` medians fall from **27.25 / 27.40 at R=0** to **24.35 / 23.75 / 23.70** (session 1) and **23.45 / 23.90 / 23.80** (session 2) at R=2 / R=4 / R=8 — about **−13 %**, and that half is the solid one: two sessions, 4x4 alternated, every R>=2 median below every R=0 median in both. Two claims made for it on 22 September are withdrawn the same day: the range "23.4–23.9" excluded session 1's R=2 (24.35), and "per-run in the record … in either" is wrong about session 1, which states "per-rep lines not retained" (`qwen36-i8-rows-2026-09-20-raw.txt:32`). The no-overlap check is per-run in session 2 and median-to-median in session 1. **The bandwidth half is SUSPENDED as of 22 September 2026**: this row used to read "342–381 GB/s against the original's 187–191", and the same 5224 calls over the same 112.07 GB read **175 GB/s** in `qwen36-dense-pinned-2026-09-21-raw.txt` one day later. A factor of 2.1. A third reading exists — **593 ms / 189 GB/s** on arm cB of `qwen36-place-clang-clean-2026-09-20-raw.txt:60-61` — and it does not settle the question: that record pins `COLI_CUDA_I8_ROWS` nowhere, describes itself as the first run on a build carrying the dense counters, and 189 GB/s sits 1 % from the R=0 arm's own 191 GB/s (`qwen36-i8-rows-2026-09-20-raw.txt:106-107`). Calling it a third R=2 measurement, as an earlier version of this row did, is a configuration deduced rather than logged — the same defect this audit charges against the −7.77. Withdrawn 22 September 2026. The suspect is host-side and the `COLI_CUDA_PROFILE` row of this table already documents the mechanism: the same 2704 calls over the same 92.37 GB read 475 ms under clang and 750 under gcc, purely from the host compiler, because a CPU slow to issue leaves the card idle inside the interval the kernel column measures. Of the three records carrying `dense_stats`, the pinned one is the only one whose header declares `OMP_NUM_THREADS=16 (7950X physical cores), WAIT_POLICY=ACTIVE`. (An earlier version of this row blamed the variable itself, calling the pinned record the only one of the week not pinning it; false -- `place-clang-clean` and `xoff-fix-noop` do not pin it either. Retracted 22 September 2026.) Do not cite a GB/s figure for this kernel until a session measures R=0 and R=2 back to back with the variable explicitly pinned. On `step()` the three widths sit within **0.65 ms** of each other in session 1 and 0.45 in session 2, and **their order does not replicate**: session 1 reads R=2 24.35 / R=4 23.75 / R=8 23.70, so R=8 is the FASTEST; session 2 reads R=2 23.45 / R=4 23.90 / R=8 23.80, so R=4 is the slowest. The default R=2 is a choice among three widths `step()` cannot separate, not a measured winner. Where R=8 is consistently the slowest of the three is the kernel GB/s (318 and 299, against 381/389 and 372/342) — and that metric is suspended earlier in this same row, so it cannot carry the choice either. An earlier version of this row said R=8 was the slowest in both sessions without saying on which metric; on `step()` that is false. The `(1 + 4/R)×` traffic model was compared against those GB/s figures — measured/predicted 1.17–1.22 at R=2 and 0.47–0.51 at R=8 — and **that comparison is suspended with them**: a ratio whose denominator is a contaminated kernel time says nothing. Two conclusions used to rest on that ratio — that the gain is not purely the `x` re-read, and that R beyond 4 costs more in registers and occupancy than it saves in traffic. **Both are suspended with it.** An earlier version of this row tried to rescue the second by re-anchoring it to `step()`, where the data says the opposite (see the ordering above); that rescue is withdrawn, 22 September 2026. What survives is R>=2 against R=0, and nothing about which R. |
| `COLI_CUDA_DOWN_ROWS` | **on, R=4** (0 where the graph is unavailable, e.g. HIP) | **On by default since the 2x2 -- and ONLY as a pair with `COLI_CUDA_GRAPH`.** Set `0` for off. Alone this kernel COSTS 1.15 ms/token: it is 18 % faster on the GPU and the token gets worse, because `take` was already near zero and the freed GPU time had nowhere to go -- and 91 % of the regression lands on `dn-sub proj`, `norm+out` and `lm_head`, the placed dense GEMVs sharing stream 0 with the expert group: 4,096 long-lived blocks holding 4 KB of shared memory each leave fewer scheduling slots than 16,384 that retire at once. With the graph on, the graph moves ~0.7 ms out of `issue` into `take` and this kernel fills it. Measured 2x2 over {GRAPH 0,1} x {DOWN_ROWS 0,4}, 6 reps per cell, identical generation in every cell: effect of DOWN_ROWS at GRAPH=0 **+1.15**, at GRAPH=1 **-0.25**, **interaction -1.40**, and the diagonal both-off -> both-on **-0.43 ms/token with 6/6 repetitions negative**. Qualified 22 September 2026: -0.43 is the MEAN; the paired median in the same record is **-0.35**, and the within-cell spread (0.70-1.90) is **1.6 to 4.4 times the mean effect**, 2.0 to 5.4 times the median. Every other record of that week quotes paired medians, so quoting the mean here was choosing the more favourable of the two. The 6/6 sign agreement is what the pair rests on, not the distance between the medians. Turning the graph off while leaving this on is the one combination known to be worse than shipping neither; `down_g4_launch` prints a warning if it sees it. Compiled off where `COLI_GPU_HAS_GRAPH` is 0, because there the arm that pays for it does not exist. Bitwise identical to `grouped_down_g4` either way (`tests/test_grouped_down_rows_cuda.cu`, `memcmp`), so no token can move. Records: `docs/experiments/qwen36-graph-downrows-2x2-2026-09-21-raw.txt` (the 2x2) and `docs/experiments/qwen36-expert-down-rows-2026-09-20-raw.txt` (the kernel alone). |
| `COLI_CUDA_X_BCAST` | on | `=0` makes `qt_issue` copy the activation vector once per expert instead of uploading it once and pointing every chunk at it. All K experts of a decode token read the **same** vector, so the copy was pure duplication: 8 KiB became 64 KiB per layer in the caller's buffer, again in the backend's pinned staging buffer, and again over PCIe — **~4.5 MB a token** of host memcpy. On a DLL older than the broadcast export (`coli_cuda_expert_group_issue_x`, resolved with `RESOLVE_OPT`) the duplicating path is taken regardless, because handing that DLL a one-row buffer would have it read `total*D` floats off the end. This flag exists so the change has a B arm without rebuilding the DLL between measurements, and as the escape hatch if the broadcast form is ever suspected. Under `COLI_TIMERS=1` the engine prints which path it took. **Measured** (RTX 4070 Ti SUPER, 2×6 alternated, identical hit rate and expert count in both arms): `h2d` **69 → 55 ms (−20 %)** and `step()` **unchanged** — paired delta 0.00 ms/token with three pairs of each sign. `take` is 0.53 ms of a 24 ms token, so the expert path's transfers are not what the token waits for, the same reason `COLI_CUDA_DOWN_ROWS`'s faster kernels bought nothing. The sub-finding is worth more than the change: h2d fell 20 % while the payload fell 8×, i.e. **15.6 µs for an 8 KiB copy** — the transfer is per-call overhead, not bytes, so the lever here is the number of driver calls (CUDA graphs), not their size. Record: `docs/experiments/qwen36-x-broadcast-2026-09-20-raw.txt`. |
| `COLI_CUDA_GRAPH` | **on** (CUDA only) | Replays the expert group as one `cudaGraphLaunch` instead of five driver calls per MoE layer. `=0` turns it off. **Alone it is worth nothing on the token**: `issue` falls 28 % and every millisecond reappears in `take` -- 2.47 -> 1.79 and 0.76 -> 1.50 within that one session. Do not read those as constants of the engine: across the week's records, at GRAPH=0 or on builds with no graph, `take` reads **0.27, 0.28, 0.31, 0.31, 0.33, 0.33, 0.43, 0.53, 0.57, 0.69, 0.70, 0.76 and 1.14** -- thirteen readings, eleven distinct values -- on arms that differ in placement, in `DOWN_ROWS` and in host compiler. (`docs/qwen36-cuda-tier.md` tabulates each one with its record, line and arm.) No spread over that whole list means anything, because it is not one configuration, and **no published run pins one configuration and reads `take` twice**. Four corrections to this passage are withdrawn, 22 September 2026 -- the fourth being the closest call and the most instructive. (Three of them put a number on the drift; the third put no number on it and is withdrawn for the reason it gave. An earlier version of this row called all four "attempts to put a number on the drift", which does not describe the third.) Two records (`expert-down-rows` 20 Sept, `graph-downrows-2x2` 21 Sept) declare the same box, engine, model, prompt, `COLI_PLACE` and `I8_ROWS=2`, and give +0.23 and +0.24 on `take`. But the same two arms move **+0.71 on `issue` and +3.35 on `step()`** -- the whole token is 14 % slower on the later day -- so +0.23 is `take`'s share of an unexplained level shift, not a drift figure; one side is a single repetition ("rip 1") and the other a six-rep aggregate; and the two sessions do not run the same DLL, because `COLI_CUDA_X_BCAST` landed 27 minutes after the earlier record and sits on the very call `take` waits for. Quoting the one axis that flatters the conclusion is the defect the other three withdrawals are about. The first said the drift is LARGER than the transfer, resting on a list containing a **1.30** that exists nowhere in the records as a reading of `take` and a **1.50** that is the GRAPH=1 cell of the same session as the 0.76. The second replaced it with a spread of **0.61** over a subset picked without a stated rule. The third refused to quote any drift on the ground that no run pins a configuration and reads `take` twice, and a review produced a counterexample -- two records whose headers matched on every axis they declare. The fourth withdrawal disposes of that counterexample (the two sessions differ on a DLL neither header names), so the original ground stands and the sentence above is the repo's position, not a claim this row then contradicts. An earlier version of this row asserted both halves and reconciled neither: the reconciliation existed only in docs/qwen36-cuda-tier.md, which is the same "retracted in one file, left standing in the other" defect this row is part of a PR to remove. Repaired 22 September 2026. **The transfer itself is NOT a paired measurement**, either. `issue` 2.13 → 1.52 with `take` 1.14 → 1.86 (`qwen36-expert-graph-2026-09-20-raw.txt:33,35`) is **one repetition**, labelled "rip 1" in that record, whose only published paired delta is `issue` **−0.60** over six pairs. `issue` 2.47 → 1.79 with `take` 0.76 → 1.50 (`qwen36-graph-downrows-2x2-2026-09-21-raw.txt:80-81`) is a difference between cell values; that record pairs only `step()`. So two sessions agree on the shape, and on `take` the two sizes (+0.72, +0.74) are close, but calling them "paired", as an earlier version of this row did, is not what the records say. And the reason the transfer buys nothing on the token is separate: the MoE phase is GPU-bound with idle CPU inside it, so a CPU-side saving there converts into waiting. It ships because that widened `take` is exactly what `COLI_CUDA_DOWN_ROWS` needs: see that row for the 2x2 and the **-0.43 ms/token, 6/6** diagonal. **The two are a pair** -- turning this off while leaving down-rows on is the one combination measured worse than shipping neither. Bitwise identical to per-call launches on capture *and* replay (`tests/test_grouped_g4_cuda.cu`, which pins this to `0` for its reference arm and is now actually executed by `make cuda-test`). Refuses to run under `COLI_CUDA_PROFILE` (four `cudaEventRecord` inside a capture would bake the timing branch into the graph), and only for a broadcast input with 1..8 resident experts. If capture or instantiation fails the call refuses, that one layer goes to the CPU, and the backend says so once on stderr. 0 on HIP: there is no `cudaStreamBeginCapture`. Records: `docs/experiments/qwen36-expert-graph-2026-09-20-raw.txt` (the graph alone) and `docs/experiments/qwen36-graph-downrows-2x2-2026-09-21-raw.txt` (the 2x2 that made it default). |
| `COLI_CUDA_DENSE_PINNED` | **off** | Pinned host staging for the resident dense GEMV (`coli_cuda_matmul`: `dnproj`, `dnout`, `attnout`, `lm_head`, the shared expert). `1` stages both copies through pinned memory and keeps them synchronous; `2` also makes them async on stream 0 with **one** `cudaStreamSynchronize` instead of two blocking waits; anything else off. **Measured, and it makes the token SLOWER** (RTX 4070 Ti SUPER, qwen36 i4 gs64, clang, 3x6 alternated, identical hit rate and `miss(CPU)` in every arm): `step()` **27.20 -> 28.75 at mode 1 (+1.60 paired, 6/6) and 29.20 at mode 2 (+2.00 paired, 6/6)**. Leave it off. **Why the premise was wrong**, which is the part worth keeping: `h2d` went **140 -> 186 ms** in mode 1, i.e. up 33 % in the arm doing strictly less work. A **pageable** host-to-device `cudaMemcpy` returns once the bytes are in the driver's staging buffer -- the DMA need not have completed -- and every call here is `S=1`, so `xb` is 8 KiB, inside that window. The two "synchronous pageable copies" were not both synchronous: the driver's hidden staging copy was not a cost, it was buying asynchrony for free on a payload where the copy is cheaper than the wait. Mode 2's mechanism did work (`d2h` **171 -> 34 ms**, with the wait moving into the launch/sync residue, 39 -> 160 ms) and the token still lost. Kept as the B arm for a fact worth not re-deriving: on this path the pageable copy is the fast one. The backend prints `[cuda] dense pinned staging active: mode=N` once when the branch fires; a flat A/B without that line means the DLL was never rebuilt. Record: `docs/experiments/qwen36-dense-pinned-2026-09-21-raw.txt`. Measured again on 27 September in a different setup (auto placement, 87 % VRAM hit, the heat table, `COLI_CUDA_KEEPALIVE=1` in both arms; clocks not logged): mode 2 **+0.45 ms/token mean, 95 % [-1.44, +2.34], 1/6 negative** -- no gain, and the slowdown above not reproduced at that resolution (`docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt`, section 9). |
| `COLI_CUDA_KEEPALIVE` | off | `=1` (exactly) keeps a one-warp spin kernel of ~1 ms relaunched on every configured device, on its own non-blocking stream at the device's least priority (the default streams' priority too: no precedence either way), from a host thread that sleeps on a blocking-sync event between launches. Armed by `coli_cuda_init`, started by the first `coli_cuda_matmul`, and paused for 50 ms by every `coli_cuda_tensor_free` (renewed by the next): each `cudaFree` waits for the running spin, and qwen36's exit makes ~41000 `cudaFree` calls (two per tensor) before `coli_cuda_shutdown`, which made every ON run 18-20 s longer end to end until the pause (inferred from the code; the pause removed it). Other synchronising calls can still wait up to ~1 ms for the spin. Why: on an RTX 4070 Ti SUPER under Windows, qwen36 decode keeps the card 15-40 % busy and after a P2 burst of 1.5-2 s the driver drops it to P3 (memory 10251 -> 5001 MHz), after which the placed dense GEMVs' median duration is about twice their minimum; the driver's "Prefer maximum performance" had no measurable effect and `nvidia-smi` clock locks need administrator rights. **Measured** there, `c/tools/misure-envab.ps1`, six ABBA pairs, no heat table: `step()` 30.87 -> 27.62 ms/token, 95 % [-4.41, -2.09] (started at init) and 30.27 -> 27.30, [-4.32, -1.61] (started at the first GEMV), 18-20 s and 17-21 s longer end to end; with the pause on frees **29.07 -> 25.62, [-3.69, -3.21]**, with no end-to-end cost visible in the log end times (both arms 23-26 s apart, so 2-3 s would not show). 6/6 each time, text identical. Clocks were logged only in the first (init-time) run: P2, 2790-2805 / 10251 MHz through every ON run (`docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt`). Off by default: it holds the card at full clocks, and so at higher power, for as long as the process runs, plus one warp on one SM and a mostly sleeping CPU thread. Changes no result (`tests/test_cuda_keepalive_cuda.cu` compares GEMVs byte for byte). Stopped and joined in `coli_cuda_shutdown`, and at exit for callers that never call it. CUDA only (a HIP build prints that it ignores it); lives in `coli_cuda.dll` on Windows, so it needs `make cuda-dll`, not an engine rebuild -- but `make cuda-dll` rewrites `c/.build-config`, so the engine is rebuilt by the next engine `make` all the same: build the DLL first, the engine last. Prints `[cuda] keep-alive active: ...` once its streams exist, and `[cuda] keep-alive: setup failed` / `launch failed` otherwise. |
| `COLI_CUDA_FLUSH` | off | `=1` (exactly) calls `cudaStreamQuery` on the expert group's stream right after each group is enqueued (`coli_cuda_expert_group_issue_x`, graph replay and per-call path alike), meant to make the driver submit the group now instead of with its next command batch. **Measured faster on qwen36 (one prompt, 128 tokens), off by default** (other engines share the DLL and are not measured; the mechanism is not shown). Measured with `c/tools/misure-envab.ps1` (six ABBA pairs, qwen36 35B with the ISTRUZIONI.md setting -- CACHE_ROUTE=1 ROUTE_J=4, QT_PREFILL_REPLAN=1, COLI_DN_GPU=1, keep-alive -- and the heat table in both arms, 128 tokens, one prompt): `step()` **16.78 -> 16.18 ms/token, -0.60, 95 % [-1.09, -0.11], 6/6**, text identical in all twelve runs. Why it exists: in an Nsight trace of qwen36 decode on Windows without it (RTX 4070 Ti SUPER, WDDM, 9 October 2026, profiled, read with `c/tools/nsys-moe.ps1`), per-group medians over 5080 decode groups: the x copy (8 KiB) takes 1.4 us, and the hidden kernel starts 33.3 us after it ends, 21.7 us after the `cudaGraphLaunch` call has returned. That looks like WDDM command batching; it is still a hypothesis, because no trace with the flush on was taken, so whether those ~22 us fell is not shown. Predicted before the A/B: at most about the `take` wait (0.84 ms/token in the 9 October run) minus the query's own cost, as a smaller wait and a larger `issue`. Not what the per-arm means (not paired) show: `issue` 2.06 -> 1.77 and the DeltaNet/lm_head/attention timers lower, while `wait` went 0.51 -> 0.59. The `cudaErrorNotReady` the query returns while the group runs would trip the next launch check that reads `cudaGetLastError()` -- the per-call group, the dense GEMV, the DeltaNet chain -- if the runtime recorded it as the last error; on CUDA 13.4 it does not (the test below), and the clearing stays for runtimes that do. A status query changes no data; `tests/test_grouped_g4_cuda.cu` checks it (bitwise on both launch paths, flushes counted, at least one query that found the group unfinished, no error left behind), run on the RTX 4070 Ti SUPER with CUDA 13.4: OK, 4 of 4 queries found the group unfinished, NotReady left as the last error in 0 of them. Record: `docs/experiments/qwen36-cuda-flush-2026-10-10-raw.txt`. Lives in `coli_cuda.dll` on Windows: `make cuda-dll`. Prints `[cuda] group flush active: ...` once, at the first flushed group. |
| `COLI_CUDA_GROUP_ZC` | off | `=1` (exactly): the expert group's CUDA graph as kernels only. A staging kernel (`group_zc_stage`) reads x and the descriptors from the pinned host buffers in place (unified addressing) instead of two copies, the hidden and down kernels run as before, and the down kernel writes its rows straight into the pinned `host_y`, so no readback copy follows; `take`'s synchronize makes them visible as it made the copy's. Same kernels on the same data: the rows are bitwise the copy path's (`tests/test_grouped_g4_cuda.cu`: two alternating inputs, each against its per-call reference, captured once and replayed, branch 3 too, and back to the copy path with a re-capture; run on the RTX 4070 Ti SUPER with CUDA 13.4: OK, the version of #83; the version of #84, run in a window where `COLI_CUDA_GROUP_ZC=1` was set, failed one capture count because of that variable, with every value comparison bitwise -- the test now clears the group switches at start, and that version (#85) ran OK in the same window). Only where the graph is eligible (graph on, `COLI_CUDA_PROFILE` off -- so only Nsight can time it -- broadcast x, branches 3 and 4, 1..8 experts; if a capture fails the same kernels run as ordinary launches, still correct); with no device address for the pinned buffers it says so once and the copy path stays. The graph signature carries it, so switching re-captures. **Measured faster on qwen36 (one prompt, 128 tokens), off by default**: `c/tools/misure-envab.ps1`, six ABBA pairs, the ISTRUZIONI.md setting with `COLI_CUDA_FLUSH=1`, `QT_ASYNC_ISSUE=1` and the heat table in both arms: `step()` **16.47 -> 15.08 ms/token, -1.38, 95 % [-2.20, -0.57], 6/6**, text identical in all twelve runs; the OFF arm was spread (15.4 to 17.3) and the last two pairs gave -0.4 and -0.6. Why, and what the trace shows: in the 10 October trace with `COLI_CUDA_FLUSH=1 QT_ASYNC_ISSUE=1` (moe2, `docs/experiments/qwen36-async-issue-2026-10-10-raw.txt`) the kernels started 6.8 us after the 52.8 us `cudaGraphLaunch` call ended, ~60 us of the 136 us group before the first kernel; the hypothesis was that the copy-engine nodes make the launch long. With the variant on (moe3, same switches, profiled, medians per decode group; one trace each, hours apart, not paired): the call 52.8 -> 32.7 us, the first GPU operation 0.8 us after it ends, call start -> results 136.0 -> 118.1 us -- the hypothesis holds in part. The down kernel, now writing host_y over PCIe (one coalesced 16-byte store per block at the default `COLI_CUDA_DOWN_ROWS` R=4), went 23.5 -> 38.2 us, more than the 6.5 us readback it replaces: from the end of the call to the results the group is 4.2 us slower (80.7 -> 84.9). That the host writes are the cause is a reading, not measured. From call start to results the traces differ by about 0.7 ms/token (profiled); with the launch on the helper that reaches the decode thread only through `wait`, so it is a bound, not an amount in the step. In the A/B's per-arm means (not paired) `wait` fell 0.50, about a third of the gain, `shared` 3.73 -> 3.18 and attn/dn/head ~0.2 together, which the traces do not explain. `COLI_CUDA_GROUP_ZC_OUT` tests the down kernel's half (measured, see its row). Record: `docs/experiments/qwen36-group-zc-2026-10-10-raw.txt`. Not measured: other prompts, longer generations, without `COLI_CUDA_FLUSH`, against the copy path without `QT_ASYNC_ISSUE`. Prints `[cuda] group zero-copy active: ...` once. Lives in `coli_cuda.dll`: `make cuda-dll`. `c/tools/nsys-moe.ps1` recognises the zero-copy group shape. |
| `COLI_CUDA_GROUP_ZC_OUT` | off | `=1` (exactly), only with `COLI_CUDA_GROUP_ZC=1` active: the down kernel writes `ctx->y` in VRAM again, as on the copy path, and one more kernel (`group_zc_out`) copies the rows into the pinned `host_y`, one 16-byte float4 per thread -- still no copy engine. Why: in the moe3 trace (see `COLI_CUDA_GROUP_ZC`; one trace each, not paired) the down kernel writing host_y directly took 38.2 us against 23.5 writing VRAM in moe2, more than the 6.5 us readback copy it replaced. Hypothesis: the same 64 KiB (8 experts x 2048 floats) in whole 16-byte stores by consecutive threads, after the down, costs less than 4,096 scattered 16-byte stores, one per block of the down. **Measured faster on qwen36 (one prompt, 128 tokens), off by default**: `c/tools/misure-envab.ps1`, `COLI_CUDA_GROUP_ZC=1`, `COLI_CUDA_FLUSH=1`, `QT_ASYNC_ISSUE=1` and the heat table in both arms, ten ABBA pairs: `step()` **14.50 -> 13.98 ms/token, -0.52, 95 % [-1.02, -0.02], 8/10**, text identical in all twenty runs; at the edge of the noise (upper end -0.02; the position check gave 0.80, about 0.4 per position, cancelled by the ABBA order if constant). A six-pair run before it, in the same window: -1.00, [-2.13, +0.13], 5/6; the 10-pair run was taken because that interval straddled zero. The two agree within their intervals (pooled 16 pairs: -0.70, [-1.16, -0.24], 13/16), and the 10-pair figure is the smaller one. The trace bounds the mechanism at ~0.4 ms/token over the span it acts on ((14.7 - 4.3) us x 40 layers, medians, call end -> results; 0.26 from the traces' per-token sums), ~0.55 counted from call start as the `COLI_CUDA_GROUP_ZC` row does (that includes a shorter launch call the variant does not explain), reaching the decode thread through `wait` (1.57 -> 1.20 in the per-arm means): the 6-pair estimate is larger in magnitude than both, the 10-pair one lies between them, both intervals contain them. In the moe4 trace (profiled, medians per decode group, against moe3; one trace each, not paired) the down kernel is back at 23.5 us, end of down -> end of `group_zc_out` 4.3 us, and call end -> results is 84.9 -> 74.5 us: the hypothesis holds there. Record: `docs/experiments/qwen36-group-zc-out-2026-10-10-raw.txt`. Measured only with `QT_ASYNC_ISSUE=1`; not measured: other prompts, longer generations, without `QT_ASYNC_ISSUE`, `COLI_CUDA_FLUSH` or the keep-alive, with `HEAT_FILE=heat.bin` rewritten at every exit, and zero-copy plus this variant against the copy path in one A/B (the two gains come from different runs and baselines and are not to be added). A copy changes no value: `tests/test_grouped_g4_cuda.cu` checks the rows bitwise against the copy path with `host_y` poisoned before every issue (two alternating inputs, branch 3 too), one capture and three replays (bit 8 of the graph signature), the kernel alone on lengths that are not a multiple of 4, and that it does nothing without `COLI_CUDA_GROUP_ZC`; run on the RTX 4070 Ti SUPER with CUDA 13.4: OK (the #85 version, which clears the group switches left in the shell, in the measurement window). Ignored without `COLI_CUDA_GROUP_ZC` active or if either buffer is not 16-byte aligned (cudaMalloc and cudaMallocHost align far beyond that); one message, at the first graph-eligible group, says which of active or ignored applies then, and only then. Prints `[cuda] group zero-copy output active: ...` once. Lives in `coli_cuda.dll`: `make cuda-dll`. `c/tools/nsys-moe.ps1` counts the groups that end with `group_zc_out` and reads that kernel as their 'copia giu'. |
| `COLI_CUDA_TC_INT4` | off | `=1` uses the W4A4 WMMA Tensor Core path (when all expert tensors are int4 and dims divide). |
| `COLI_CUDA_TC_MIN_ROWS` | `8` | Min rows-per-expert to engage the W4A4 Tensor Core path. |
| `COLI_CUDA_TC_W4A16` | off | `=1` uses the lossless W4A16 Tensor Core path (compute capability ≥7). |
| `COLI_CUDA_TC_W4A16_MIN` | `16` | Per-expert row threshold above which W4A16 TC tiles dispatch (smaller batches fall back to the naive kernel). |
| `COLI_CUDA_SHARED_W4A16` | off | `=1` uploads shared-expert weights and runs the shared-MLP W4A16 Tensor Core kernel. |
| `COLI_CUDA_SHARED_W4A16_MIN_ROWS` | `32` | Min row count to engage the shared-MLP W4A16 kernel. |
| `CUDA_RAW_EXPERTS` | unset | Experimental ANS build only: keep this many hottest experts raw, then store subsequent VRAM experts losslessly compressed. Requires `COLI_ANS_SIDECAR`. |
| `COLI_ANS_SIDECAR` | unset | Experimental ANS build only: path to the sequential compressed-expert sidecar. |
| `COLI_ANS_PACK` | `0` | Experimental ANS build only: `=1` creates `COLI_ANS_SIDECAR` during pinning and exits before inference. |
| `COLI_ANS_DIRECT` | `0` | Experimental ANS build on Linux: `=1` reads the sidecar with aligned `O_DIRECT`, bypassing page-cache overhead. Falls back to buffered I/O if unavailable. |
| `COLI_ANS_PROFILE` | `0` | Experimental ANS build: print sidecar header, read, staging/allocation, and H2D enqueue timings on first use. |
| `COLI_METAL_UNTRACKED` | off (Metal only) | `=1` sets `MTLResourceHazardTrackingModeUntracked` on Metal buffers (reduces hazard-tracking overhead). |

> **Windows note.** On Windows, a bare `coli chat` / `coli run` / `coli serve`
> (no `--gpu`/`--vram`/`--auto-tier`) **auto-enables the GPU** when it detects a
> CUDA build (`coli_cuda.dll` next to the engine) and at least one GPU via
> `nvidia-smi`. The expert-tier VRAM budget is then sized automatically from the
> card's free VRAM (same computation as `--auto-tier`). If `nvidia-smi` is not on
> `PATH` the run falls back to CPU with a warning — pass `--vram N` (or add
> `nvidia-smi` to `PATH`) to enable CUDA in that case. `--gpu none` forces
> CPU-only. (Linux/macOS behaviour is unchanged: pass a flag to enable CUDA.)

---

## Advanced / experimental / debug

These are for testing, benchmarking, or internal use — not part of the everyday surface, and some may change without notice.

| Variable | Default | Effect |
|---|---|---|
| `SPEC` | `1` | Speculative decoding on/off. |
| `DRAFT` | `-1` (auto: 3 with MTP, else 0) | Number of speculative draft tokens per step. |
| `GRAMMAR` | unset | Path to a GBNF grammar file to constrain generation. Takes precedence over `SCHEMA`. |
| `SCHEMA` | unset | Path to a JSON-Schema file compiled to GBNF to constrain generation (consulted only when `GRAMMAR` is empty). |
| `GRAMMAR_DRAFT` | unset | Max grammar-forced draft span length. |
| `COLI_DRAFT_CORPUS` | unset | Path to a file of frozen token ids (whitespace-separated, `-1` separates spans) used as a speculative draft source: the engine proposes the continuation that followed the longest suffix of the live context found in the corpus. Off when unset. Build one from any run with `TOKENS=1`. See [corpus-draft.md](corpus-draft.md). |
| `COLI_CORPUS_K` | `8` (max 48) | Proposal depth for `COLI_DRAFT_CORPUS`. Deeper raises the forward multiplier and the per-forward cost. |
| `COLI_CORPUS_MINACC` | `50` | Acceptance floor (percent) for the corpus source. Below it over a 24-proposal window the source pauses for 256 tokens, then re-arms — rejected drafts cost real time. |
| `EXPERT_BUDGET` | `0` (off) | Cap experts loaded per layer (MoE-Spec). **Quarantined:** silently forced to `0` unless `EXPERT_BUDGET_EXPERIMENTAL` is set — every tested value is either no faster or incoherent (issue #303). |
| `EXPERT_BUDGET_EXPERIMENTAL` | unset | Setting it (any value) allows `EXPERT_BUDGET>0` to actually take effect (expect garbage, #294). |
| `DSA` | on | Dynamic Sparse Attention indexer. `DSA=0` disables. |
| `DSA_FORCE` | `0` | Force the DSA path on. |
| `DSA_TOPK` | model value | Override the DSA index top-k (testing). |
| `LOOKA` | `0` | Measure router predictability (instrumentation). |
| `I4_ACC512` / `I4_ACC512_TEST` | off | int4 512-wide accumulator kernel toggle / self-test. |
| `NOPACK` | off | Disable weight packing. |
| `DROP` | off | Drop-related debug toggle. |
| `PIN_FILL` | `0` | Fill the pinned store even without usage data. |
| `MTP_DEBUG` / `MTP_PRENORM` / `MTP_SWAP` | off | MTP head debugging / ablations. |
| `STATS` | unset | Write an expert-usage histogram to `STATS=<file>` at end of run. |
| `TOKENS` | unset | If set, dumps generated token ids to stderr for A/B comparison. |
| `SCORE` | unset | Scoring/eval mode over `SCORE=<file>`. |
| `SCORE_PREFIX` | on | If unset or `≠0`, prepends `[gMASK]<sop>` to scoring contexts (GLM-family only). |
| `REPIN_VERBOSE` | off | If set, prints per-swap `[REPIN]` diagnostics during VRAM repin. |
| `REF` / `REF_FORCE` | `ref_glm.json` | Reference-output comparison mode. |
| `REPLAY` | unset | Replay mode. |
| `TF` | unset | Teacher-forcing mode. |
| `CHAT_TEMPLATE` | `1` | Apply the GLM chat template (`0` = raw prompt). |
| `PPL` | off (`olmoe.c`, `qwen38.c`, `qwen36.c`) | `PPL=1` enters teacher-forced NLL/perplexity meter mode in the OLMoE, Qwen3.8 and Qwen3.6 engines. `qwen36` scores a ref `.json` (`full_ids` after `prompt_ids`) or, given a text file, the file's own tokens (see `PPL_CTX`, docs/qwen36.md). |
| `PPL_CTX` | `1` | `qwen36`, `PPL=1` on a text file: how many of the first tokens are context only; every later token is scored. Clamped to 1..tokens-1. |
| `CONSIST` | off | Prefill/decode self-consistency: the engine against itself at two batch sizes — one batched pass over the whole sequence against a prefix prefill followed by token-by-token decode, comparing the logits of the same positions. No oracle and no reference implementation, so it runs on any model, quantization and backend, including ones no CI runner has a GPU for. **Two separate implementations, one per engine, and they differ.** `colibri.c`: enabled by the variable being *set at all*, so `CONSIST=0` turns it **on**; takes its tokens from `PROMPT` (prepending GLM's `[gMASK]<sop>` unless `CHAT_TEMPLATE=0`) or from the ref file; one split point, `CONSIST_NP`; 512-token ceiling (`step_all` writes `S*D` into a `512*D` buffer); also covers `COLI_PREFILL_CHUNK`, which only the decode arm honours. `qwen36.c`: enabled only by `CONSIST=1` exactly; takes whatever token sequence the run already has (the ref file's `full_ids`, else the encoded prompt); **several** split points; needs ≥4 tokens and at most `QWEN36_ATTN_MAX_CTX`; ignores `CONSIST_NP`. What the qwen36 one guards is the code that **branches on `S`** — every placement knob is an `S == 1` fast path a batch never takes — plus state continuity across `step()` boundaries (KV length, RoPE positions, the DeltaNet carry). Neither can see a defect the two arms **share**: both start from the same state, so a wrong initialisation or a mask applied to both sides reports zero. That half belongs to the tiny-model oracle. Both exit non-zero when the largest relative gap exceeds `CONSIST_TOL`. CI gates the qwen36 one only. |
| `CONSIST_TOL` | `1e-2` | Gate for `CONSIST`, honoured by both engines. The quantity separates the two failure classes: reordered f32 accumulation over the hidden dim lands near `D*eps` — 2.4e-4 at qwen36's hidden 2048, ~1e-3 at colibri's 7168 — while a wrong mask or a misaddressed KV row lands at O(1). Measured on qwen36: a skipped decode-only state update gives 0.25–0.30, an `S == 1` RoPE off-by-one 0.04–0.06. An error confined to one layer can still dilute through the residual stream and the lm_head before it is measured. Argmax flips are reported but never gated — a flip requires the top two within `2*gap` by construction, so gating them would be an assertion that cannot fail. |
| `CONSIST_NP` | half the tokens (`colibri.c` only) | How many tokens the decode arm prefills before stepping the rest one at a time. Clamped to at least 2, and rejected if it leaves no continuation. **`qwen36.c` ignores it silently** — it picks its own split points — so a value learned on one engine does nothing on the other. |
| `ABLATE_SCORE` | unset | Causal-ablation sweep over `ABLATE_SCORE=<file>`, with a per-target-position final-logit read-out. Runs before `SCORE` and exits when done. |
| `ABLATE_OUT` | unset | Where the ablation sweep writes its logit read-out. Pair with `ABLATE_SCORE`; an optional `ROUTE_TRACE` records the post-ablation router trace. |
| `DEBUG_LOGITS` | unset | In reference-comparison mode, dump per-position logit diagnostics. |
| `COLI_LOGIT_DUMP` | unset | `=1` prints the top-5 `id:logit` pairs per step to stderr — for comparing two engine configs on identical forced context (backend-exactness triage). |
| `I3_AVX512` | auto | Force the AVX-512 int3 kernel on (`1`) or off (`0`). |
| `I3_AVX512_TEST` | unset | Run the AVX-512 int3 self-test and exit. |
| `COLI_GPU_FAIL_AFTER` | unset | Fault injection: make GPU compute calls start failing after N of them, to exercise the CPU fallback without real hardware faults. Uploads and queries are not gated. |
| `COLI_VK_TEST_BALLAST` | `0` | Allocate N extra dummy Vulkan buffers to reproduce decode attention degrading with expert-tier size even when VRAM is free (measured 7.9s @2.6k buffer objects → 15.6s @4.3k with 2.9 GB still free). |
| `COLI_SERVE_ALL_STOPS` | unset | In batched serve mode, keep every stop token instead of filtering to the EOS-like ones. Trades the #401 tool-call safety for behaviour some non-tool clients prefer. |
| `VK_PROF` | unset | If set, time the Vulkan expert-group path and report it. |
| `COLI_USAGE` | `<model>/.coli_usage` | Path to the expert-usage history to seed the ranking from, and to write back to. Shared by every engine (`route_trace.h`). |
| `COLI_USAGE_DECAY` | `1.0` (no decay) | Per-run multiplier applied to the recorded counts before ranking, i.e. a half-life. Without one the ranking freezes: after ~18M recorded selections one more turn moves it by 0.2% and the profile stops following the workload (#780). Values outside `(0,1]` are ignored. |
| `USAGE_SAVE` | `1` (on) | `=0` runs read-only — the usage history is loaded but never written back. For benchmark loops that would otherwise skew the profile they are measuring. |
| `RANS_PATH` | auto (best available) | Force a specific rANS kernel (`scalar`, `neon`, `avx512`, …). An unavailable choice yields `invalid` and fails loudly — never a silent downgrade. |
| `RANS_NEON` | on where built | `=0` kill-switch for the NEON rANS path. |
| `RANS_AVX512` | on where built | `=0` kill-switch for the AVX-512 rANS path. |
| `OMP_NUM_THREADS` | unset | Standard OpenMP variable. Setting it disables the engine's own OpenMP hot-thread tuning entirely — the user is assumed to be in charge. |

---

## GLM-5.3-Flash engine (`glm53`)

Read **only** by `c/glm53.c`. Like the other siblings it has its own loader,
cache and precision selection and shares none of the `colibri` knobs above.
See `docs/glm53-flash.md`.

| Variable | Default | Effect |
|---|---|---|
| `GLM53_BITS` | `4` | Precision of the resident dense weights: 4, 8 or 32. Routed experts are not affected — they arrive already quantized in the container and are never requantized. |
| `GLM53_EXPERT_GB` | measured | RAM budget (GB) for the expert LRU cache; per-layer slots are derived from it. Unset, it is taken from reclaimable physical memory after the weights are loaded (Linux `MemAvailable`, Windows available physical memory, macOS free+inactive+purgeable pages), minus a 3 GB margin. A fixed number is wrong in both directions: too small on a large machine leaves memory idle while the disk does all the work. |
| `GLM53_MAXT` | `8192` | KV state capacity in tokens, and the session size in serve mode. |
| `GLM53_PREFILL_CHUNK` | `128` | Prefill chunk size in tokens. Smaller keeps the workspace smaller; too small re-reads experts once per chunk per layer instead of amortizing them. |
| `GLM53_MAX_IMAGE_TOKENS` | checkpoint's (8000) | Ceiling on tokens per image. Each covers 28×28 pixels, so 256 keeps ordinary text legible and 64 keeps shapes and colours. The image is shrunk, not cropped. Lower it: 8000 is 2691 tokens for a 1080p photo, i.e. a prefill nobody will sit through. |
| `GLM53_VERBOSE` | unset | Print the parsed geometry, the expert budget and the per-token cache cost to stderr. |
| `GLM53_DUMP_INDEX` | unset | Print the rows the sparse indexer selected. The first place to look when the engine diverges only at certain lengths. |
| `COLI_VULKAN` | `0` | Route the resident matrices through the shared Vulkan backend. Needs a `VK=1` build and the compiled shaders (`COLI_VK_SHADERS`). Experts stay on the CPU: they arrive from disk on every use, so uploading one costs what reading it costs. |

## Kimi K3 engine (`kimi_k3`)

Read **only** by `c/kimi_k3.c`. The K3 engine has its own loader, cache and quantization selection, so it does not share the `colibri` knobs above.

| Variable | Default | Effect |
|---|---|---|
| `K3_BITS` | `4` | Expert quantization width. Setting it at all also pins the choice (the engine otherwise infers it from the container). |
| `K3_MLA_BITS` | `8` | Quantization width for the MLA attention tensors. |
| `K3_HEAD_BITS` | `8` | Quantization width for the LM head. |
| `K3_MMAP` | `0` (off) | Map fully prepared U8 matrices and F32 sidecars read-only. CPU-only; refuses conversion and enabled GPU backends rather than falling back. |
| `K3_EXPERT_GB` | `8.0` | RAM budget (GB) for the expert LRU cache; per-layer slots are derived from it. |
| `K3_LAYERS` | `0` (all) | Load only the first N layers — for smoke tests and trace-only runs. |
| `K3_MAXT` | `np + ngen` one-shot, `8192` in serve | KV cache capacity in tokens. In serve mode it is also the prompt-rejection bound. |
| `K3_CHUNK` | `32` | Prefill chunk size in tokens. Clamped to [1,512]. |
| `K3_DIRECT` | `1` (on) | Use `O_DIRECT`/unbuffered reads for expert loads. `=0` for buffered. |
| `K3_IDOT` | `1` (on) | Integer dot-product kernels. `=0` uses exact f32 (A/B numerical checks). |
| `K3_PIPE` | `1` (on) | Overlap expert disk-load with compute. `=0` serializes. |
| `K3_LOAD_THREADS` | `4` | Loader threads for the pipe path. Clamped to [1,16]. |
| `K3_DIRS` | unset | Extra shard directories (`;`/`,`-separated) for a multi-drive split, as `COLI_MODEL_DIRS` is for `colibri`. |
| `K3_TOPP` | `0` (off) | Prune routed experts to this cumulative gate weight. A quality lever — A/B it against `K3_LOGITS`. |
| `K3_THINK` | `1` (on) | Emit a reasoning block. `=0` disables. |
| `K3_VK` | `1` (on where built) | Vulkan expert tier. `=0` forces CPU-only. |
| `K3_VK_GB` | `0` (driver budget) | VRAM cap (GB) for the K3 Vulkan expert tier. |
| `K3_VK_UP` | `8` | Expert uploads allowed per step while filling the VRAM tier. |
| `K3_PREFIX_LOG` | unset | Log the KV-prefix reuse decision either way, with the reason when it is "no" — "it did not get faster" is otherwise indistinguishable from "reuse is off". |
| `K3_CHAT_IDS` | unset | Print the chat-template token ids for the built prompt, then continue. |
| `K3_TRACE` | unset | Write a routing trace to `K3_TRACE=<file>`. |
| `K3_LOGITS` | unset | Write per-step logits to `K3_LOGITS=<file>`. |
| `K3_X0` | unset | Read input rows `[T, hidden]` as f32 from this file, bypassing the embedding — for feeding activations captured elsewhere. |

## Inkling engine (`inkling`)

Read **only** by `c/inkling.c`.

| Variable | Default | Effect |
|---|---|---|
| `CTX_MAX` | `8192` | Served KV bound. A prompt plus its requested generation beyond this is rejected rather than truncated. |
| `PIN_N` | `cap / 2` | Experts pinned per layer. Measured on the 975B: `cap/4` (19/layer) gave 83.6% hit / 0.32 tok/s, 40/layer gave 95.6% / 0.80 tok/s — decode fills run at queue depth ~1, so every pinned expert removes a ~35 ms stall. Clamped to `cap - 8`. |
| `REP_PEN` | `1.1` | Repetition penalty over a 128-token history (prompt tail + emitted). |
| `INK_DENSE_Q4` | auto | Use the `dense-int4g64/` sidecar for dense weights when that directory exists. `=0` forces the unquantized dense path. |
| `INK_SHARED_BATCH` | auto | Prefill rows per shared-expert batch, bounded to 64 MiB of scratch. `=0` restores the scalar per-token path for A/B/debugging; a positive value caps the chunk size. Decode (`S=1`) is unchanged. |
| `INK_METAL_MIN_S` | `1` | Minimum batch S to send the MoE block to Metal. `=2` restores the prefill-only gate (which mattered when the residency set was absent and per-block `useResource` churn cost ~135 ms). |
| `INK_PREFIX_LOG` | unset | Log the KV-prefix reuse decision and its reason, as `K3_PREFIX_LOG` does for K3. |
| `COLI_PREFIX_LOG` | unset | Same line for the engines that take the shared record (Qwen3.6, OLMoE): reports how many prompt tokens were reused, or why none were. |
| `COLI_KV_PREFIX` | on, except DeepSeek V4.1 | `0` disables KV-prefix reuse; on `deepseek_v41` reuse is OFF until you set `1`. That engine reads one set of index keys for a prefilled position and another for a decoded one, both the vendor's, so a prefix holding an earlier turn's generated tokens answers differently than the same text read cold. A resumed prefill is exact. |
| `GPU_DEV` | `0` | CUDA device index for the inkling CUDA backend. |
| `NOGPU` | unset | If set, skip GPU init entirely (both CUDA and Metal), regardless of the other GPU variables. |

## Qwen3.6 engine (`qwen36`)

Read **only** by `c/qwen36.c`. See [qwen36.md](qwen36.md) for the model layout
and the CPU/GPU execution split.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DENSE_I8` | `1` (on) | Quantize resident dense matrices to per-row int8 at startup. `=0` keeps the f32 reference path for quality A/Bs. |
| `COLI_DENSE_IDOT` | unset (off) | `=1`: `matmul_d`, the CPU GEMV of the dense trunk, quantizes the activation to int8 once per call and runs integer dot products (`idot.h`: maddubs on AVX2, vpdpbusd on AVX-VNNI / AVX-512 VNNI) instead of converting every int8 weight to f32, and prints `[qwen36] dense trunk on the CPU: integer dot` once. Changes the numbers, hence opt-in (the house rule below); upstream, where it is the default, measured +1.0% perplexity on the 35B. Under the CUDA tier a matrix placed in VRAM keeps the GPU GEMV (attnout, attnproj and dnout only at decode, so a prompt batch takes them on the CPU); the matrices left on the CPU take it. Unset or `=0`: the f32-activation path, byte for byte. See docs/qwen36.md. |
| `QWEN_EXPERT_ACT` | unset (f32) | `=int8`: the routed experts' activation quantized to int8 once per row (`expert_ffn.h` mode 1), with one `[qwen36] routed experts: int8 activations` line. Only where `QWEN_EXPERT_KERNEL` runs, so not under the CUDA expert tier. Changes the numbers, hence opt-in; upstream measured +0.1% perplexity, expert compute 22.7 to 15.9 ms/token. |
| `CACHE_ROUTE`, `ROUTE_J`, `ROUTE_M`, `ROUTE_P`, `ROUTE_ALPHA`, `ROUTE_AGREE` | off; 2, 12, 0, 1, auto | The GLM engine's cache-aware routing (rows in the colibri table above, docs/CACHE_ROUTE.md), read by `qwen36` too: a VRAM-resident expert outranks a RAM-resident one inside the window. Lossy; refused with `CONSIST=1`; for an A/B set `QT_UPLOAD_SYNC=1` in both arms, leave `PILOT` off and give each run a fresh copy of the same `HEAT_FILE`, or none (the engine rewrites it at exit). Under `CACHE_ROUTE=1` the meters are always on (`ROUTE_AGREE=0` does not silence them). `ROUTE_TRACE` is not read by `qwen36`. See docs/qwen36.md. |
| `QT_UPLOAD_SYNC` | unset (off) | `=1`: every expert upload queued so far, including the swaps the layer-0 LFRU pass queues, lands before the next GPU group is formed. Costs the upload/compute overlap; for tests and for a reproducible `CACHE_ROUTE` A/B. |
| `QT_PREFILL_REPLAN` | unset (off) | `=1` (exactly): CUDA expert tier. After each prefill layer, swap the VRAM residents this prompt routed to least for its most-routed non-residents of that layer. The swaps are budget-neutral and upload while the rest of the prefill computes, as many as the 48-entry queue takes; the rest start at the next layer's re-plan or, after the prefill, four per decode token. Each step then rebuilds the int8 RAM copies of swapped-out experts and drops those of swapped-in ones. Port of upstream's re-plan, without its in-place overwrite (no DLL change). Under `CACHE_ROUTE=1` the text can change. Measured on top of `CACHE_ROUTE=1 ROUTE_J=4`: -1.00 ms/token, [-1.26, -0.74], and TTFT +0.21 s, [+0.14, +0.28]: the generation (prefill plus 127 decode steps) about 0.08 s longer at 128 tokens, computed per pair, [+0.01, +0.15], breaking even at about 210 only if the per-token gain holds beyond 128 (not measured); one 25-token prompt, quality not measured; see [qwen36-cuda-tier.md](qwen36-cuda-tier.md#the-residents-follow-the-prompt-qt_prefill_replan1). |
| `QT_PREFILL_REPLAN_MAX` | `24` | Cap on the swaps `QT_PREFILL_REPLAN` plans per prefill layer (`0` plans none). |
| `QT_ASYNC_ISSUE` | unset (off) | `=1` (exactly), **qwen36 only**: the CUDA tier launches each expert group from a helper thread instead of the decode thread. `qt_issue` decides the residents as before, hands the launch over and returns at once, so the decode thread starts the misses and the shared expert while the launch (a ~30 us `cudaGraphLaunch` in the 9 October trace) runs beside it; `qt_take` waits for the helper first. A launch the backend refuses is known only then: its experts come back from `qt_take_redo()` and the engine computes them on the CPU after `qt_take` (counted in `qt_stats`: `async issue: N groups ..., M experts refused and recomputed`). An engine must ask for it with `qt_async_allow()` before `qt_init`; qwen38, which links the same tier but does not recompute refused experts, gets a refusal message instead. One card only (the tested shape); more are refused with a message. **Measured: no gain by itself** (`c/tools/misure-envab.ps1`, six ABBA pairs, ISTRUZIONI.md setting with `COLI_CUDA_FLUSH=1` and the heat table in both arms: `step()` 16.48 -> 16.73, +0.25, 95 % [-0.87, +1.37], 3/6; per-arm means, not paired: `issue` 1.86 -> 0.05, `wait` 0.78 -> 2.69 -- what the decode thread no longer spends launching it waits for the GPU; `docs/experiments/qwen36-async-issue-2026-10-10-raw.txt`). That is what the model below predicts when the GPU side is the longer one. Measured again the same evening with `COLI_CUDA_GROUP_ZC=1` in both arms (same script and setting): `step()` 15.12 -> 14.98, -0.13, 95 % [-1.14, +0.87], 3/6 -- again nothing distinguishable; per-arm means: `issue` 0.82 -> 0.04, `wait` 0.42 -> 1.52 (`docs/experiments/qwen36-group-zc-2026-10-10-raw.txt`, section 4). Kept, off by default, because with it on the decode thread waits ~2 ms/token (unprofiled `wait`; 1.52 with `COLI_CUDA_GROUP_ZC=1`) for the launch on the helper plus the GPU group, so a shorter launch-plus-group could count up to about that -- per layer, through max(C, h + L + P), not one for one. The model: Per layer, with L the launch, C the CPU work, P the time from the launch's return to the results and h the hand-off: synchronous L + max(C, P), asynchronous max(C, h + L + P). The GPU cannot start before the launch whichever thread makes it, so the saving is at most min(L, C - P) on layers where the CPU is the longer side, and about nothing where P >= C; the 9 October run had the two sides about even. It can also lose: a spinning helper shares a core with the 16-thread OpenMP team (no pinning on Windows). Prints `[qtier] QT_ASYNC_ISSUE=1: expert groups launched from a helper thread ...` at startup. Lives in the engine (`qwen36_tier.c`), not in the DLL. `tests/test_qwen36_tier_async.c`: refused without `qt_async_allow()`, same sums as the synchronous path, the launch on another thread, a refusal handed back, 2000 rounds spinning and 2000 sleeping, one card that is device 1 accepted, two cards refused. |
| `QT_ASYNC_SPIN_US` | `2000` | With `QT_ASYNC_ISSUE=1`: how long the helper spins (with a pause instruction) after each job before it sleeps on a condition variable. Decode posts a job every MoE layer, about step/40 apart (~400 us at 16 ms a token), so at the default it spins through a token and sleeps across idle time; `0` sleeps after every job and pays a wake-up, on the GPU's critical path, per layer. Microseconds 0..1000000; any other value is refused with a message and 2000 is used. Not measured. |
| `QT_NO_WARMSTART` | unset (off) | `=1`: skip the tier's warmstart, which otherwise fills the VRAM budget and loads the experts into RAM (every expert, unless `RAM_GB` bounds the cache) before the first token. Then the RAM cache fills lazily. |
| `QWEN36_AVX512` | `1` (on where built) | On a build whose `-march` includes AVX-512 (`ARCH=native` on Zen 4/5 or a Xeon SP), the int8 GEMVs — the dense projections and the grouped expert kernel — take a 512-bit inner loop (`simd_i8f.h`) instead of the AVX2 one. `=0` restores the AVX2 kernels on the same binary, which is how to A/B it. The two differ in summation order, so results differ in the last bits; both are float accumulation of the same products. A build without AVX-512 ignores this variable. |
| `QWEN36_QUANT_AT_LOAD` | `1` (on) | With `COLI_DENSE_I8` on, quantize each dense matrix right after it is read and release its f32 block, instead of loading the whole f32 trunk and quantizing afterwards: the f32 trunk and its int8 copy are never resident together, so peak RSS drops by roughly the f32 trunk size. Same int8 bytes (ids and logits byte-identical, pinned in CI). `=0` restores the old order. Off under `COLI_KEEP_F32`. |
| `QWEN36_EMBED_I8` | `0` (off) | Opt-in. Quantize the token embedding to per-row int8 (~1.9 GB -> ~0.47 GB on the 35B) and dequantize one row per token. **Changes numerics** (every input row is rounded): measure output quality before relying on it. |
| `QWEN_EXPERT_KERNEL` | `1` (on) | Routed experts run through the shared `expert_ffn.h` kernel: the int4 stays packed in RAM (planar layout, half the expert-cache RSS of the int8 unpack), gate+up are one pass, and a layer is two OpenMP regions over (expert, row-chunk) items instead of 3 x top-k GEMV regions. Takes effect on an int4 gs=64 container whose hidden and expert widths are multiples of 64, and not under the CUDA expert tier. `=0` restores the unpack-to-int8 path; the two produce the same tokens (1024-token decode on the real container byte-identical; pinned on the tiny int4 fixture in CI), only the f32 accumulation order inside a dot differs. Measured at cap 256 on the real container: 12.8 -> 15.7 tok/s, peak RSS 29 -> 17 GB. |
| `QWEN_DENSE_BATCH` | `1` (on) | On AVX2/FMA, reuse each dense-int8 weight decode across two prompt rows. `=0` restores one GEMV call per row. Decode `S=1` is unchanged. |
| `QWEN_SHARED_BATCH` | bounded by 32 MiB scratch | Batch the CPU shared expert across prompt rows. `=0` restores scalar calls; a positive integer caps rows per chunk. The CUDA-tier overlap path is unchanged. |
| `QWEN36_CONV_OMP` | `1` (on) | The DeltaNet causal depthwise conv (and its ring advance, fused into the same pass) runs across threads instead of on one core. Channels are independent (`groups=conv_dim`), so it is bit-exact — same operations, same order, spread over a team; CI pins ids and logits on a fixture whose `conv_dim` clears the `>= 256` threshold. It was serial on the reasoning that ~33k FLOP cannot be worth a fork/join, but the cost is one `expf` per channel, and the engine's own timer puts the block at 5.2-5.9 ms/token on the 35B — 180 us per layer on one core. `=0` restores the serial pass. |
| `Q36_MAXT` | conservative engine default | Lower the served/context capacity; it cannot raise the model's compiled safety ceiling. |

## Qwen3.8 engine (`qwen38`)

Read **only** by `c/qwen38.c`. See [qwen38.md](qwen38.md) for the native FP8
checkpoint layout and the text-only capability boundary.

| Variable | Default | Effect |
|---|---|---|
| `Q38_MAXT` | `8192` | Served context capacity. Values above the model's native 262,144-token limit are clamped; malformed or non-positive values restore the default. |
| `Q38_EOS` | tokenizer/config stop IDs | Override the served end-of-sequence token ID for controlled experiments. Normally the engine stops on the tokenizer's `<|im_end|>` / `<|endoftext|>` IDs, falling back to `eos_token_id`. |
| `Q38_NATIVE_FP8` | `1` (on) | Keep routed E4M3 expert bytes and their F32 128×128 block scales native in the LRU. `=0` restores expanded-FP32 slots for A/B validation. |
| `Q38_NATIVE_BF16` | `1` (on) | Keep resident and routed BF16 matrices in two-byte storage while retaining FP32 activations/accumulation. `=0` restores the expanded-FP32 reference. |
| `Q38_PREFILL_BATCH` | `1` (on) | Route prompt rows in bounded expert-major chunks and batch resident shared-expert/DeltaNet projections. `=0` restores row-at-a-time prompt execution for A/B diagnosis; decode is unchanged. |
| `COLI_TIMERS` | `0` (off) | Set to `1` for the detailed Qwen3.8 phase breakdown on stderr. The shared per-request `PROF` frame is emitted regardless. |

## DeepSeek V4 engine (`deepseek_v4`)

The V4 engine has its own knob set (~70 variables: GPU tier, prefill segments/
chunks, prefix checkpoints, expert I/O, speculative decoding, profilers). It is
documented with defaults in
[deepseek-v4.md — Environment reference](deepseek-v4.md#environment-reference-v4-engine);
the ones you are most likely to set: `DSV4_CUDA` (GPU tier on/off),
`COLI_CUDA_ATTN_BATCH=1`, `COLI_CUDA_MOE_BATCH=1`, `DSV4_CUDA_EXPERT_MIRRORS`,
`V4_MOE_REFILL_GROUP`, `V4_PREFILL_SEGMENT`, `V4_PREFIX_CKPT*`, `CTX`.
`COLI_V4_SAVE_USAGE=0` is an engine-specific alias that disables only V4's
usage rewrite; the shared `USAGE_SAVE=0` covers this engine too.

| Variable | Default | Effect |
|---|---|---|
| `COLI_V4_ROWS16` | `1` (on) | Repack hot-pinned experts into the vectorized `rows16` layout. **While this is on, greedy output varies run to run on the same machine** (#1136): rows16 and the reference matvec accumulate in different orders, and which experts take which kernel follows the expert-cache state. `=0` runs the reference matvec for every expert — slower, but the kernel variable is gone. **Set `=0` for any quality A/B on this engine**; throughput A/Bs do not need it. |

**Reproducible greedy runs (#1136):** greedy text on this engine varies with
the expert-cache state — hot experts run the vectorized `rows16` kernel, cold
ones run the reference matvec, the two accumulate in different orders, and
which experts are hot follows the autopin history (`.coli_usage`, rewritten by
every run). This is a known defect, not a documented trade-off — the house
rule since the olmoe/inkling IDOT cases (#1044, #1080) is that a fast path
which changes tokens is opt-in, and a convergence fix (reference path adopting
rows16's accumulation order) is planned under #1136. Until it lands: for
byte-identical output across runs, either freeze the history (`USAGE_SAVE=0`,
after seeding it once) or remove the variable entirely
(`COLI_V4_ROWS16=0 COLI_V4_AUTOPIN=0 USAGE_SAVE=0`: reference kernels only, no
history). Details in [deepseek-v4.md — CPU-only behaviour](deepseek-v4.md).

## OLMoE engine (`olmoe`)

Read **only** by `c/olmoe.c`. This is the sister engine used for streaming-cache research, so most of these are experiment knobs.

| Variable | Default | Effect |
|---|---|---|
| `CHAT` | unset | Interactive chat mode; bypasses the `ref.json` harness entirely. |
| `MAX_NEW` | `512` | Max tokens to generate in chat mode. |
| `HOT` | `0` | Number of hottest experts to pin at startup. |
| `WARMUP` | `5` | Tokens observed before the hot set is considered learned. |
| `WIDE` | `1` | Router width multiplier for the prefetch prediction. Clamped to [1,4]. |
| `SMOOTH` | `0.3` | EMA factor for routing momentum (gate logits smoothed across tokens). Clamped to [0, 0.95]. |
| `CONF_LIMIT` | `0.92` | Confidence ceiling for the router prediction. Clamped to [0.1, 1.0]. |
| `EXPERT_DROP` | `0` (off) | Drop experts below the confidence threshold instead of loading them (quality/speed experiment). |

---

## Server / CLI (`openai_server.py`, `coli`)

These are read by the Python programs (not the `glm` engine), so they don't appear in `glm.c`. They cover the OpenAI-compatible server, tool calling, and the debug view.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DEBUG` | `0` (off) | Tee the engine transaction to stderr, by level. **`1`** = decoded model output stream only (byte-by-byte, on both the tool-call and plain paths). **`2`** = both sides — the fully-rendered prompt the engine received *and* the output, bracketed and correlated by request id, so stderr reads as the whole conversation. Invaluable for seeing what the model received vs. emitted during an OpenCode session. |
| `COLI_TOOL_SALVAGE` | `0` (off) | Opt-in de-mangler: reconstruct a malformed int4 tool call by mapping its lone payload onto the tool's primary parameter. Never rewrites well-formed output; recommended for int4 deployments. |
| `COLI_THINK` | `0` (off) | Make thinking the default when the client sends *neither* `reasoning_effort` nor `enable_thinking`. Any explicit client value still wins. |
| `COLI_MODEL` | unset | Default model directory (fallback for `--model`). |
| `COLI_MODEL_ID` | `glm-5.2-colibri` | Model id reported by the API. |
| `COLI_API_KEY` | unset | Required bearer token for the server. |
| `COLI_IMAGE_ROOT` | unset (local paths denied) | Directory under which an `image_url.url` naming a local path or `file://` URI may be read. Unset, the server refuses local paths: a client sends images as base64 `data:` URIs (`coli chat` and `coli web` do), because a file read here happens with the server's own rights and an inference client is not the operator. Set it to allow paths under one directory only; symlinks are resolved before the check. |
| `COLI_ALLOWED_HOSTS` | unset | Comma-separated hostnames or IP addresses accepted by the DNS-rebinding guard in addition to loopback and the bind address. Equivalent to repeating `--allowed-host`. |
| `COLI_MAX_QUEUE` | `8` | Max queued requests. |
| `COLI_QUEUE_TIMEOUT` | `300` | Seconds a request may wait in the queue. |
| `COLI_KV_SLOTS` | `1` | Independent KV conversation slots (→ engine `KV_SLOTS`). |
| `COLI_POLICY` | `quality` | Resource policy (shared with the engine): `quality` \| `balanced` \| `experimental-fast`. |
| `COLI_CHAT_STATS` | `full` | Default for `coli chat --stats`: the footer after each answer. `full` = tokens, seconds, tok/s; `compact` = tokens, tok/s; `off` = no footer. Counts are exact (no `~`) when the server reports `completion_tokens` in the streamed usage block, the chars/4 estimate otherwise. The flag wins over the variable. |
| `COLI_COLOR` | auto (TTY) | `COLI_COLOR=1` forces colored `coli` output when not a TTY. |
| `COLI_RAW` | `0` | `coli` raw output mode. |

> **Debugging an OpenCode session:** `COLI_DEBUG=1` watches the model's output stream; `COLI_DEBUG=2` shows both sides (prompt + output) as a transcript. Add `COLI_TOOL_SALVAGE=1` on int4 to catch mangled tool calls.

## Set by the CLI (don't usually set by hand)

`coli` / `openai_server.py` set these internally to select a run mode or pass through a flag:

- `SNAP` — model snapshot directory (required by `glm`; set from `--model`).
- `SERVE`, `SERVE_BATCH` — select serve / batched-serve mode.
- `PROMPT` — one-shot text mode (the engine also honors `COLI_PROMPT`, preferred cross-platform; `PROMPT` is ignored on Windows if it contains cmd.exe `$`-metacharacters).
- `COLI_OMP_TUNED` — internal sentinel guarding the OMP re-exec (see `COLI_NO_OMP_TUNE`); not user-facing.

---

## Worked example — the fast, reproducible Apple-Silicon config

```bash
# fast (sampling, non-deterministic by design):
COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"

# same, but reproducible (greedy):
COLI_TEMP=0 COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"
```
| `V41_ENGRAM_ROWS` | 65536 | DeepSeek V4.1: rows of engram cache per table. The n-gram traffic is Zipfian, so a small cache absorbs most of it; 65536 rows is 64 MB per table on the released head_dim. |
| `V41_INDEX_OWNER` | unset | DeepSeek V4.1: score each layer against its OWN index keys instead of the last published cache. The default reproduces the released inference code; this changes the model's behaviour, see docs/deepseek-v41.md. |
| `V41_MAX_IMAGE_TOKENS` | the checkpoint's `max_image_tokens` | DeepSeek V4.1: ceiling on what one image costs in prompt tokens. |
| `V41_TRACE` | unset | DeepSeek V4.1: print per-sublayer checksums, matching tools/dsv41_ref.py's, to locate a divergence by diffing two columns. `2` follows the first row of a speculative step rather than the last. |
| `V41_DSPARK` | on when the checkpoint carries the head | DeepSeek V4.1: `0` disables the DSpark draft head, which is then not loaded. Drafts never change what a turn produces, only how many forwards it takes: measured +17% on the real checkpoint from a cold cache (24 tokens in 99.3 s against 116.6). |
| `V41_DSPARK_MAX` | the checkpoint's `dspark_block_size` | DeepSeek V4.1: how many drafted tokens go in front of the main model per round. Fewer costs less when a round is rejected and caps the win when it is not. |
| `V41_DSPARK_MINACC` | 60 | DeepSeek V4.1: percent of drafts that must be accepted over a window of ten before drafting pauses for 64 tokens. 60 is the measured break-even. |
| `V41_SPEC_FORCE` | unset | DeepSeek V4.1, oracle mode only: draft the reference's own tokens (`1`), corrupt the last one (`2`), or keep the head's (`3`), so the verification path runs on a fixture whose draft head is random noise. |
