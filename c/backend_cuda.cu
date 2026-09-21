#include "backend_cuda.h"

#include "backend_gpu_compat.h"

/* Optional fmt=8 decode candidate (COLI_CUDA_F8_WARP=2): cuda_fp8.h maps
 * __nv_cvt_fp8_to_halfraw to an sm_89+ cvt instruction, with a bit-manip
 * fallback below 890. CUDA-only; the HIP build keeps the LUT decode. */
#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)) && defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define COLI_F8_HWCVT 1
#else
#define COLI_F8_HWCVT 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef COLI_ANS
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <sys/stat.h>
#endif

struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float **latent_pages,**rope_pages;
    int length,page_count,K,R;
};

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

static void ragged_kv_clear(RaggedKVEntry *e) {
    for (int i=0;i<e->page_count;i++) {
        if (e->latent_pages[i]) cudaFree(e->latent_pages[i]);
        if (e->rope_pages[i]) cudaFree(e->rope_pages[i]);
    }
    std::free(e->latent_pages);
    std::free(e->rope_pages);
    e->latent_pages=e->rope_pages=nullptr;
    e->length=e->page_count=0;
}

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    size_t scale_count;        /* floats in `scales`: O per-row, O*ng grouped */
    int tracked;
    int weights_owned;
#ifdef COLI_ANS
    size_t archive_bytes;
    int compressed;
#endif
    RaggedKVEntry ragged[512];
    int ragged_count;
};

#ifdef COLI_ANS
struct AnsArenaChunk { uint8_t *p; size_t used,cap; };
#endif
typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    /* Staging of the resident dense matvec (coli_cuda_matmul), apart from
     * x/y: the expert group (coli_cuda_expert_group_issue) runs on ctx->stream
     * asynchronously while the engine's thread keeps computing -- qwen38's
     * shared expert and the Qwen3.8 trunk go through coli_cuda_matmul between
     * qt_issue and qt_take. Sharing x/y there overwrote the group's input and
     * output mid-flight (silent, output-only corruption, no CUDA error). */
    float *dx, *dy; size_t dx_cap, dy_cap;
    /* COLI_CUDA_DENSE_PINNED: host side of that same staging. A cudaMemcpy
     * out of PAGEABLE memory is not one copy, it is two -- the driver stages
     * through a pinned buffer of its own -- and it blocks the calling thread
     * until the whole thing is done. This path runs ~80 times a token and
     * both of its copies are that kind, which is the measurement that made
     * these fields worth adding.
     *
     * They cannot share host_x/host_y for the same reason dx/dy cannot share
     * x/y: the expert group owns those while it is in flight on ctx->stream,
     * and the engine calls coli_cuda_matmul between qt_issue and qt_take. */
    float *host_dx, *host_dy; size_t host_dx_cap, host_dy_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv; size_t host_x_cap,host_y_cap,host_kv_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[27]; size_t pipe_cap[27];   /* scratch persistenti del resident pipeline */
    cudaStream_t stream;
    cudaEvent_t ev_done; int ev_done_ok;        /* resident-group issue completion (#431 PR-C0) */
    /* COLI_CUDA_PROFILE on the ASYNC group path. The sync path creates and
     * destroys four events per call; this one runs once per MoE layer per
     * token, so its events are created once and kept. `group_timed` says
     * whether the call now in flight recorded them, because take() reads the
     * elapsed times and must not read stale ones from an untimed call.
     * ev_grp_ok is tri-state -- 0 untried, 1 ready, -1 creation failed --
     * because "not yet" and "won't work" must not share a value here: the
     * events are persistent, so retrying would overwrite live handles. */
    cudaEvent_t ev_grp[4]; int ev_grp_ok, group_timed;
    /* Same tri-state as ev_grp, for the resident dense GEMV path. */
    cudaEvent_t ev_dm[4]; int ev_dm_ok;
    void *group_desc; size_t group_desc_cap;
    /* Host side of the descriptor upload. It used to be a stack local in
     * coli_cuda_expert_group_issue_x, which a CUDA graph cannot carry: a graph
     * records the memcpy's SOURCE ADDRESS, and a stack frame is gone by the
     * time the graph replays. Pinned and per-context, so the address is fixed
     * for the life of the process and only the CONTENT changes between
     * launches -- which is exactly what a replayed graph needs. */
    void *host_desc; size_t host_desc_cap;
    /* COLI_CUDA_GRAPH: one instantiated graph per `count` (1..8 resident
     * experts), captured lazily on the first launch at that count.
     *
     * A graph is a promise that nothing about the sequence changes except the
     * bytes in the buffers it points at. Two things can break that promise,
     * and both are checked rather than assumed: a reserve() that reallocates
     * one of x/y/gate/up/group_desc (buf_gen counts those, and a mismatch
     * throws the graphs away), and the profiling events, which the graph path
     * refuses to run under at all -- four cudaEventRecord inside a capture
     * would bake the timing branch into the graph, so COLI_CUDA_PROFILE=1
     * takes the ordinary path and measures the ordinary path. */
#if COLI_GPU_HAS_GRAPH
    cudaGraphExec_t graph_exec[9]; unsigned graph_gen[9];
#endif
    unsigned buf_gen;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
#ifdef COLI_ANS
    void *ans_raw; size_t ans_raw_cap;
    void *ans_host; size_t ans_host_cap;
    int ans_copy_pending;
    dietgpu::StackDeviceMemory *ans_scratch;
    std::vector<AnsArenaChunk> *ans_chunks;
#endif
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
    /* Row offset into the group's INPUT buffer, which `offset` used to serve
     * as well. They are the same for every caller that hands over one input
     * row per expert -- and they differ for the one that does not: in decode
     * all K experts of a token consume the SAME activation vector, so the
     * caller can upload it once and point every chunk at row 0. `offset`
     * keeps driving gate/up/y, which are genuinely per-expert and must not
     * collide.
     *
     * SET IT EXPLICITLY, EVERYWHERE. C++ aggregate initialisation zero-fills
     * a trailing field nobody mentions, and zero here means "read row 0" --
     * a broadcast nobody asked for, silent at compile time. Adding this field
     * did exactly that to five hand-built GroupDesc literals in tests/, and
     * test_fp8_cuda caught it at 76777 mismatches. The tests are the gate;
     * the alternative design, if this ever bites again, is a kernel parameter
     * instead of a struct field, because then every missed site is a compile
     * error rather than a wrong number. */
    int xoff;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
/* Resident dense GEMVs (coli_cuda_matmul): lm_head, dnproj, dnout, attnout,
 * attnproj. Measured separately from the expert groups because the question is
 * different -- the groups are about launch overhead on 40 calls a token, these
 * are about whether a 24 MB GEMV runs anywhere near the card's bandwidth.
 *
 * That question came from dividing the engine's existing timers by their byte
 * counts, and one of those two timers is not the clean divisor it looks like:
 * g_dn_sub[0] (qwen36.c) closes after dn_b and dn_a as well as the fused
 * projection, two CPU-only matmuls whose bytes are in nobody's numerator. They
 * are a low single-digit share of the work at these shapes, so the ~67 GB/s
 * that motivated this patch is a FLOOR on the dnproj kernel, not a reading of
 * it. Which is the whole point of measuring here instead: these counters time
 * the GEMV and nothing else.
 *
 * wall_ms is the H2D -> D2H window, NOT the whole function: the tensor upload,
 * find_ctx/select_ctx and the dx/dy reserve all sit before it. That is
 * deliberate rather than lazy -- the upload is a one-off that moves the entire
 * matrix, and folding it in would poison every average with the first call.
 * What is left is the steady-state round-trip, which is the thing in question:
 * wall minus the three GPU phases is what the CPU spends not computing. */
static uint64_t g_dense_calls;
static double g_dense_h2d_ms, g_dense_kernel_ms, g_dense_d2h_ms, g_dense_wall_ms;
static uint64_t g_dense_bytes;            /* weights AND scales the kernel reads */
/* Per device as well as in total, for the reason coli_cuda_group_stats already
 * has a _device form: auto_place splits the trunk across cards by budget, so on
 * the mixed pair docs/qwen36-cuda-tier.md measures (4070 Ti + Quadro RTX 4000)
 * a single blended GB/s describes neither of them. Aggregating is the caller's
 * choice here, not this counter's. */
static uint64_t g_dev_dense_calls[COLI_CUDA_MAX_DEVICES];
static double g_dev_dense_h2d_ms[COLI_CUDA_MAX_DEVICES], g_dev_dense_kernel_ms[COLI_CUDA_MAX_DEVICES];
static double g_dev_dense_d2h_ms[COLI_CUDA_MAX_DEVICES], g_dev_dense_wall_ms[COLI_CUDA_MAX_DEVICES];
static uint64_t g_dev_dense_bytes[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_calls[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_experts[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_rows[COLI_CUDA_MAX_DEVICES];
static double g_device_group_h2d_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_kernel_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_d2h_ms[COLI_CUDA_MAX_DEVICES];
static std::mutex g_group_stats_mu;
#ifdef COLI_ANS
static FILE *g_ans_sidecar;
static int g_ans_sidecar_pack;
#if defined(__linux__)
static int g_ans_direct_fd=-1;
static off_t g_ans_direct_off;
#endif
static uint64_t g_ans_load_records;
static double g_ans_header_s,g_ans_read_s,g_ans_stage_s,g_ans_enqueue_s;
static int g_ans_profile_printed;
static double ans_now_s(){
    using clock=std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
#endif

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();   /* consume the sticky error: a failed call must
                                   not poison the next launch's error check */
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 7) return (size_t)(I + 1) / 2;   /* MXFP4: e2m1 nibbles, 2 per byte */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    if (fmt == 8) return (size_t)I;             /* fp8-e4m3: raw bytes, layout of fmt=1 */
    return 0;
}

/* The E8 codebook, uploaded once per device from quant.h's e8_grid so the table
 * has a single source of truth and cannot drift from the CPU decoder. */
__constant__ uint8_t c_e8_grid[256][4];

/* The fmt=8 e4m3 decode table, uploaded once per device from quant.h's
 * E4M3_LUT — same single-source-of-truth arrangement as c_e8_grid. Uploads of
 * fmt=8 tensors are refused until it is published (g_fp8_lut_ready): a kernel
 * reading the zero-initialized table would compute silent zeros, the exact
 * failure mode this format's dispatch work exists to prevent. */
__constant__ float c_e4m3[256];
static int g_fp8_lut_ready;

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__device__ __forceinline__ uint32_t e8_ld_u32(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__device__ __forceinline__ float e8_fp16(const uint8_t *p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__device__ __forceinline__ void e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid[idx[l*2+0]], *g1=c_e8_grid[idx[l*2+1]];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

/* ---- MXFP4 (OCP microscaling FP4), fmt=7 -----------------------------------
 * Same layout the CPU path decodes in quant.h's matmul_mxfp4, and the same two
 * tricks, so the two agree bit for bit:
 *
 *   packed [O, I/2]  u8 — e2m1 nibbles, LOW nibble = even column, bit3 = sign,
 *                         bits 0..2 index {0,.5,1,1.5,2,3,4,6}
 *   scales [O, I/32] u8 — ue8m0 exponent per 32-column group, w = v * 2^(s-127)
 *
 * The exponent is decoded as a bit pattern rather than exp2f: (uint32)s << 23
 * reinterpreted as float IS 2^(s-127) for s in [1,254], and reproduces the CPU
 * path's documented edge behaviour exactly -- s=0 gives +0 and s=255 gives +inf
 * on both sides. Using exp2f here would agree for the normal range and diverge
 * at the ends, which is precisely where a silent mismatch would hide.
 *
 * The LUT holds DOUBLED values so every entry is an exact small integer; the
 * compensating 0.5f rides along in mx4_scale_dev, as it does on the CPU. */
/* Decoded arithmetically rather than from a __constant__ table: a file-scope
 * __constant__ array with static linkage is initialised per translation unit,
 * and this kernel is also compiled into the HIP build and the DLL, where that
 * silently yields garbage. The magnitude is 2^(exp-1) * 0.5 for exp in 1..3 and
 * 0 for exp 0, which is exactly the OCP e2m1 table {0,.5,1,1.5,2,3,4,6}. */
__device__ static inline float mx4_decode(int n) {
    int mant = n & 1, exp = (n >> 1) & 3;
    float mag = exp ? ldexpf(1.0f + 0.5f * (float)mant, exp - 1) : 0.5f * (float)mant;
    return (n & 8) ? -mag : mag;
}

__device__ static inline float mx4_scale_dev(uint8_t s) {
    union { uint32_t u; float f; } b;
    b.u = static_cast<uint32_t>(s) << 23;
    return b.f;
}

/* e2m1 nibble at column i of a packed row. */
__device__ static inline float mx4_weight_at(const uint8_t *q, int i) {
    uint8_t v = q[i >> 1];
    return mx4_decode((i & 1) ? (v >> 4) : (v & 15));
}

/* Generic per-element weight decode for the kernels that need one (the absorb
 * kernels and the generic grouped-expert path). fmt=3 (int2) is now an EXPLICIT
 * branch and the fall-through is a refusal.
 *
 * It used to be the other way round: int2 was the fall-through, so every format
 * this function does not decode -- fmt=5 (int3-g64), fmt=6 (E8/IQ3), fmt=8
 * (fp8-e4m3), and anything added later -- was read as 2-bit values and returned
 * numbers. Meanwhile the CPU functions doing the same job on the same tensor,
 * qt_addrow and qt_matvec_rows (colibri.c), both exit(1) naming the function and
 * the fmt. Two backends, identical unsupported input, one refusing and one
 * fabricating: that asymmetry is the defect, independent of any particular
 * format's arrival.
 *
 * WHY __trap() AND NOT A DIAGNOSTIC. This is device code inside a running
 * kernel; there is no stderr to name the tensor on and no way to unwind. __trap
 * aborts the kernel and poisons the context, so the next cuda_ok() call on the
 * host reports a failure instead of the caller consuming fabricated values --
 * the same "stop rather than misread" outcome as the CPU's exit(1), reached the
 * only way device code can reach it.
 *
 * IT IS A BACKSTOP, NOT THE PRIMARY GATE -- and the gates above it are several
 * DIFFERENT checks, not one. Naming them exactly, because "every launcher checks
 * coli_cuda_weight_at_supported" would be false and a reader will verify it:
 *   - UPLOAD is the widest gate. coli_cuda_tensor_upload{,_g} refuse when
 *     row_bytes(fmt,I) == 0, which is every fmt this file has no row stride for --
 *     fmt=5 (int3-g64), every negative fmt, every unknown fmt. Those can never
 *     become a ColiCudaTensor at all, so the uploadable set is {0,1,2,3,4,6,7,8}.
 *   - quant_matmul dispatches 6, 7, 4 and 8 in explicit branches of its own, so
 *     the generic else that calls this function sees only 0/1/2/3 out of that set.
 *   - the generic grouped-expert path refuses gf>3 || uf>3 || df>3 on the host.
 *   - the absorb call sites (the other caller of this function) are the ones
 *     gated by coli_cuda_weight_at_supported, via absorb_fmt_ok below -- one
 *     caller, not "every launcher".
 * Each of those is sufficient on its own for the sites it covers; this trap
 * exists because none of them is a property of THIS function. Reaching this line
 * means a launch site got past its own gate, which is a bug in that gate. */
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2 || fmt == 4) {                               /* fmt=4: same nibble layout */
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    if (fmt == 3) {                                           /* int2 */
        uint8_t v = q[i >> 2];
        return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
    }
    __trap();
    return 0.0f;   /* not reached: __trap() does not return */
}

/* Scale for output `row`, input element `k`. fmt=4 (grouped int4) stores ng
 * scales per row at scales[row*ng + k/gs]; every other quantized format has
 * one scale per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
__device__ static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

/* ---- fmt=8 (fp8-e4m3) warp decode/accumulate helpers -----------------------
 * The fmt=8 dot products are compared against the CPU reference (quant.h
 * matmul_fp8) by the cross-tier parity tests, so this TU must never be built
 * with fast-math: nvcc's --use_fast_math implies -ftz=true, which flushes the
 * scale*subnormal contributions the denormal-scale test pins down. (-ftz has
 * no macro of its own to test; the Makefile pins -ftz=false on the nvcc line.) */
#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "backend_cuda.cu: fmt=8 kernels forbid fast-math builds (FTZ breaks CPU parity)"
#endif

/* Fixed-order 5-shuffle reduce over a 32-lane logical warp. Width pinned to 32
 * so a HIP wave64 device does not fold two output rows into one reduction. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define f8_shfl_down(v,o) __shfl_down((v),(o),32)
#else
#define f8_shfl_down(v,o) __shfl_down_sync(0xffffffffu,(v),(o),32)
#endif

/* Decode one e4m3 byte. Default: a __shared__ copy of c_e4m3 — per-thread
 * data-dependent indices are the __constant__ cache's documented worst case
 * (accesses to different addresses within a warp serialize), while shared
 * memory takes them at full rate. HW=1 (COLI_CUDA_F8_WARP=2, CUDA-only): the
 * cuda_fp8.h route. Every finite e4m3 value is exactly representable in f16,
 * so the two paths should agree bit for bit — but the on-device 256-value
 * sweep is the authority, and the LUT stays the default until the sweep
 * certifies the cvt path on the target silicon. */
template<int HW>
__device__ __forceinline__ float f8_dec(const float *slut, uint8_t b){
#if COLI_F8_HWCVT
    if (HW) return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
#endif
    return slut[b];
}

/* One 128-column scale block, one warp: 32 lanes x 4 bytes is the exact
 * FP8_BLOCK fit. Weights are decoded once and reused across ns activation
 * rows (weights are the traffic; activations come from cache), then each
 * row's per-lane products are tree-reduced in fixed order — the block partial
 * lands in LANE 0 of gp[]/up_[] (other lanes hold partials). vec picks the
 * LOAD width only (uchar4/float4 on aligned full blocks, guarded bytes on
 * tails and unaligned bases — e.g. zero-copy views: correct, slower); both
 * load paths feed ONE shared product/accumulate sequence below, and the
 * vec/byte parity test asserts their bit-equality on identical data. Tail
 * padding contributes +0.0f terms, which can at most flip a -0.0 partial to
 * +0.0. DUAL folds a second weight row (gate+up) against the same
 * activations without re-reading x. The per-row arithmetic never depends on
 * ns or s, so S-tiling cannot reorder a dot (B5). */
template<int HW,int DUAL>
__device__ __forceinline__ void f8w_block(const float *slut,const uint8_t *gr,
        const uint8_t *ur,const float *xs,int xstride,int ns,int base,int len,
        int vec,float *gp,float *up_){
    int lane=threadIdx.x&31,i0=base+lane*4,full=vec&&len==128;
    float gw[4],uw[4]={0,0,0,0};
    if(full){
        uchar4 q=*(const uchar4*)(gr+i0);
        gw[0]=f8_dec<HW>(slut,q.x);gw[1]=f8_dec<HW>(slut,q.y);
        gw[2]=f8_dec<HW>(slut,q.z);gw[3]=f8_dec<HW>(slut,q.w);
        if(DUAL){uchar4 r=*(const uchar4*)(ur+i0);
            uw[0]=f8_dec<HW>(slut,r.x);uw[1]=f8_dec<HW>(slut,r.y);
            uw[2]=f8_dec<HW>(slut,r.z);uw[3]=f8_dec<HW>(slut,r.w);}
    }else for(int k=0;k<4;k++){
        int in=i0+k<base+len;
        gw[k]=in?f8_dec<HW>(slut,gr[i0+k]):0.f;
        if(DUAL)uw[k]=in?f8_dec<HW>(slut,ur[i0+k]):0.f;
    }
    for(int s=0;s<ns;s++){
        const float *xr=xs+(size_t)s*xstride;
        float xv[4];
        if(full){
            float4 xf=*(const float4*)(xr+i0);
            xv[0]=xf.x;xv[1]=xf.y;xv[2]=xf.z;xv[3]=xf.w;
        }else for(int k=0;k<4;k++) xv[k]=i0+k<base+len?xr[i0+k]:0.f;
        float g=0,u=0;
        for(int k=0;k<4;k++){g+=xv[k]*gw[k];if(DUAL)u+=xv[k]*uw[k];}
        for(int off=16;off;off>>=1){g+=f8_shfl_down(g,off);if(DUAL)u+=f8_shfl_down(u,off);}
        gp[s]=g;if(DUAL)up_[s]=u;
    }
}

/* Dense fmt=8 rework: quant_matmul's fmt=8 branch as its OWN kernel, so
 * COLI_CUDA_F8_WARP=0 restores the fully original dense behavior too (the
 * host picks in quant_matmul_launch). Same grid/block contract as
 * quant_matmul: one 256-thread block per (o,s). Accumulation mirrors the CPU
 * reference: f32 partial per 128-block (f8w_block), the scale applied ONCE
 * per partial, double across blocks. Warps stride the block axis; the
 * cross-warp double sum runs in fixed warp order, so the reduction order is
 * a pure function of the dims. Decode reads a shared copy of c_e4m3
 * (data-dependent __constant__ indices serialize); the cvt candidate stays
 * grouped-path-only until certified. NaN bytes decode to NaN and propagate,
 * same policy as the CPU path. */
__global__ static void quant_matmul_f8w(float *y,const float *x,const void *weights,
                                        const float *scales,int S,int I,int O){
    int o=blockIdx.x,s=blockIdx.y;
    const float *xs=x+(size_t)s*I;
    __shared__ float slut[256];
    __shared__ double dsum[32];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) slut[i]=c_e4m3[i];
    __syncthreads();
    const uint8_t *wrow=(const uint8_t*)weights+(size_t)o*I;
    const float *scl=scales+(size_t)(o>>7)*(size_t)((I+127)>>7);
    int warp=threadIdx.x>>5,nw=blockDim.x>>5,nblk=(I+127)>>7;
    int vec=!(I&3)&&!((size_t)wrow&3)&&!((size_t)xs&15);
    double a=0;
    for(int bi=warp;bi<nblk;bi+=nw){
        int base=bi<<7,len=I-base<128?I-base:128;
        float p;
        f8w_block<0,0>(slut,wrow,nullptr,xs,0,1,base,len,vec,&p,nullptr);
        if(!(threadIdx.x&31)) a+=(double)p*scl[bi];
    }
    if(!(threadIdx.x&31)) dsum[warp]=a;
    __syncthreads();
    if(!threadIdx.x){
        for(int w=1;w<nw;w++) a+=dsum[w];
        y[(size_t)s*O+o]=(float)a;
    }
    (void)S;
}

/* fmt=1 per-row int8 dense GEMV, R OUTPUT ROWS PER BLOCK (COLI_CUDA_I8_ROWS).
 *
 * The generic branch below gives one 256-thread block to one output row, and
 * every block reads the WHOLE activation vector. So a call moves
 *
 *     weights   I * O * 1 byte
 *     x         I * O * 4 byte      (O blocks, I floats each)
 *
 * -- four fifths of the traffic is activations, and dense_stats counts only
 * the weights. With R rows per block, x is read once per R rows and the total
 * falls from 5x the weight bytes to (1 + 4/R)x: at R=8 that is 1.5x, a factor
 * of 3.3. The qwen3.6 trunk measures 29 % of this card's bandwidth -- 1/3.4 --
 * so this is the first hypothesis that predicts the observed number rather
 * than merely being plausible.
 *
 * WHAT NSIGHT COMPUTE ACTUALLY SAYS, measured on a dnproj call (grid 12288,
 * 113 us): occupancy 91.9 %, DRAM throughput 34.6 %, and 60 % of a 27-cycle
 * issue gap spent stalled on an L1TEX scoreboard -- 0.64 eligible warps per
 * scheduler out of 11 active. The kernel is MEMORY-LATENCY-bound, not
 * bandwidth-bound, and ncu's own prescription is ILP, unrolling, pipelining.
 *
 * That is why the first two attempts failed. quant_matmul_i8w (#28) widened
 * the loads to char4/float4 -- the right move for bandwidth, the wrong one for
 * latency -- and lengthened the dependency chain with a per-block reduction
 * and double accumulation: 1.5-5 % SLOWER. The first cut of THIS kernel
 * blocked rows but kept `if (r < nr)` between the loads in the hot loop, which
 * is exactly the shape that lets the compiler serialise what should be
 * independent: flat, 178 -> 184 GB/s across R = 0, 2, 4, 8.
 *
 * So the hot path now issues R*U independent weight loads before consuming
 * any of them, with no runtime predicate among them, and the short last block
 * gets its own guarded loop. R*U is held at 16 across all three
 * instantiations (R=2/U=8, R=4/U=4, R=8/U=2) so the ILP budget is constant and
 * only its shape changes.
 *
 * ONE VARIABLE. The per-thread partition (thread t takes i = t, t+256, ...),
 * the byte-at-a-time weight read, the 256-wide shared reduction tree and the
 * trailing f32 scale are all EXACTLY the generic branch's. The only change is
 * that one block now carries R rows and reads each x element once for all of
 * them. The R trees run in lockstep under the same eight barriers, so the
 * barrier count per output FALLS by R as well, but that is a side effect of
 * the blocking rather than a second thing being changed.
 *
 * Consequence worth more than the speed: the result should be BIT-IDENTICAL
 * to the original kernel, because every addition happens in the same order on
 * the same operands. tests/test_int8_rows_cuda.cu asserts exactly that with
 * memcmp rather than a tolerance. An optimisation that cannot move a logit
 * cannot flip a token, which is what made #28's reordering need the oracle
 * before any default could change.
 *
 * Off by default all the same: unmeasured is unmeasured. COLI_CUDA_I8_ROWS=8
 * (or 2/4) turns it on. */
template<int R, int U>
__global__ static void quant_matmul_i8r(float *y, const float *x, const void *weights,
                                        const float *scales, int S, int I, int O) {
    /* The launch fixes 256; as a constant it lets the compiler fold the stride
     * arithmetic instead of re-deriving it from blockDim every iteration. */
    const int BS = 256;
    int o0 = (int)blockIdx.x * R, s = blockIdx.y;
    const float *xs = x + (size_t)s * I;
    const int8_t *w = static_cast<const int8_t *>(weights);
    int nr = O - o0 < R ? O - o0 : R;        /* the last block may be short */
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    int i = (int)threadIdx.x;
    if (nr == R) {
        /* HOT PATH. R*U independent weight loads are issued before any of them
         * is consumed -- the whole point, and the reason the R loop carries no
         * runtime predicate here: an `if (r < nr)` between the loads is exactly
         * what lets the compiler serialise them again. The short last block
         * takes the guarded loop below instead, once per call. */
        const int8_t *wr[R];
#pragma unroll
        for (int r = 0; r < R; r++) wr[r] = w + (size_t)(o0 + r) * I;
        for (; i + (U - 1) * BS < I; i += U * BS) {
            float xv[U], wv[R][U];
#pragma unroll
            for (int u = 0; u < U; u++) xv[u] = xs[i + u * BS];
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int u = 0; u < U; u++) wv[r][u] = (float)wr[r][i + u * BS];
            /* u outermost keeps the per-row summation order ascending in i --
             * i, i+BS, i+2BS, ... -- which is what the original kernel does and
             * what the bitwise-identity contract rests on. */
#pragma unroll
            for (int u = 0; u < U; u++)
#pragma unroll
                for (int r = 0; r < R; r++) acc[r] += xv[u] * wv[r][u];
        }
    }
    /* Tail of the strided range, and the whole of a short last block. */
    for (; i < I; i += BS) {
        float xv = xs[i];
#pragma unroll
        for (int r = 0; r < R; r++)
            if (r < nr) acc[r] += xv * static_cast<float>(w[(size_t)(o0 + r) * I + i]);
    }

    /* The generic branch's tree, R of them under the same barriers. */
    __shared__ float partial[R][256];
#pragma unroll
    for (int r = 0; r < R; r++) if (r < nr) partial[r][threadIdx.x] = acc[r];
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < (unsigned)n)
#pragma unroll
            for (int r = 0; r < R; r++)
                if (r < nr) partial[r][threadIdx.x] += partial[r][threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x < (unsigned)nr) {
        int o = o0 + (int)threadIdx.x;
        y[(size_t)s * O + o] = partial[threadIdx.x][0] * scales[o];
    }
    (void)S;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK/COLI_E8_SUB), e8_fp16(blk+96), w);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 7) {
        /* MXFP4: one ue8m0 exponent per 32 columns. The SCALAR CPU reference
         * (quant.h matmul_mxfp4's scalar loop -- its AVX2 sibling applies the
         * scale per 8-lane FMA instead and disagrees with it at s=255)
         * accumulates each 32-group UNSCALED and multiplies the group
         * subtotal by the scale once. For scales in [0,254] -- every value a
         * real checkpoint contains -- the scale is a finite power of two,
         * scaling each element or the subtotal is exact either way, so
         * threads stride over columns exactly as before and outputs stay
         * bit-identical with prior builds. s=255 decodes to +inf (the
         * documented edge), and there the application point is visible:
         * per-element scaling turns every zero product into 0*inf = NaN,
         * where the scalar reference's finite-subtotal-times-inf keeps the
         * group's sign. Rows carrying a 255 scale byte therefore take the
         * per-group path mirroring that scalar loop; the row's +-inf/NaN
         * classification does not depend on summation order unless several
         * large-finite (s~254) group results overflow only in aggregate --
         * out-of-domain either way -- so the tree reduce below needs no
         * change. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const uint8_t *scl = reinterpret_cast<const uint8_t *>(scales) + (size_t)o * ng;
        __shared__ int scale_inf;
        if (!threadIdx.x) scale_inf = 0;
        __syncthreads();
        for (int g = threadIdx.x; g < ng; g += blockDim.x)
            if (scl[g] == 255) scale_inf = 1;
        __syncthreads();
        if (!scale_inf) {
            for (int i = threadIdx.x; i < I; i += blockDim.x) {
                int g = i >> 5;
                if (g >= ng) g = ng - 1;
                sum += xs[i] * mx4_weight_at(wrow, i) * mx4_scale_dev(scl[g]);
            }
        } else {
            for (int g = threadIdx.x; g < ng; g += blockDim.x) {
                /* same element->group mapping as the clamp above: the last
                 * group takes every remaining column, a group past the data
                 * contributes nothing */
                int base = g << 5, end = base + 32;
                if (g == ng - 1 || end > I) end = I;
                if (base >= end) continue;
                float ga = 0.0f;
                for (int i = base; i < end; i++)
                    ga += xs[i] * mx4_weight_at(wrow, i);
                sum += ga * mx4_scale_dev(scl[g]);
            }
        }
    } else if (fmt == 4 || (fmt == 1 && gs > 0)) {
        /* Grouped int4, and grouped int8 (fmt 1 uploaded with a gs, the qwen38
         * dense trunk): one f32 scale per gs elements along I (ng groups per
         * row). Scale layout: scales[o*ng + g]. Each thread strides through I,
         * applying the appropriate group scale as it crosses group boundaries.
         * This matches the CPU matmul_i4_grouped accumulation exactly; for int8
         * it is the engine's Q38_TRUNK_CPU_INT8 reference loop. */
        const float *scl = scales + (size_t)o * ng;
        /* i / gs per element is an integer division on the GPU, about a
         * quarter of this loop's cost at gs 64; every group size in use is a
         * power of two, so shift instead when it is (the division stays for
         * an odd gs, same result either way). Measured on the qwen38 trunk,
         * 553 matrices per token: 75 -> 69 ms resident-mm (per-row: 63). */
        const int gshift = (gs & (gs - 1)) ? -1 : (__ffs(gs) - 1);
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            int g = gshift >= 0 ? (i >> gshift) : (i / gs);
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
        }
    } else if (fmt == 8) {
        /* fp8-e4m3 (matmul_fp8): one byte per weight (layout of fmt=1), one f32
         * scale per 128x128 BLOCK of [O,I] — scales[(o/128)*ceil(I/128) + i/128].
         * The block edge is a fixed property of the format (FP8_BLOCK), so the
         * geometry derives from I alone and gs/ng are ignored: this branch is
         * correct no matter which call site launched it. NaN bytes decode to NaN
         * through the LUT and propagate, same policy as the CPU path. This is
         * the ORIGINAL dense path, kept for COLI_CUDA_F8_WARP=0; the default
         * routes fmt=8 to quant_matmul_f8w instead (quant_matmul_launch). */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const float *scl = scales + (size_t)(o >> 7) * (size_t)((I + 127) >> 7);
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * c_e4m3[wrow[i]] * scl[i >> 7];
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * weight_at(weights, fmt, row, i);
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        /* fmt 4/6/7/8 already applied their scaling inside the loop: 4 and 7 are
         * per-group (one scale per gs / per 32 columns), 6 carries it in the
         * block header, 8 reads a per-128 block scale alongside the weights.
         * Only the per-row formats get the trailing multiply --
         * and for fmt=7 `scales` points at ue8m0 BYTES, so reading it as float
         * here does not merely double-scale, it reads garbage. */
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6 && fmt != 7 && fmt != 8 && !(fmt == 1 && gs > 0)) ? partial[0] * scales[o] : partial[0];
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
__global__ static void e8_rot_rows_kernel(float *rows, int dim, int off, int n){
    extern __shared__ float sh[];
    __shared__ uint8_t sbits[4096/8];
    if (!threadIdx.x) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    __syncthreads();
    float *row = rows + (size_t)blockIdx.x*dim + off;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    __syncthreads();
    for (int len=1;len<n;len<<=1){
        for (int j=threadIdx.x;j<n/2;j+=blockDim.x){
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        __syncthreads();
    }
    float sc=rsqrtf((float)n);
    for (int i=threadIdx.x;i<n;i+=blockDim.x) row[i]=sh[i]*sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim, cudaStream_t stream){
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        e8_rot_rows_kernel<<<(unsigned)nr, 256, (size_t)b*sizeof(float), stream>>>(rows, dim, off, b);
        if (cudaGetLastError() != cudaSuccess) return 0;
        off += b;
    }
    return 1;
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    __syncwarp();   /* i due fratelli sopra la portano; stessa lettura cross-lane di out[warp]
                     * subito sotto -> stesso requisito di visibilita' (racecheck: 3 hazard qui).
                     * EN: the two sibling kernels carry this; same cross-lane read follows. */
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.xoff+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

/* Native fmt=6 expert groups.  One block owns one (expert,row,output) and
 * expands each 32-weight E8 sub-block once for both gate/up projections. */
__global__ static void grouped_hidden_e8_dual(float *gate,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,D);
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*rb;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*rb;
    const float *xs=x+(size_t)(d.xoff+s)*D;float ga=0,ua=0;
    int nsub=(D+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *gb=gr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        const uint8_t *ub=ur+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float gw[COLI_E8_SUB],uw[COLI_E8_SUB];
        e8_expand_sub_dev(gb,ib,e8_fp16(gb+96),gw);
        e8_expand_sub_dev(ub,ib,e8_fp16(ub+96),uw);
        int off=sb*COLI_E8_SUB,n=D-off<COLI_E8_SUB?D-off:COLI_E8_SUB;
        for(int k=0;k<n;k++){float v=xs[off+k];ga+=v*gw[k];ua+=v*uw[k];}
    }
    __shared__ float gp[256],up[256];gp[threadIdx.x]=ga;up[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];up[threadIdx.x]+=up[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){float g=gp[0];gate[(size_t)(d.offset+s)*I+o]=(g/(1.f+expf(-g)))*up[0];}
}

__global__ static void grouped_down_e8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,I);const uint8_t *wr=(const uint8_t*)d.d+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    int nsub=(I+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *blk=wr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float w[COLI_E8_SUB];e8_expand_sub_dev(blk,ib,e8_fp16(blk+96),w);
        int off=sb*COLI_E8_SUB,n=I-off<COLI_E8_SUB?I-off:COLI_E8_SUB;
        for(int k=0;k<n;k++)sum+=xs[off+k]*w[k];
    }
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.xoff+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.xoff+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
__global__ static void grouped_hidden_g4_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.xoff+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_g4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* COLI_CUDA_DOWN_ROWS: R output rows per block for grouped_down_g4.
 *
 * The down projection is the worst-shaped kernel in the expert path. Its grid
 * is (D, rows, chunks) -- one block per output row -- and each block reads the
 * whole activation row. For qwen36 (D=2048, I=512, top-8) that is 16,384
 * blocks of 256 threads per launch, in which every thread reads exactly ONE
 * packed byte ((I+1)/2 = 256 bytes over 256 threads) and then joins a 256-wide
 * reduction with nine __syncthreads(). It also moves 2048 bytes of x per block
 * against 256 bytes of weights: x is 8x the weight traffic, served by L2
 * rather than DRAM but paid for in latency at every one of those blocks.
 *
 * R rows per block is the same change quant_matmul_i8r made to the dense GEMV:
 * x is read once per R rows, the grid drops by R, and each thread issues R
 * independent byte loads instead of one -- which is what the dense kernel's
 * Nsight profile said this family actually wants (memory-LATENCY-bound, not
 * bandwidth-bound). The per-thread partition of the input, the multiplication
 * order (x * nibble * scale), the reduction tree and the epilogue are all
 * untouched, so every output is BITWISE identical to grouped_down_g4's and
 * this cannot move a logit. tests/test_grouped_down_rows_cuda.cu asserts that
 * with memcmp.
 *
 * MEASURED, AND IT LOSES -- leave it off. On an RTX 4070 Ti SUPER the kernel
 * does exactly what the paragraph above predicts: expert-group kernel time
 * 558 -> 438 ms at R=4, and qtier take 0.53 -> 0.33 ms/token, two independent
 * measurements agreeing the group finishes earlier. The token still gets 5 %
 * SLOWER (24.05 -> 25.20 ms/token, 4/4 repetitions same sign).
 *
 * Two reasons, and the first is the one that matters for anyone else working
 * on this path. moe_total does not move: take was 0.53 ms of a 24 ms token,
 * so these kernels were ALREADY off the critical path, overlapped with CPU
 * work that finishes later, and 1.9 ms/token of freed GPU time had nowhere to
 * go. Second, 91 % of the regression lands on dn-sub proj, norm+out and
 * lm_head -- the placed dense GEMVs, which run on stream 0 alongside the
 * expert group: 4,096 long-lived blocks holding 4 KB of shared memory each
 * leave fewer scheduling slots than 16,384 that retire almost at once.
 *
 * It stays in the tree because it is bitwise identical (so it costs nothing
 * while unset), because it is the only way to reproduce that finding, and
 * because it stops being a bad idea the day the expert path does land on the
 * critical path -- a larger batch, or a configuration where cpu-miss is
 * fixed. docs/experiments/qwen36-expert-down-rows-2026-09-20-raw.txt */
template<int R>
__global__ static void grouped_down_g4r(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o0=(int)blockIdx.x*R, s=(int)blockIdx.y, c=(int)blockIdx.z;
    GroupDesc d=desc[c]; if(s>=d.rows)return;
    int nr = D-o0 < R ? D-o0 : R;
    int dgs = d.dgs>0?d.dgs:I;
    size_t ng = (size_t)((I+dgs-1)/dgs), rb=(size_t)((I+1)/2);
    const uint8_t *rw[R]; const float *dsc[R];
#pragma unroll
    for(int r=0;r<R;r++){
        /* the tail block clamps to o0 instead of predicating the hot loop:
         * those lanes compute a duplicate row nobody stores. */
        int o = o0 + (r<nr?r:0);
        rw[r]=(const uint8_t*)d.d+(size_t)o*rb;
        dsc[r]=d.ds+(size_t)o*ng;
    }
    const float *xs=x+(size_t)(d.offset+s)*I;
    float acc[R];
#pragma unroll
    for(int r=0;r<R;r++) acc[r]=0.0f;
    for(int b=threadIdx.x;b<(int)rb;b+=blockDim.x){
        int i=b*2;
        float a[R],z[R];
#pragma unroll
        for(int r=0;r<R;r++) unpack_s4(rw[r][b],&a[r],&z[r]);
        float x0=xs[i], x1=(i+1<I)?xs[i+1]:0.0f;
        /* one integer division per iteration, not R: dgs is a runtime value,
         * so i/dgs is a real divide and the unrolled body below would be
         * relying on CSE to notice it is loop-invariant. */
        int gi=i/dgs;
#pragma unroll
        for(int r=0;r<R;r++){
            float sv=dsc[r][gi];
            acc[r]+=x0*a[r]*sv;
            if(i+1<I)acc[r]+=x1*z[r]*sv;
        }
    }
    __shared__ float p[R][256];
#pragma unroll
    for(int r=0;r<R;r++) p[r][threadIdx.x]=acc[r];
    __syncthreads();
    for(int n=128;n;n>>=1){
        if(threadIdx.x<(unsigned)n){
#pragma unroll
            for(int r=0;r<R;r++) p[r][threadIdx.x]+=p[r][threadIdx.x+n];
        }
        __syncthreads();
    }
    if(threadIdx.x<(unsigned)nr){
        int r=(int)threadIdx.x;
        y[(size_t)(d.offset+s)*D+o0+r]=p[r][0];
    }
}

/* ON BY DEFAULT AT R=4, AND ONLY BECAUSE THE GRAPH IS ON TOO.
 *
 * Measured alone, this kernel COSTS 1.15 ms/token (+4 %): it is faster on the
 * GPU -- expert-group kernel time 697 -> 572 ms, -18 % -- and the token gets
 * worse, because `take` was already near zero and the freed GPU time had
 * nowhere to go while 4,096 long-lived blocks squeezed the placed dense GEMVs
 * sharing the SMs.
 *
 * COLI_CUDA_GRAPH changes that. The graph moves ~0.7 ms out of `issue` and
 * into `take`, which is exactly the headroom this kernel needs to convert. A
 * 2x2 over {GRAPH 0,1} x {DOWN_ROWS 0,4} measured the interaction directly:
 *
 *     effect of DOWN_ROWS with GRAPH=0:   +1.15 ms/token
 *     effect of DOWN_ROWS with GRAPH=1:   -0.25 ms/token
 *     INTERACTION:                        -1.40 ms/token
 *     both on vs both off (the diagonal): -0.43, 6/6 repetitions negative
 *
 * So the default below is a PAIR, not two independent choices, and it is
 * compiled off wherever the graph cannot exist: on HIP there is no
 * cudaStreamBeginCapture, so COLI_GPU_HAS_GRAPH is 0, so the arm that makes
 * this kernel pay is not available and the measured-bad half would ship
 * alone. Turning it on there needs its own measurement on that hardware.
 *
 * RTX 4070 Ti SUPER, qwen36 i4 gs64, clang, 4 cells x 6 alternated reps,
 * identical generation in every cell:
 * docs/experiments/qwen36-expert-down-rows-2026-09-20-raw.txt and the 2x2
 * record alongside it. */
/* Defined further down, next to the graph capture it controls. Declared here
 * because the default below is justified by it and down_g4_launch warns when
 * the two disagree -- the same declaration-order trap CI caught the first
 * time this file grew a cross-reference. */
static int graph_mode(void);

#ifndef COLI_DOWN_ROWS_DEFAULT
#  if COLI_GPU_HAS_GRAPH
#    define COLI_DOWN_ROWS_DEFAULT 4
#  else
#    define COLI_DOWN_ROWS_DEFAULT 0
#  endif
#endif

/* Unset reads as the compiled default; an explicit 0 is how "off" is asked
 * for now that the default is on; 2, 4 and 8 are the instantiated widths;
 * anything else falls back to the default rather than rounding to a
 * neighbour, so a typo in a measurement script gives the shipped arm rather
 * than a third arm nobody meant to run. Same shape as i8_rows_mode(). */
static int down_rows_mode(void){
    const char *e=getenv("COLI_CUDA_DOWN_ROWS");
    if(!e||!*e) return COLI_DOWN_ROWS_DEFAULT;
    char *end; long v=strtol(e,&end,10);
    if(*end) return COLI_DOWN_ROWS_DEFAULT;
    if(v==0) return 0;
    return (v==2||v==4||v==8)?(int)v:COLI_DOWN_ROWS_DEFAULT;
}

/* Launch counter, read by tests/test_grouped_down_rows_cuda.cu: the
 * bitwise-identity contract is ALSO satisfied by a dispatch that never fires,
 * so the test needs a way to tell "identical" from "never ran". */
static uint64_t g_downr_launches;

static void down_g4_launch(float *y,const float *x,const GroupDesc *dev,
                           int D,int I,int max_rows,int count,cudaStream_t st){
    int R=down_rows_mode();
    if(R){
        /* Say so once, on stderr, the first time the branch actually fires.
         *
         * Under CUDA_DLL=1 every kernel here lives in coli_cuda.dll, which only
         * `make cuda-dll` rebuilds: a measurement script that rebuilds the
         * engine and not the DLL sets COLI_CUDA_DOWN_ROWS, changes nothing, and
         * reports a clean flat result. That cost a day on the dense GEMV. A
         * line in the log is the cheapest way for a run to prove the toggle
         * reached the binary it is running, rather than the one on disk when
         * the script started. Silent when the variable is unset, so the default
         * path's output is unchanged. */
        static int announced;
        if(!announced){ announced=1;
            /* Only when the variable was SET. The announcement exists to prove
             * a measurement's toggle reached the binary it is running; now
             * that the default is on, printing it unset would put a line on
             * every ordinary run for no one's benefit. The shipped default is
             * asserted by the dispatch probe in
             * tests/test_grouped_down_rows_cuda.cu, which is the right place
             * for it -- a test, not a log line. */
            const char *e=getenv("COLI_CUDA_DOWN_ROWS");
            if(e&&*e) fprintf(stderr,"[cuda] expert down-rows active: R=%d\n",R);
            /* The one combination measured WORSE than shipping neither. This
             * kernel's default is justified only by the headroom the graph
             * opens in `take`; without it the same kernel cost +1.15 ms/token
             * on the box where both were measured. Not silently corrected --
             * a flag that means one thing is worth more than a flag that
             * quietly rewrites another -- but not silent either. */
            if(!graph_mode())
                fprintf(stderr,"[cuda] WARNING: expert down-rows is on with "
                               "COLI_CUDA_GRAPH off. Measured +1.15 ms/token "
                               "in that combination; set COLI_CUDA_DOWN_ROWS=0 "
                               "too, or leave the graph on.\n");
        }
        g_downr_launches++;
        dim3 og((unsigned)((D+R-1)/R),(unsigned)max_rows,(unsigned)count);
        if(R==2)      grouped_down_g4r<2><<<og,256,0,st>>>(y,x,dev,D,I);
        else if(R==4) grouped_down_g4r<4><<<og,256,0,st>>>(y,x,dev,D,I);
        else          grouped_down_g4r<8><<<og,256,0,st>>>(y,x,dev,D,I);
        return;
    }
    dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
    grouped_down_g4<<<og,256,0,st>>>(y,x,dev,D,I);
}

/* fmt=8 fp8-e4m3 variants: same structure as the g4 kernels, but one byte per
 * weight (decoded through c_e4m3) and the scale is per 128x128 BLOCK of the
 * member's [O,I] matrix — sc[(o/128)*ceil(I/128) + i/128]. The block edge is a
 * property of the format, so the geometry derives from the dims alone; these
 * kernels require every member to be fmt=8 (no ride-along: a per-row member's
 * scales are [O], which this indexing would read out of bounds). */
__global__ static void grouped_hidden_f8_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.xoff+s)*D;float ga=0,ua=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){float xv=xs[i];int b=i>>7;
        ga+=xv*c_e4m3[gr[i]]*gsc[b];ua+=xv*c_e4m3[ur[i]]*usc[b];}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same fused epilogue as the w4/g4 duals: scales applied in the
     * accumulation, silu(gate)*up lands in gate[], up[] is never written */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_f8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=xs[i]*c_e4m3[row[i]]*dsc[i>>7];
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* fmt=8 warp rework (COLI_CUDA_F8_WARP, default on): one WARP owns one
 * (expert,output-row) pair and walks the row block-major through f8w_block —
 * weights stream once per tile of up to 4 activation rows instead of once per
 * (o,s) 256-thread block, and the accumulation is the CPU reference's (f32
 * block partial, scale once per block, double across blocks) instead of the
 * old flat per-element-scale f32 sum. Same descriptor surface as the old
 * kernels; grid (ceil(O/8), count): 8 warps per 256-thread block, rows looped
 * in-kernel. The old kernels stay compiled and selectable (=0) as the field
 * escape hatch. */
template<int HW>
__global__ static void grouped_hidden_f8w_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=I)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    int vec=!(D&3)&&!(((size_t)gr|(size_t)ur)&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        /* xoff, not offset: this reads the ACTIVATION, which the broadcast
         * contract may hand over as a single row shared by every chunk.
         * offset addresses gate/up/y, which stay per-expert. The mode-0
         * sibling grouped_hidden_f8_dual was converted when xoff was added
         * and this one was not -- and this is the CUDA default
         * (COLI_F8_DEFAULT), so the arm that got the fix is the arm nobody
         * runs. See tests/test_group_bcast_arms_cuda.cu. */
        const float *xs=x+(size_t)(d.xoff+s0)*D;
        double ga[4]={0,0,0,0},ua[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=D-base<128?D-base:128;
            float gp[4],up_[4];
            f8w_block<HW,1>(slut,gr,ur,xs,D,ns,base,len,vec,gp,up_);
            if(!lane){float sg=gsc[bi],su=usc[bi];
                for(int s=0;s<ns;s++){ga[s]+=(double)gp[s]*sg;ua[s]+=(double)up_[s]*su;}}
        }
        /* same fused epilogue as the old dual: silu(gate)*up lands in gate[],
         * up[] is never written */
        if(!lane)for(int s=0;s<ns;s++){
            float g=(float)ga[s],u=(float)ua[s];
            gate[(size_t)(d.offset+s0+s)*I+o]=(g/(1.f+expf(-g)))*u;
        }
    }
    (void)up;
}
template<int HW>
__global__ static void grouped_down_f8w(float *y,const float *x,const GroupDesc *desc,int D,int I){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=D)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    int vec=!(I&3)&&!((size_t)row&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*I;
        double a[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=I-base<128?I-base:128;
            float p[4];
            f8w_block<HW,0>(slut,row,nullptr,xs,I,ns,base,len,vec,p,nullptr);
            if(!lane){float sd=dsc[bi];
                for(int s=0;s<ns;s++)a[s]+=(double)p[s]*sd;}
        }
        if(!lane)for(int s=0;s<ns;s++)
            y[(size_t)(d.offset+s0+s)*D+o]=(float)a[s];
    }
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale,
                                                int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,
        int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* every warp must read red[0] above before red[] is reused below */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *const *latent,const float *const *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,int page_stride,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;int pg=t/COLI_KV_PAGE_TOKENS,pt=t%COLI_KV_PAGE_TOKENS;
        const float *lt=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        const float *rt=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* every warp must read red[0] above before red[] is reused below */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++){
        const float *lt=latent[(size_t)s*page_stride+t/COLI_KV_PAGE_TOKENS]+
                        (size_t)(t%COLI_KV_PAGE_TOKENS)*K;
        a+=scores[t]*lt[k];}cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*
            absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

__global__ static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,
        int K,int R,int page_stride){
    int s=blockIdx.x,n=add[s],base=offset[s];
    for(int t=0;t<n;t++){
        int pos=old_len[s]+t,pg=pos/COLI_KV_PAGE_TOKENS,pt=pos%COLI_KV_PAGE_TOKENS;
        float *lp=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        float *rp=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;
        for(int k=threadIdx.x;k<K;k+=blockDim.x)lp[k]=packed[base+(size_t)t*K+k];
        for(int r=threadIdx.x;r<R;r+=blockDim.x)rp[r]=packed[base+(size_t)n*K+(size_t)t*R+r];
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

/* Same as reserve_pinned, for a buffer that is not floats. The descriptor
 * staging needs to be pinned for two reasons at once: cudaMemcpyAsync from
 * pageable memory is not capture-safe, and a graph records the SOURCE
 * ADDRESS, so it has to be one that outlives the call. */
static int reserve_pinned_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

/* COLI_CUDA_GRAPH: replay the expert group as one cudaGraphLaunch instead of
 * five driver calls. ON by default, and `=0` turns it off.
 *
 * Measured alone it is worth nothing on the token: `issue` falls 28 %
 * (2.47 -> 1.79 ms/token) and every millisecond of it reappears in `take`
 * (0.76 -> 1.50), because the MoE phase is GPU-bound with idle CPU inside it
 * and a CPU-side saving there just converts into waiting.
 *
 * It ships because of what that widened `take` lets the expert down-rows
 * kernel do -- the interaction is measured, and the argument is written out
 * over down_rows_mode(). The two are a pair: turning THIS off while leaving
 * COLI_CUDA_DOWN_ROWS on is the one combination known to be worse than
 * shipping neither, and down_g4_launch says so on stderr if it sees it.
 *
 * Capture is not assumed to succeed. If cudaStreamBeginCapture or
 * cudaGraphInstantiate fails, the call refuses (qt_issue hands that one layer
 * to the CPU, slower and correct), says so once on stderr, and records the
 * signature so it does not retry every call. */
static int graph_mode(void){
#if COLI_GPU_HAS_GRAPH
    /* Deliberately NOT cached in a static, unlike gprof a few lines down.
     * The cost argument that applies there -- 40 calls a token is no place
     * for a getenv -- is real but small next to the ~10 us driver call this
     * function decides whether to make, and it falls out of a paired A/B
     * because both arms pay it. What caching would cost is the only test
     * this path has: tests/test_grouped_g4_cuda.cu flips the variable
     * mid-process to compare the graph against per-call launches, and a
     * latched static would make that test measure one arm twice. */
    const char *e=getenv("COLI_CUDA_GRAPH");
    /* On unless explicitly disabled. Unset is the shipped arm now: see the
     * interaction measurement above down_rows_mode(). */
    return !e||!*e||*e!='0';
#else
    return 0;
#endif
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

#ifdef COLI_ANS
static void *ans_arena_alloc(DeviceContext *ctx,size_t bytes){
    bytes=(bytes+255)&~size_t(255);
    if(!ctx->ans_chunks)ctx->ans_chunks=new std::vector<AnsArenaChunk>;
    if(ctx->ans_chunks->empty()||ctx->ans_chunks->back().cap-ctx->ans_chunks->back().used<bytes){
        size_t cap=256ull<<20;if(cap<bytes)cap=bytes;
        uint8_t *p=nullptr;if(!cuda_ok(cudaMalloc(&p,cap),"ANS arena chunk"))return nullptr;
        ctx->ans_chunks->push_back({p,0,cap});
    }
    AnsArenaChunk &c=ctx->ans_chunks->back();void *p=c.p+c.used;c.used+=bytes;return p;
}
static int ans_host_reserve(DeviceContext *ctx,size_t bytes){
    if(ctx->ans_copy_pending){
        if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS sidecar upload synchronize"))return 0;
        ctx->ans_copy_pending=0;
    }
    if(ctx->ans_host_cap>=bytes)return 1;
    if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
    ctx->ans_host=nullptr;ctx->ans_host_cap=0;
    if(!cuda_ok(cudaMallocHost(&ctx->ans_host,bytes),"ANS pinned staging allocation"))return 0;
    ctx->ans_host_cap=bytes;return 1;
}
static int prepare_group_weights(DeviceContext *ctx,
        ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs,int count,GroupDesc *host){
    int n=0; size_t total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        for(int k=0;k<3;k++) if(q[k]->compressed){n++;total+=q[k]->weight_bytes;}
    }
    if(!n) return 1;
    if(!g_ans_profile_printed&&std::getenv("COLI_ANS_PROFILE")){
        g_ans_profile_printed=1;
        std::fprintf(stderr,
            "[ANS] load profile: %llu records | header %.2fs | read %.2fs | "
            "staging/alloc %.2fs | enqueue %.2fs\n",
            (unsigned long long)g_ans_load_records,g_ans_header_s,g_ans_read_s,
            g_ans_stage_s,g_ans_enqueue_s);
    }
    if(!ctx->ans_scratch||!reserve_bytes(&ctx->ans_raw,&ctx->ans_raw_cap,total)) return 0;
    std::vector<const void*> in; in.reserve(n);
    std::vector<void*> out; out.reserve(n);
    std::vector<uint32_t> cap; cap.reserve(n);
    size_t off=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        const void **dst[3]={&host[c].g,&host[c].u,&host[c].d};
        for(int k=0;k<3;k++) if(q[k]->compressed){
            void *raw=(uint8_t*)ctx->ans_raw+off;
            in.push_back(q[k]->weights);out.push_back(raw);cap.push_back((uint32_t)q[k]->weight_bytes);
            *dst[k]=raw;off+=q[k]->weight_bytes;
        }
    }
    dietgpu::ANSCodecConfig config(11,false);
    dietgpu::ansDecodeBatchPointer(*ctx->ans_scratch,config,(uint32_t)n,in.data(),out.data(),
                                   cap.data(),nullptr,nullptr,ctx->stream);
    return cuda_ok(cudaGetLastError(),"ANS expert decode launch");
}
#else
static int prepare_group_weights(DeviceContext *,ColiCudaTensor *const *,
        ColiCudaTensor *const *,ColiCudaTensor *const *,int,GroupDesc *){return 1;}
#endif

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e8_grid, grid, sizeof(c_e8_grid)), "E8 codebook upload"))
            return 0;
    }
    return 1;
}

/* Publish quant.h's E4M3_LUT the same way — one source of truth for the fmt=8
 * decode on CPU and GPU. Until this succeeds, fmt=8 uploads are refused. */
extern "C" int coli_cuda_fp8_set_lut(const float *lut) {
    if (!lut || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e4m3, lut, sizeof(c_e4m3)), "e4m3 LUT upload"))
            return 0;
    }
    g_fp8_lut_ready = 1;
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* #509: the ROCm runtime (comgr, MIOpen, roctracer) reads $TEMP as a temp-dir
     * path. A stray numeric TEMP (the engine's legacy sampling alias) makes comgr's
     * lazy init fail inside the first stream create -- SIGSEGV in the error-unwind
     * on gfx1100, clean hipErrorOutOfMemory on gfx1030. The engine has already
     * parsed g_temp by the time we get here, so a TEMP that is not a real directory
     * is safe to drop before the first ROCm call; a genuine temp-dir is preserved. */
    {
        const char *t = std::getenv("TEMP");
        struct stat st;
        /* Same test on both hosts; only the CRT spelling differs. The MSVC CRT
         * (Windows hipcc's host pass) has no S_ISDIR and no unsetenv — it spells
         * the directory bit _S_IFDIR/_S_IFMT and clears a variable by assigning
         * an empty value. This is an OS/CRT difference, NOT a vendor one, so it
         * stays a _WIN32 branch and adds no CUDA-vs-HIP conditional. */
#ifdef _WIN32
        if (t && *t && (stat(t, &st) != 0 ||
                        (st.st_mode & _S_IFMT) != _S_IFDIR)) _putenv_s("TEMP", "");
#else
        if (t && *t && (stat(t, &st) != 0 || !S_ISDIR(st.st_mode))) unsetenv("TEMP");
#endif
    }
#endif
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
#ifdef COLI_ANS
        if(std::getenv("CUDA_RAW_EXPERTS")){
            ctx->ans_scratch = new dietgpu::StackDeviceMemory(device, 1ull << 30);
            ctx->ans_chunks = new std::vector<AnsArenaChunk>;
        }
#endif
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" int coli_cuda_available_device_count(void) {
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess) return 0;
    return available;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->dx) cudaFree(ctx->dx);
        if (ctx->dy) cudaFree(ctx->dy);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        for(int b=0;b<27;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_dx) cudaFreeHost(ctx->host_dx);
        if (ctx->host_dy) cudaFreeHost(ctx->host_dy);
        if (ctx->host_kv) cudaFreeHost(ctx->host_kv);
        if (ctx->host_desc) cudaFreeHost(ctx->host_desc);
        ctx->host_desc=nullptr; ctx->host_desc_cap=0;
#if COLI_GPU_HAS_GRAPH
        /* Before the stream goes: an instantiated graph belongs to it. */
        for(int gi=0;gi<9;gi++) if(ctx->graph_exec[gi]){
            cudaGraphExecDestroy(ctx->graph_exec[gi]);
            ctx->graph_exec[gi]=nullptr; ctx->graph_gen[gi]=0; }
#endif
        if (ctx->ev_grp_ok>0) { for(int e=0;e<4;e++) cudaEventDestroy(ctx->ev_grp[e]); }
        ctx->ev_grp_ok=0;
        if (ctx->ev_dm_ok>0) { for(int e=0;e<4;e++) cudaEventDestroy(ctx->ev_dm[e]); }
        ctx->ev_dm_ok=0;
        ctx->group_timed=0;
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
#ifdef COLI_ANS
        if(ctx->ans_copy_pending)cudaStreamSynchronize(ctx->stream);
        if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
        if (ctx->ans_raw) cudaFree(ctx->ans_raw);
        if(ctx->ans_chunks){for(auto &c:*ctx->ans_chunks)cudaFree(c.p);delete ctx->ans_chunks;}
        delete ctx->ans_scratch;
        ctx->ans_scratch=nullptr;ctx->ans_chunks=nullptr;ctx->ans_raw=nullptr;ctx->ans_raw_cap=0;
        ctx->ans_host=nullptr;ctx->ans_host_cap=0;ctx->ans_copy_pending=0;
#endif
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->dx = ctx->dy = nullptr; ctx->dx_cap = ctx->dy_cap = 0;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=ctx->host_kv=nullptr;ctx->stream=nullptr;
        ctx->host_dx=ctx->host_dy=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
        ctx->host_dx_cap=ctx->host_dy_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
#ifdef COLI_ANS
    if(g_ans_sidecar){std::fclose(g_ans_sidecar);g_ans_sidecar=nullptr;}
#if defined(__linux__)
    if(g_ans_direct_fd>=0){close(g_ans_direct_fd);g_ans_direct_fd=-1;g_ans_direct_off=0;}
#endif
#endif
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    return prop.integrated ? 1 : 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    int index=-1;
    for(int i=0;i<g_nctx;i++) if(g_ctx[i].device==device){ index=i; break; }
    if(calls) *calls=index<0?0:g_device_group_calls[index];
    if(experts) *experts=index<0?0:g_device_group_experts[index];
    if(rows) *rows=index<0?0:g_device_group_rows[index];
    if(h2d_ms) *h2d_ms=index<0?0:g_device_group_h2d_ms[index];
    if(kernel_ms) *kernel_ms=index<0?0:g_device_group_kernel_ms[index];
    if(d2h_ms) *d2h_ms=index<0?0:g_device_group_d2h_ms[index];
}

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = ((fmt==4 || fmt==1) && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && !scales)) return 0;
    if (fmt == 8 && !g_fp8_lut_ready) return 0;   /* kernels would read a zero LUT */
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = ((fmt==4 || fmt==1) && g_upload_gs>0) ? g_upload_gs : 0;   /* fmt 1 + gs: grouped int8 (dense trunk) */
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    if (fmt == 8) {   /* per-128x128-block scales: [ceil(O/128), ceil(I/128)] */
        t->ng = (I + 127) / 128;
        t->scale_count = (size_t)((O + 127) / 128) * (size_t)t->ng;
    }
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    /* Ownership is a fact of the allocation, not the copy: set it BEFORE the
     * memcpy, or a failed H2D upload frees the tensor while weights_owned is
     * still 0 and free()'s ownership gate leaks the device buffer. */
    t->weights_owned=1;
    if (!cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt && fmt != 6) {
        if (!cuda_ok(cudaMalloc(&t->scales, t->scale_count * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, t->scale_count * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    if (fmt == 6) t->scale_count = 0;      /* in-block scales: nothing separate to track */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + ((fmt && fmt != 6) ? t->scale_count * sizeof(float) : 0);
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

#ifdef COLI_ANS
struct AnsSidecarHeader {
    uint32_t magic,raw_bytes,archive_bytes,fmt,I,O;
};
static FILE *ans_sidecar(void){
    if(g_ans_sidecar) return g_ans_sidecar;
    const char *path=std::getenv("COLI_ANS_SIDECAR");
    if(!path||!*path) return nullptr;
    g_ans_sidecar_pack=std::getenv("COLI_ANS_PACK")&&std::atoi(std::getenv("COLI_ANS_PACK"));
    g_ans_sidecar=std::fopen(path,g_ans_sidecar_pack?"wb":"rb");
#if defined(__linux__)
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&std::getenv("COLI_ANS_DIRECT")&&
       std::atoi(std::getenv("COLI_ANS_DIRECT"))){
        g_ans_direct_fd=open(path,O_RDONLY|O_DIRECT);
        if(g_ans_direct_fd<0)std::fprintf(stderr,"[ANS] O_DIRECT unavailable; using buffered sidecar\n");
    }
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&g_ans_direct_fd<0)
        posix_fadvise(fileno(g_ans_sidecar),0,0,POSIX_FADV_SEQUENTIAL);
#endif
    return g_ans_sidecar;
}
extern "C" int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
        const void *weights,const float *scales,int fmt,int I,int O,int device){
    if(fmt!=2 || !tensor || *tensor) return 0;  /* prototype: per-row int4 experts only */
    FILE *sidecar=ans_sidecar();
    if(sidecar&&!g_ans_sidecar_pack){
        AnsSidecarHeader h{};
        size_t expected=((size_t)I+1)/2*(size_t)O;
        double t0=ans_now_s();
#if defined(__linux__)
        size_t direct_delta=0,direct_bytes=0,direct_data=0;
        if(g_ans_direct_fd>=0){
            alignas(4096) uint8_t first[8192];
            off_t start=g_ans_direct_off&~off_t(4095);
            direct_delta=(size_t)(g_ans_direct_off-start);
            ssize_t got=pread(g_ans_direct_fd,first,sizeof(first),start);
            if(got<(ssize_t)(direct_delta+sizeof(h))){
                std::fprintf(stderr,"[ANS] direct header read failed at %lld: got %lld errno %d\n",
                    (long long)g_ans_direct_off,(long long)got,errno);
                return 0;
            }
            std::memcpy(&h,first+direct_delta,sizeof(h));
        }else
#endif
        if(std::fread(&h,sizeof(h),1,sidecar)!=1)return 0;
        if(h.magic!=0x31534e41u||
           h.fmt!=(uint32_t)fmt||h.I!=(uint32_t)I||h.O!=(uint32_t)O||
           expected>UINT32_MAX||h.raw_bytes!=(uint32_t)expected||
           !h.archive_bytes||h.archive_bytes>dietgpu::getMaxCompressedSize(h.raw_bytes)){
            std::fprintf(stderr,"[ANS] invalid or mismatched sidecar record\n");
            return 0;
        }
        g_ans_header_s+=ans_now_s()-t0;
        DeviceContext *ctx=find_ctx(device); if(!ctx||!select_ctx(ctx)) return 0;
        ColiCudaTensor *t=(ColiCudaTensor*)std::calloc(1,sizeof(*t)); if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=h.raw_bytes;
        t->scale_count=(size_t)O;t->archive_bytes=h.archive_bytes;t->compressed=1;
        t0=ans_now_s();
        t->weights=ans_arena_alloc(ctx,h.archive_bytes);
        size_t scale_bytes=(size_t)O*sizeof(float),scale_off;
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            size_t record_bytes=sizeof(h)+(size_t)h.archive_bytes;
            direct_data=direct_delta+sizeof(h);
            direct_bytes=(direct_delta+record_bytes+4095)&~size_t(4095);
            scale_off=(direct_bytes+255)&~size_t(255);
        }else
#endif
            scale_off=(h.archive_bytes+255)&~size_t(255);
        if(!t->weights||!ans_host_reserve(ctx,scale_off+scale_bytes)||
           !cuda_ok(cudaMalloc(&t->scales,scale_bytes),"ANS sidecar scales")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_stage_s+=ans_now_s()-t0;
        t0=ans_now_s();
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            off_t start=g_ans_direct_off&~off_t(4095);
            ssize_t got=pread(g_ans_direct_fd,ctx->ans_host,direct_bytes,start);
            if(got<(ssize_t)(direct_data+h.archive_bytes)){
                std::fprintf(stderr,
                    "[ANS] direct record read failed at %lld: need %zu got %lld errno %d\n",
                    (long long)g_ans_direct_off,direct_data+h.archive_bytes,
                    (long long)got,errno);
                coli_cuda_tensor_free(t);return 0;
            }
            g_ans_direct_off+=(off_t)sizeof(h)+(off_t)h.archive_bytes;
        }else
#endif
        if(std::fread(ctx->ans_host,h.archive_bytes,1,sidecar)!=1){
            std::fprintf(stderr,"[ANS] truncated sidecar record\n");
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_read_s+=ans_now_s()-t0;
        std::memcpy((uint8_t*)ctx->ans_host+scale_off,scales,scale_bytes);
        t0=ans_now_s();
        void *archive_src=
#if defined(__linux__)
            g_ans_direct_fd>=0?(uint8_t*)ctx->ans_host+direct_data:
#endif
            ctx->ans_host;
        if(!cuda_ok(cudaMemcpyAsync(t->weights,archive_src,h.archive_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar upload")||
           !cuda_ok(cudaMemcpyAsync(t->scales,(uint8_t*)ctx->ans_host+scale_off,scale_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar scale upload")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_enqueue_s+=ans_now_s()-t0;g_ans_load_records++;
        ctx->ans_copy_pending=1;
        t->tracked=1;ctx->tensor_count++;ctx->tensor_bytes+=h.archive_bytes+(size_t)O*sizeof(float);
        *tensor=t;return 1;
    }
    if(!sidecar||!g_ans_sidecar_pack) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor *t=*tensor;
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->ans_scratch||!select_ctx(ctx)){ coli_cuda_tensor_free(t);*tensor=nullptr;return 0; }
    uint32_t raw=(uint32_t)t->weight_bytes;
    uint32_t bound=dietgpu::getMaxCompressedSize(raw), *dsize=nullptr;
    void *tmp=nullptr;
    if(!cuda_ok(cudaMalloc(&tmp,bound),"ANS archive allocation")||
       !cuda_ok(cudaMalloc(&dsize,sizeof(*dsize)),"ANS size allocation")){
        if(tmp)cudaFree(tmp);if(dsize)cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ANSCodecConfig config(11,false);
    /* tensor_upload converted offset-binary nibbles on the legacy stream.
     * ctx->stream is explicitly non-blocking, so it does not inherit the
     * legacy-stream dependency. Finish that one-time conversion before the
     * encoder reads the bytes. */
    if(!cuda_ok(cudaStreamSynchronize(0),"ANS source conversion synchronize")){
        cudaFree(tmp);cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ansEncodeBatchStride(*ctx->ans_scratch,config,1,t->weights,raw,raw,nullptr,
                                  tmp,bound,dsize,ctx->stream);
    uint32_t used=0;
    int ok=cuda_ok(cudaMemcpyAsync(&used,dsize,sizeof(used),cudaMemcpyDeviceToHost,ctx->stream),
                   "ANS size download")&&
           cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS encode synchronize")&&used>0&&used<raw;
    cudaFree(dsize);
    if(!ok){cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;}
    if(g_ans_sidecar_pack){
        std::vector<uint8_t> archive(used);
        AnsSidecarHeader h{0x31534e41u,raw,used,(uint32_t)fmt,(uint32_t)I,(uint32_t)O};
        ok=cuda_ok(cudaMemcpy(archive.data(),tmp,used,cudaMemcpyDeviceToHost),"ANS sidecar download")&&
           std::fwrite(&h,sizeof(h),1,sidecar)==1&&
           std::fwrite(archive.data(),archive.size(),1,sidecar)==1;
        cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;
        if(!ok)return 0;
        t=(ColiCudaTensor*)std::calloc(1,sizeof(*t));if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=raw;
        t->archive_bytes=used;t->compressed=1;*tensor=t;
        return 1;
    }
    return 0;
}
#endif

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
#ifdef COLI_ANS
    if(tensor->compressed) return 0;
#endif
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    /* fmt=6 has no scale buffer at all (scales live in-block, scale_count 0), and
     * the fallback below would otherwise copy O floats out of a NULL host pointer. */
    return !tensor->fmt || tensor->fmt==6 || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (tensor->scale_count?tensor->scale_count:(size_t)tensor->O)*sizeof(float),
        cudaMemcpyHostToDevice),"scale refresh");
}

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    return fa && g_gpu_calls++ >= std::atol(fa);
}

/* COLI_CUDA_F8_WARP mode, re-read per dispatch like its siblings. Strict
 * parse: a non-numeric value selects the DEFAULT, not atoi's silent 0.
 * Default 1 (warp kernels) on CUDA; 0 (original kernels) on HIP — the warp
 * kernels' wave64 width-32 shuffle sub-grouping has never been validated on
 * AMD silicon, and a scoring instrument must not default onto an untested
 * reduction. Opt in explicitly with =1/=2 once hip-test certifies it. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define COLI_F8_DEFAULT 0
#else
#define COLI_F8_DEFAULT 1
#endif
static int f8_warp_mode(void) {
    const char *e = std::getenv("COLI_CUDA_F8_WARP");
    if (!e || !*e) return COLI_F8_DEFAULT;
    char *end; long v = std::strtol(e, &end, 10);
    return *end ? COLI_F8_DEFAULT : (int)v;
}

/* COLI_CUDA_I8_ROWS: rows per block for the fmt=1 per-row GEMV. 0 keeps the
 * original one-row-per-block kernel; 2, 4 and 8 are the instantiated widths.
 * Unset, or set to a width this build has no instantiation for, reads as the
 * default rather than silently rounding to a neighbouring width.
 *
 * Default 2 is measured on one GPU only (RTX 4070 Ti SUPER, sm_89, qwen36
 * i4 gs64): the R>=2 kernels run the placed dense GEMVs at 342-381 GB/s
 * against the original's 187-191. R=2 and R=4 landed inside each other's
 * spread across two sessions, R=8 was the slowest of the three in both, so R=2 is the pick
 * for being no worse on the measurements and the cheapest in registers and
 * shared memory -- the side to err on for an architecture nobody has
 * measured. COLI_CUDA_I8_ROWS=0 restores the original kernel; the outputs
 * are bitwise identical either way (tests/test_int8_rows_cuda.cu). */
#ifndef COLI_I8_ROWS_DEFAULT
#define COLI_I8_ROWS_DEFAULT 2
#endif
/* Launch counter, read by tests/test_int8_rows_cuda.cu.
 *
 * The bitwise-identity contract quant_matmul_i8r is built around is ALSO
 * satisfied by a dispatch that never fires: the same kernel running twice is
 * trivially identical, so a test that only compares outputs passes whether or
 * not the branch was ever taken. This counter is how the test tells those two
 * worlds apart. One increment on a path that is already launching a kernel. */
static uint64_t g_i8r_launches;

static int i8_rows_mode(void) {
    const char *e = std::getenv("COLI_CUDA_I8_ROWS");
    if (!e || !*e) return COLI_I8_ROWS_DEFAULT;
    char *end; long v = std::strtol(e, &end, 10);
    if (*end) return COLI_I8_ROWS_DEFAULT;
    if (v == 0) return 0;
    return (v == 2 || v == 4 || v == 8) ? (int)v : COLI_I8_ROWS_DEFAULT;
}

/* One launch site for the dense matvec so fmt=8 honors the same toggle as
 * f8_group_launch: mode 0 runs the original quant_matmul branch (fully
 * original behavior), anything else the warp/shared-LUT rework. */
static void quant_matmul_launch(float *y, const float *x, const void *w,
        const float *sc, int fmt, int S, int I, int O, size_t rb, int gs, int ng) {
    dim3 grid((unsigned)O, (unsigned)S);
    if (fmt == 8 && f8_warp_mode()) {
        quant_matmul_f8w<<<grid, 256>>>(y, x, w, sc, S, I, O);
        return;
    }
    /* gs <= 0 only: grouped int8 keeps its own branch, whose per-group scale
     * application order is part of its CPU-reference contract. The grid is
     * O/R blocks here, not O -- the one place this path's launch geometry
     * differs from the original's. */
    if (fmt == 1 && gs <= 0) {
        int R = i8_rows_mode();
        if (R) {
            g_i8r_launches++;
            dim3 rgrid((unsigned)((O + R - 1) / R), (unsigned)S);
            /* R*U = 16 independent loads in flight in every shape. */
            if (R == 2)      quant_matmul_i8r<2,8><<<rgrid, 256>>>(y, x, w, sc, S, I, O);
            else if (R == 4) quant_matmul_i8r<4,4><<<rgrid, 256>>>(y, x, w, sc, S, I, O);
            else             quant_matmul_i8r<8,2><<<rgrid, 256>>>(y, x, w, sc, S, I, O);
            return;
        }
    }
    quant_matmul<<<grid, 256>>>(y, x, w, sc, fmt, S, I, O, rb, gs, ng);
}

/* COLI_CUDA_DENSE_PINNED: stage the resident dense GEMV's two host copies
 * through pinned memory instead of handing pageable pointers to the driver.
 *
 *   0 / unset  the original path: two synchronous pageable cudaMemcpy.
 *   1          pinned staging, both copies still synchronous.
 *   2          pinned staging, both copies async on stream 0, ONE sync.
 *
 * Three values rather than two because there are two separate mechanisms in
 * here and lumping them into one flag would produce a number nobody can
 * attribute: 1 prices the pageable staging alone, 2 adds the collapse from
 * two blocking round-trips to one. Anything else reads as off rather than
 * rounding, matching COLI_CUDA_DOWN_ROWS.
 *
 * Off by default. Three measured-and-correct optimisations in this area were
 * worth exactly zero on the token because the time they saved was already
 * being covered by something else; this one has no right to be assumed
 * different until it is measured on the target machine.
 *
 * Deliberately not cached in a static, for the reason graph_mode() gives:
 * tests/test_dense_pinned_cuda.cu flips the variable mid-process to compare
 * the arms against each other, and a latched static would make it measure
 * one arm three times. One getenv against a round-trip measured at ~55 us
 * is not the cost worth saving here. */
static int dense_pinned_mode(void){
    const char *e=getenv("COLI_CUDA_DENSE_PINNED");
    if(!e||!*e) return 0;
    char *end; long v=strtol(e,&end,10);
    if(*end) return 0;
    return (v==1||v==2)?(int)v:0;
}

/* Read by tests/test_dense_pinned_cuda.cu: bitwise identity between the arms
 * is ALSO satisfied by a toggle that never fires, so the test needs a way to
 * tell "identical" from "never ran". */
static uint64_t g_dense_pinned_calls;

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device, int gs) {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->dx, &ctx->dx_cap, xb) || !reserve(&ctx->dy, &ctx->dy_cap, yb)) return 0;

    int pin = dense_pinned_mode();
    if (pin) {
        /* An allocation failure here is a real out-of-memory, not a reason to
         * abort a call that has a correct slower path available. Take it --
         * but SAY SO, once: a run that quietly drops back to pageable copies
         * and reports a flat A/B is the exact failure mode that cost a day on
         * the dense GEMV, and it is indistinguishable from "the toggle does
         * nothing" unless the binary says which arm it actually ran. */
        if (!reserve_pinned(&ctx->host_dx, &ctx->host_dx_cap, xb) ||
            !reserve_pinned(&ctx->host_dy, &ctx->host_dy_cap, yb)) {
            static int warned;
            if (!warned) { warned = 1;
                fprintf(stderr, "[cuda] dense pinned staging unavailable, "
                                "falling back to pageable copies\n"); }
            pin = 0;
        } else {
            /* Same announcement contract as the expert down-rows kernel: under
             * CUDA_DLL=1 every line of this file lives in coli_cuda.dll, which
             * only `make cuda-dll` rebuilds. A script that rebuilds the engine
             * alone sets this variable, changes nothing, and reports a clean
             * flat result. Silent when the variable is unset. */
            static int announced;
            if (!announced) { announced = 1;
                fprintf(stderr, "[cuda] dense pinned staging active: mode=%d\n", pin); }
            g_dense_pinned_calls++;
        }
    }

    /* Opt-in for the same reason the group path's is: four cudaEventRecord on
     * a call this short are not free. Off, this is one cached int. */
    static int dprof = -1;
    if (dprof < 0) { const char *e = getenv("COLI_CUDA_PROFILE"); dprof = e && atoi(e); }
    if (dprof && ctx->ev_dm_ok == 0) {
        int ok = 1;
        for (int i = 0; i < 4; i++)
            if (!cuda_ok(cudaEventCreate(&ctx->ev_dm[i]), "dense profile event create")) {
                for (int j = 0; j < i; j++) cudaEventDestroy(ctx->ev_dm[j]);   /* (#B8) */
                ok = 0; break; }
        ctx->ev_dm_ok = ok ? 1 : -1;      /* latched: retrying overwrites live handles */
    }
    /* Stream 0 explicitly: quant_matmul_launch takes no stream, so the kernel
     * and both cudaMemcpy run on the default one and the events must too. */
    int timed = dprof && ctx->ev_dm_ok > 0;
    std::chrono::steady_clock::time_point t0;
    if (timed) { t0 = std::chrono::steady_clock::now(); cudaEventRecord(ctx->ev_dm[0], 0); }

    /* The host-side half of the pinned arms. It sits INSIDE the wall window on
     * purpose: it is real cost this arm pays and the pageable arm does not, and
     * hiding it would flatter the comparison this toggle exists to make. It is
     * CPU work, so cudaEvent timings do not see it directly -- on an idle
     * stream it widens the ev[0]->ev[1] gap anyway, because ev[0]'s device
     * timestamp is taken before the memcpy starts. Either way it is counted. */
    if (pin) std::memcpy(ctx->host_dx, x, xb);
    const float *src = pin ? ctx->host_dx : x;
    float *dst = pin ? ctx->host_dy : y;

    if (pin == 2) {
        if (!cuda_ok(cudaMemcpyAsync(ctx->dx, src, xb, cudaMemcpyHostToDevice, 0),
                     "input upload")) return 0;
    } else {
        if (!cuda_ok(cudaMemcpy(ctx->dx, src, xb, cudaMemcpyHostToDevice),
                     "input upload")) return 0;
    }
    if (timed) cudaEventRecord(ctx->ev_dm[1], 0);
    quant_matmul_launch(ctx->dy, ctx->dx, t->weights, t->scales, fmt, S, I, O, rb, t->gs, t->ng);
    if (timed) cudaEventRecord(ctx->ev_dm[2], 0);
    if (!cuda_ok(cudaGetLastError(), "matmul launch")) return 0;
    /* Stream 0 for the async arm, same stream the kernel and the events are on,
     * so the ordering H2D -> kernel -> D2H is the stream's own and needs no
     * extra synchronisation to be correct -- only the one below, to know the
     * bytes have landed before they are read. */
    if (pin == 2) {
        if (!cuda_ok(cudaMemcpyAsync(dst, ctx->dy, yb, cudaMemcpyDeviceToHost, 0),
                     "output download")) return 0;
    } else {
        if (!cuda_ok(cudaMemcpy(dst, ctx->dy, yb, cudaMemcpyDeviceToHost),
                     "output download")) return 0;
    }
    if (timed) cudaEventRecord(ctx->ev_dm[3], 0);
    /* The whole point of pin==2: ONE blocking wait for the round-trip instead
     * of two. Nothing below may touch host_dy before it returns. */
    if (pin == 2 && !cuda_ok(cudaStreamSynchronize(0), "dense round-trip synchronize"))
        return 0;
    if (pin) std::memcpy(y, ctx->host_dy, yb);

    if (timed) {
        /* Wall BEFORE the sync, or the sync's own cost lands in the number that
         * exists to price the round-trip. Every arm has already waited for the
         * device by this point -- the pageable and pin==1 arms in their
         * synchronous D2H, pin==2 in the cudaStreamSynchronize above -- so this
         * event sync only waits for the record. */
        double wall = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
        cudaEventSynchronize(ctx->ev_dm[3]);
        float a = 0, b = 0, c = 0;
        cudaEventElapsedTime(&a, ctx->ev_dm[0], ctx->ev_dm[1]);
        cudaEventElapsedTime(&b, ctx->ev_dm[1], ctx->ev_dm[2]);
        cudaEventElapsedTime(&c, ctx->ev_dm[2], ctx->ev_dm[3]);
        std::lock_guard<std::mutex> lock(g_group_stats_mu);
        int index = (int)(ctx - g_ctx);
        g_dense_calls++;
        g_dense_h2d_ms += a; g_dense_kernel_ms += b; g_dense_d2h_ms += c;
        g_dense_wall_ms += wall;
        g_dev_dense_calls[index]++;
        g_dev_dense_h2d_ms[index] += a; g_dev_dense_kernel_ms[index] += b;
        g_dev_dense_d2h_ms[index] += c; g_dev_dense_wall_ms[index] += wall;
        /* Weights AND scales. qwen36_tier.c's qt_dense_init already sizes a
         * resident matrix as I*O + O*ng*4 and this counter feeds a GB/s, so
         * dropping the scale term would understate the traffic and make the
         * kernel look further from peak than it is -- the wrong direction for
         * the one question this exists to answer. ng is 1 for per-row scales,
         * ceil(I/gs) for grouped ones; the dense trunk is int8/int4 with f32
         * scales, which is what sizeof(float) assumes. */
        size_t traffic = rb * (size_t)O + (size_t)O * (size_t)t->ng * sizeof(float);
        g_dense_bytes += traffic;
        g_dev_dense_bytes[index] += traffic;
    }
    return 1;
}

extern "C" void coli_cuda_dense_stats(int device, uint64_t *calls, uint64_t *weight_bytes,
                                      double *h2d_ms, double *kernel_ms,
                                      double *d2h_ms, double *wall_ms) {
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    /* device < 0 is every card summed; a device with no context reports zeros
     * rather than the totals, so a typo in COLI_GPUS cannot read as a result. */
    int index = -1;
    if (device >= 0) { for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) { index = i; break; } }
    if (device >= 0 && index < 0) {
        if (calls) *calls = 0; if (weight_bytes) *weight_bytes = 0;
        if (h2d_ms) *h2d_ms = 0; if (kernel_ms) *kernel_ms = 0;
        if (d2h_ms) *d2h_ms = 0; if (wall_ms) *wall_ms = 0;
        return;
    }
    if (calls)        *calls        = index < 0 ? g_dense_calls     : g_dev_dense_calls[index];
    if (weight_bytes) *weight_bytes = index < 0 ? g_dense_bytes     : g_dev_dense_bytes[index];
    if (h2d_ms)       *h2d_ms       = index < 0 ? g_dense_h2d_ms    : g_dev_dense_h2d_ms[index];
    if (kernel_ms)    *kernel_ms    = index < 0 ? g_dense_kernel_ms : g_dev_dense_kernel_ms[index];
    if (d2h_ms)       *d2h_ms       = index < 0 ? g_dense_d2h_ms    : g_dev_dense_d2h_ms[index];
    if (wall_ms)      *wall_ms      = index < 0 ? g_dense_wall_ms   : g_dev_dense_wall_ms[index];
}

/* MXFP4 matmul, stateless. Separate from coli_cuda_matmul on purpose: that one
 * takes scales as const float* and caches an uploaded tensor, while MXFP4
 * scales are ue8m0 BYTES -- passing them through the float* parameter would
 * compile and silently reinterpret the buffer. Kimi K3's routed experts stream
 * (a fill-once tier at decode), so there is nothing to cache here anyway; the
 * weights go up with the call.
 *
 * Returns 0 and leaves y untouched on any failure, which is the contract the
 * engine's GPU paths already use to fall back to CPU. */
extern "C" int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                      const uint8_t *q4, const uint8_t *e8s,
                                      int S, int I, int O) {
    if (fault_injected()) return 0;
    if (S < 1 || I < 1 || O < 1 || !y || !x || !q4 || !e8s) return 0;
    DeviceContext *ctx = find_ctx(0);
    if (!select_ctx(ctx)) return 0;

    size_t rb = (size_t)(I + 1) / 2, ng = (size_t)(I + 31) / 32;
    size_t wb = (size_t)O * rb, sb = (size_t)O * ng;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);

    uint8_t *dw = nullptr, *ds = nullptr;
    if (!cuda_ok(cudaMalloc(&dw, wb), "mxfp4 weight alloc")) return 0;
    if (!cuda_ok(cudaMalloc(&ds, sb), "mxfp4 scale alloc")) { cudaFree(dw); return 0; }

    int ok = reserve(&ctx->x, &ctx->x_cap, xb) && reserve(&ctx->y, &ctx->y_cap, yb) &&
             cuda_ok(cudaMemcpy(dw, q4, wb, cudaMemcpyHostToDevice), "mxfp4 weight upload") &&
             cuda_ok(cudaMemcpy(ds, e8s, sb, cudaMemcpyHostToDevice), "mxfp4 scale upload") &&
             cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "mxfp4 input upload");
    if (ok) {
        dim3 grid((unsigned)O, (unsigned)S);
        quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, dw, reinterpret_cast<const float *>(ds),
                                    7, S, I, O, rb, 32, (int)ng);
        ok = cuda_ok(cudaGetLastError(), "mxfp4 launch") &&
             cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "mxfp4 output download");
    }
    cudaFree(dw);
    cudaFree(ds);
    return ok;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    quant_matmul_launch(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng);
    quant_matmul_launch(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 && !e8_rot_rows_dev(ctx->gate, S, I, 0)) return 0;
    quant_matmul_launch(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||!COLI_GPU_HAS_WMMA||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Single launch site for the fmt=8 group kernels, shared by the sync and the
 * issue/take dispatches so both stay one-line call sites (rebase-tolerant
 * against #935's e8_group_launch refactor of the adjacent branch).
 * COLI_CUDA_F8_WARP (docs/ENVIRONMENT.md, parsed by f8_warp_mode): 1 = warp
 * kernels (the CUDA default; HIP defaults to 0), 0 = the original per-(o,s)
 * kernels (field escape hatch, dense path included via quant_matmul_launch),
 * 2 = warp kernels decoding through cuda_fp8.h — a real cvt instruction only
 * on sm_89+, the header's bit-manip emulation below that, and plain =1
 * behavior where cuda_fp8.h is absent (HIP); experimental until the
 * 256-value sweep certifies it bit-identical on the target silicon. */
static void f8_group_launch(DeviceContext *ctx,GroupDesc *dev,int I,int D,
                            int max_rows,int count){
    int mode=f8_warp_mode();
    if(mode){
        dim3 hg((unsigned)((I+7)/8),(unsigned)count),og((unsigned)((D+7)/8),(unsigned)count);
#if COLI_F8_HWCVT
        if(mode==2){
            grouped_hidden_f8w_dual<1><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_f8w<1><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
            return;
        }
#endif
        grouped_hidden_f8w_dual<0><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        grouped_down_f8w<0><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        return;
    }
    dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
    grouped_hidden_f8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
    grouped_down_f8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
}

static int expert_group_impl(ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             const int *rows, int count,
                             float *y, const float *x,
                             int pin_small_batch) {
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0,any_e8=0,all_e8=1,any_f8=0,all_f8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs,total};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    /* Mixed E8/FP8 groups cannot use a homogeneous grouped kernel. */
    if((any_e8&&!all_e8)||(any_f8&&!all_f8)){
        int off=0;
        for(int c=0;c<count;c++){
            if(!coli_cuda_expert_mlp(gates[c],ups[c],downs[c],
                    y+(size_t)off*D,x+(size_t)off*D,rows[c])) return 0;
            off+=rows[c];
        }
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")){
        for(int j=0;j<i;j++) cudaEventDestroy(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    /* grouped_s4_wmma's body needs __CUDA_ARCH__>=750 AND the s4 fragment type:
     * on a build where either is missing (COLI_HIP_NO_WMMA, or rocWMMA, which
     * has no sub-byte precision and defines __CUDA_ARCH__ 700) the launch
     * succeeds with an EMPTY kernel and the output buffer silently keeps the
     * previous call's data (#1499). So the gate mirrors the body's own guard:
     * the s4 flag from backend_gpu_compat.h plus the device's compute
     * capability, not WMMA availability alone. */
    tc=tc&&!pin_small_batch&&COLI_GPU_HAS_WMMA&&COLI_GPU_HAS_S4_WMMA&&
       (ctx->compute_major*10+ctx->compute_minor>=75)&&
       all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(all_e8){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_f8){
        /* fp8-e4m3 groups: silu fused in the dual epilogue, like the w4/g4 duals. */
        f8_group_launch(ctx,dev,I,D,max_rows,count);
    }else if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(!pin_small_batch&&all_s4&&COLI_GPU_HAS_WMMA&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a Tensor Core tile. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback). */
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        down_g4_launch(ctx->y,ctx->gate,dev,D,I,max_rows,count,ctx->stream);
    }else{
        /* generic path decodes fmt 0/1/2/3 only — refuse everything else rather
         * than whitelist known offenders: a fmt=4 group that slipped the gates
         * above (odd gs) must NOT be silently decoded as int2 (#334), and any
         * group/block-scaled format that gains CUDA tensors later (fmt=5, fmt=8)
         * carries scale geometry this kernel's epilogue does not apply.
         * STRICTER than weight_at's own admissible set on purpose: fmt=4 belongs
         * on the g4 path above, and returning 0 here keeps the CORRECT CPU
         * fallback, which is the outcome we want — weight_at's device-side
         * __trap backstop is for a launch that got past a gate like this one,
         * not a substitute for having the gate. */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          int index=(int)(ctx-g_ctx);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c;
          g_device_group_h2d_ms[index]+=a;
          g_device_group_kernel_ms[index]+=b;
          g_device_group_d2h_ms[index]+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,0);
}

extern "C" int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gates,
                                               ColiCudaTensor *const *ups,
                                               ColiCudaTensor *const *downs,
                                               const int *rows, int count,
                                               float *y, const float *x,
                                               int pin_small_batch) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,pin_small_batch);
}

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
/* x_rows: how many rows `x` holds. 0 means the original contract -- one input
 * row per expert row -- and is what the legacy export below passes, so an old
 * caller keeps its exact behavior. 1 means every chunk reads the same row.
 *
 * This is a SEPARATE export rather than a wider signature on the old one
 * because the two halves ship apart: under CUDA_DLL=1 the kernels live in
 * coli_cuda.dll and only `make cuda-dll` rebuilds it, so a new engine can
 * meet an old DLL. Widening the old symbol would have that DLL ignore x_rows
 * and read total*D floats out of a one-row buffer. The loader resolves this
 * one optionally and the caller keeps its duplication path for when it is
 * absent. */
/* Read by tests/test_grouped_g4_cuda.cu. The graph has two observable states
 * -- captured and replayed -- and without a way to tell them apart a test
 * cannot check that a moved buffer causes a RE-CAPTURE rather than a replay
 * against a freed pointer. Bitwise identity does not distinguish them: a
 * stale graph that happens to read memory nothing else has reused yet
 * returns the right answer too. */
static uint64_t g_graph_captures, g_graph_replays;

/* One call's worth of group accounting. Exists because the graph replay path
 * returns early and silently stopped counting when it was added. */
static void group_account(DeviceContext *ctx,int count,int total){
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    int index=(int)(ctx-g_ctx);
    g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
    g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
    g_device_group_rows[index]+=(uint64_t)total;
}

extern "C" int coli_cuda_expert_group_issue_x(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x, int x_rows) {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    if (x_rows < 0) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1,any_e8=0,all_e8=1,
        all_q4=1,any_g4=0,any_f8=0,all_f8=1;
    GroupDesc host[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs,total};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(any_e8&&!all_e8) return 0;
    if(any_f8&&!all_f8) return 0;   /* mixed FP8: no homogeneous kernel, sync path has the per-expert loop */
    if(total>8) return 0;                       /* decode-scale only */
    /* x_rows says how many rows `x` actually holds. Two shapes are accepted
     * and nothing else: one input row per expert row (x_rows == total, the
     * original contract), or ONE row shared by every chunk (x_rows == 1),
     * which is decode -- K experts, one token, one activation vector. The
     * broadcast form requires rows[c] == 1 everywhere, because a chunk with
     * two rows would read xoff+1 and there is no such row. Anything else is
     * refused rather than guessed at: a wrong x_rows reads off the end of the
     * caller's buffer. */
    if(x_rows==0) x_rows=total;          /* legacy contract */
    if(x_rows!=total){
        if(x_rows!=1||max_rows!=1) return 0;
        for(int c=0;c<count;c++) host[c].xoff=0;
    }
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    /* Input and output are sized apart now that they can differ: x carries
     * x_rows rows, y still carries one per expert row. */
    size_t xb=(size_t)x_rows*D*sizeof(float), yb=(size_t)total*D*sizeof(float),
           ib=(size_t)total*I*sizeof(float);
    /* A graph points at addresses. reserve() only grows, so in steady state
     * nothing moves -- but the first calls do grow these, and a graph captured
     * before that points at freed memory. Rather than assume warm-up is over,
     * watch the pointers: any reallocation bumps buf_gen and every graph
     * captured under the old generation is thrown away. */
    /* EIGHT pointers, not five. A graph bakes in every address it was
     * captured with, and three of them are on the HOST: the capture records
     * cudaMemcpyAsync(group_desc <- host_desc), (x <- host_x) and
     * (host_y <- y). Watching only the device buffers left a hole where
     * reserve_pinned could cudaFreeHost one of those and hand the graph a
     * dangling address with buf_gen unchanged -- the signature would still
     * match, the replay would still be attempted, and it would read freed
     * pinned memory.
     *
     * AND THE COMPARISON RUNS EVEN WHEN A RESERVE FAILS. It used to sit
     * behind the `return 0`, so a reserve that succeeded and MOVED a buffer,
     * followed by one that failed, left the old pointers armed in a graph
     * that had just been freed underneath. reserve() nulls what it could not
     * grow, so a failure always shows up in this comparison -- but only if
     * the comparison is reached. */
    const void *pre[8]={ctx->x,ctx->y,ctx->gate,ctx->up,ctx->group_desc,
                        ctx->host_x,ctx->host_y,ctx->host_desc};
    int reserved_=reserve(&ctx->x,&ctx->x_cap,xb)&&reserve(&ctx->y,&ctx->y_cap,yb)&&
       reserve(&ctx->gate,&ctx->gate_cap,ib)&&reserve(&ctx->up,&ctx->up_cap,ib)&&
       reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))&&
       reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)&&
       reserve_pinned(&ctx->host_y,&ctx->host_y_cap,yb)&&
       reserve_pinned_bytes(&ctx->host_desc,&ctx->host_desc_cap,
                            (size_t)count*sizeof(GroupDesc));
    if(pre[0]!=ctx->x||pre[1]!=ctx->y||pre[2]!=ctx->gate||pre[3]!=ctx->up||
       pre[4]!=ctx->group_desc||pre[5]!=ctx->host_x||pre[6]!=ctx->host_y||
       pre[7]!=ctx->host_desc){
        ctx->buf_gen++;
#if COLI_GPU_HAS_GRAPH
        /* The signature check alone would stop them being USED; they still
         * have to be freed, or a run that keeps growing its buffers keeps a
         * dead graph per generation. */
        for(int gi=0;gi<9;gi++) if(ctx->graph_exec[gi]){
            cudaGraphExecDestroy(ctx->graph_exec[gi]);
            ctx->graph_exec[gi]=nullptr; ctx->graph_gen[gi]=0; }
#endif
    }
    if(!reserved_) return 0;          /* after the invalidation, never before */
    std::memcpy(ctx->host_desc,host,(size_t)count*sizeof(GroupDesc));
    std::memcpy(ctx->host_x,x,xb);
    /* Timing is opt-in because measuring it changes it: four cudaEventRecord
     * per call, 40 calls a token, is launch overhead added to the path whose
     * launch overhead is the thing in question. Off by default the branch is
     * one already-read getenv and nothing reaches the driver. */
    static int gprof=-1;   /* read once: 40 calls a token is no place for getenv */
    if(gprof<0){ const char *e=getenv("COLI_CUDA_PROFILE"); gprof=e&&atoi(e); }
    if(gprof&&ctx->ev_grp_ok==0){
        int ok=1;
        for(int i=0;i<4;i++)
            if(!cuda_ok(cudaEventCreate(&ctx->ev_grp[i]),"group profile event create")){
                /* (#B8) don't leak the events already created -- and latch the
                 * failure, or the next call re-enters this loop and overwrites
                 * whatever handles it did manage to allocate. */
                for(int j=0;j<i;j++) cudaEventDestroy(ctx->ev_grp[j]);
                ok=0; break; }
        ctx->ev_grp_ok=ok?1:-1;
    }
    ctx->group_timed=gprof&&ctx->ev_grp_ok>0;
    /* Which of the five dispatch arms will run. Computed HERE, before
     * anything is enqueued, because the graph is keyed on it: a group whose
     * formats change must not replay a graph captured for other kernels.
     * Only arms 3 and 4 are ever captured -- arm 1 can `return 0` in the
     * middle of its sequence (e8_rot_rows_dev) and arm 5 can return before
     * launching anything, and a return inside a stream capture leaves the
     * stream captured. */
    int w4p_=!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED"));
    int branch_= all_e8?1 : all_f8?2 : (all_s4&&w4p_)?3 : (all_q4&&any_g4)?4 : 5;
    unsigned gsig_=(ctx->buf_gen<<8)|((unsigned)branch_<<4)|(unsigned)count;
    int graphable_= graph_mode() && !ctx->group_timed && x_rows==1 &&
                    count>=1 && count<=8 && (branch_==3||branch_==4);
#if COLI_GPU_HAS_GRAPH
    if(graphable_ && ctx->graph_exec[count] && ctx->graph_gen[count]==gsig_){
        /* Everything the launch depends on is already in the graph except the
         * CONTENTS of host_desc and host_x, which were just refreshed above.
         * One driver call replaces the descriptor copy, the input copy, two
         * kernels and the readback. */
        if(!cuda_ok(cudaGraphLaunch(ctx->graph_exec[count],ctx->stream),
                    "expert group graph launch")) return 0;
        ctx->group_pending=1; ctx->group_pending_bytes=yb;
        g_graph_replays++;
        /* The replay is a call, an expert count and a row count like any
         * other. This early return used to skip the accounting, so with the
         * graph on by default `qt_stats` reported only the CAPTURES -- tens
         * of calls where there were thousands. It never corrupted a published
         * number, because the graph refuses to run under COLI_CUDA_PROFILE
         * and every recorded group_stats figure came from a profiled pass,
         * but a counter that is right only when you are watching it is not a
         * counter. Shared helper rather than a second copy, so the two sites
         * cannot drift again. */
        group_account(ctx,count,total);
        return 1;
    }
    int capturing_=0;
    if(graphable_ && !(ctx->graph_exec[count]&&ctx->graph_gen[count]==gsig_) &&
       ctx->graph_gen[count]!=gsig_){
        /* graph_gen == gsig with a null exec means "tried at this signature
         * and failed"; do not retry every call. */
        capturing_ = cudaStreamBeginCapture(ctx->stream,
                        cudaStreamCaptureModeThreadLocal)==cudaSuccess;
    }
#endif
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,ctx->host_desc,
                                (size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue descriptors")){
#if COLI_GPU_HAS_GRAPH
        /* A return inside a stream capture leaves the stream captured, and
         * every later launch on it fails. Close it before leaving. */
        if(capturing_){ cudaGraph_t g_=nullptr; cudaStreamEndCapture(ctx->stream,&g_);
                        if(g_) cudaGraphDestroy(g_); cudaGetLastError(); }
#endif
        return 0;
    }
    /* ev_grp[0] goes AFTER the descriptor copy because expert_group_impl puts
     * ev[0] after its own: "h2d" is already a defined quantity in this file,
     * and both paths add into the same counter. Widening it here would make
     * the printed total the sum of two different measurements -- which is the
     * failure this instrumentation exists to end, not to repeat. */
    if(ctx->group_timed) cudaEventRecord(ctx->ev_grp[0],ctx->stream);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue upload")){
#if COLI_GPU_HAS_GRAPH
        /* A return inside a stream capture leaves the stream captured, and
         * every later launch on it fails. Close it before leaving. */
        if(capturing_){ cudaGraph_t g_=nullptr; cudaStreamEndCapture(ctx->stream,&g_);
                        if(g_) cudaGraphDestroy(g_); cudaGetLastError(); }
#endif
        return 0;
    }
    if(ctx->group_timed) cudaEventRecord(ctx->ev_grp[1],ctx->stream);
    if(all_e8){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_f8){
        /* fp8-e4m3 groups on the async decode path: same launch helper as the
         * sync dispatch, silu fused in the dual epilogue. */
        f8_group_launch(ctx,(GroupDesc*)ctx->group_desc,I,D,max_rows,count);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual) grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else {
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(
                ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else if(all_q4&&any_g4){
        /* grouped int4 (fmt=4) present in the async decode path: per-group
         * scales via the #334 kernels (fmt=2 members ride along as ng=1). The
         * previous fallback ran quant_matmul with gs=0,ng=1, which silently
         * applied one per-row scale to a grouped container -> wrong output. */
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        /* silu is fused in the dual kernel's epilogue (like the sync path):
         * an extra silu_mul here would re-apply it against the never-written
         * ctx->up buffer. */
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        down_g4_launch(ctx->y,ctx->gate,dev,D,I,max_rows,count,ctx->stream);
    } else {
        /* Fallback runs quant_matmul with gs=0,ng=1 — per-row-scale semantics.
         * That is only correct for fmt 0/1/2/3: refuse group/block-scaled
         * members (fmt=4 with odd gs today; fmt=5/8 if they ever gain CUDA
         * tensors) instead of silently mis-scaling them, mirroring the sync
         * path's refusal (#334). fmt=6 cannot reach here (any_e8 gates above). */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        /* x16 is the ACTIVATION and takes xoff; y16 is this expert's output
         * and keeps offset. Same miss as the fp8 warp kernel: this generic
         * arm (fmt 0/1/3 -- an int8 expert tier lands here) was not converted
         * when GroupDesc gained xoff. */
        float *x16=ctx->x+(size_t)host[c].xoff*D,*y16=ctx->y+(size_t)host[c].offset*D;
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
            host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
            host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
        silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
        quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
            host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
    }}
    if(ctx->group_timed) cudaEventRecord(ctx->ev_grp[2],ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"expert group issue launch")||
       /* yb, not xb. The two were the same number until x_rows existed, and
        * the readback needs the OUTPUT size: with a broadcast input xb is one
        * row, so this copied expert 0's result and left the other seven rows
        * of host_y holding whatever the previous call had put there. qt_take
        * summed stale memory into the token -- wrong logits, a generation
        * that diverges, and a profile that reads as a 14 % slowdown. */
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,yb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group issue download")){
#if COLI_GPU_HAS_GRAPH
        /* A return inside a stream capture leaves the stream captured, and
         * every later launch on it fails. Close it before leaving. */
        if(capturing_){ cudaGraph_t g_=nullptr; cudaStreamEndCapture(ctx->stream,&g_);
                        if(g_) cudaGraphDestroy(g_); cudaGetLastError(); }
#endif
        return 0;
    }
    if(ctx->group_timed) cudaEventRecord(ctx->ev_grp[3],ctx->stream);
#if COLI_GPU_HAS_GRAPH
    if(capturing_){
        /* Nothing above actually ran: capture records, it does not execute.
         * Close it, instantiate, and launch the result -- so this call gets
         * its work done by the same graph every later call will replay. */
        cudaGraph_t g=nullptr;
        cudaError_t ec=cudaStreamEndCapture(ctx->stream,&g);
        /* Mark the attempt BEFORE it can fail: a null exec at this signature
         * is how the branch above remembers not to try again until something
         * changes. */
        ctx->graph_gen[count]=gsig_;
        if(ctx->graph_exec[count]){ cudaGraphExecDestroy(ctx->graph_exec[count]);
                                    ctx->graph_exec[count]=nullptr; }
        if(ec==cudaSuccess&&g){
            if(cudaGraphInstantiate(&ctx->graph_exec[count],g,0)!=cudaSuccess)
                ctx->graph_exec[count]=nullptr;
            cudaGraphDestroy(g);
        }
        cudaGetLastError();                      /* swallow a failed capture */
        if(!ctx->graph_exec[count]){
            /* Capture or instantiate failed and NOTHING was enqueued. Refuse
             * the call: qt_issue hands these experts to the CPU for this one
             * layer, which is slower and correct. Silently returning success
             * here would leave host_y holding the previous call's rows. */
            static int said=0;
            if(!said){ said=1;
                fprintf(stderr,"[cuda] expert group graph capture failed; "
                               "falling back to per-call launches\n"); }
            return 0;
        }
        if(!cuda_ok(cudaGraphLaunch(ctx->graph_exec[count],ctx->stream),
                    "expert group graph launch")) return 0;
        /* Gated on the variable being SET, for the reason down_g4_launch
         * gives: on by default, so an unset run must not grow a log line.
         * The capture-FAILED message above stays unconditional -- that one is
         * a problem the user needs to see whether or not they asked. */
        g_graph_captures++;
        static int said_ok=0;
        if(!said_ok){ said_ok=1;
            const char *ge=getenv("COLI_CUDA_GRAPH");
            if(ge&&*ge)
                fprintf(stderr,"[cuda] expert group graph active (count=%d)\n",count); }
    }
#endif
    ctx->group_pending=1; ctx->group_pending_bytes=yb;   /* the readback, not the upload */
    group_account(ctx,count,total);
    return 1;
}

/* The original symbol, unchanged for every caller and every old DLL. */
extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                            ColiCudaTensor *const *ups,
                                            ColiCudaTensor *const *downs,
                                            const int *rows, int count,
                                            const float *x) {
    return coli_cuda_expert_group_issue_x(gates,ups,downs,rows,count,x,0);
}

/* Linked directly (no CUDA_DLL), so the broadcast form is always present.
 * backend_loader.c has the other definition, which asks the DLL. */
extern "C" int coli_cuda_has_group_x_broadcast(void){ return 1; }

extern "C" const float *coli_cuda_expert_group_take(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group take")) return nullptr;
    /* The stream is drained, so all four events have completed and no extra
     * synchronisation is needed to read them. */
    if(ctx->group_timed){
        float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ctx->ev_grp[0],ctx->ev_grp[1]);
        cudaEventElapsedTime(&b,ctx->ev_grp[1],ctx->ev_grp[2]);
        cudaEventElapsedTime(&c,ctx->ev_grp[2],ctx->ev_grp[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          int index=(int)(ctx-g_ctx);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c;
          g_device_group_h2d_ms[index]+=a;
          g_device_group_kernel_ms[index]+=b;
          g_device_group_d2h_ms[index]+=c; }
        ctx->group_timed=0;
    }
    return ctx->host_y;
}


/* The absorb kernels decode `w` through weight_at + absorb_scale, which know
 * per-row and fmt=4 group scales only. Refuse anything else (fmt=5/6/8) rather
 * than mis-decode it — the caller keeps its CPU attention path. (`proj`
 * tensors are exempt: they run through quant_matmul, which dispatches every
 * format it uploads.) A dedicated block-scale absorb for fmt=8 is follow-up
 * work, same shape as routing fmt=4 through the grouped kernels was.
 *
 * The admissible set is weight_at's own, taken from the shared predicate rather
 * than restated as `fmt <= 4`: this gate and weight_at's device-side backstop
 * must not be able to drift apart, and the old inequality also admitted
 * NEGATIVE fmt values, which weight_at would then have fallen through on. Same
 * truth table for every fmt a container can actually carry (0..8), so no
 * existing container changes behaviour here. */
static int absorb_fmt_ok(const ColiCudaTensor *w){
    return w && coli_cuda_weight_at_supported(w->fmt);
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!old||!add||!off){std::free(old);std::free(add);std::free(off);return 0;}
    int page_stride=0;
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            ragged_kv_clear(e);
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        int need=(lengths[s]+COLI_KV_PAGE_TOKENS-1)/COLI_KV_PAGE_TOKENS;
        if(need>e->page_count){
            float **nl=(float**)std::calloc((size_t)need,sizeof(*nl));
            float **nr=(float**)std::calloc((size_t)need,sizeof(*nr));
            if(!nl||!nr){std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;}
            for(int i=0;i<e->page_count;i++){nl[i]=e->latent_pages[i];nr[i]=e->rope_pages[i];}
            int made=e->page_count;
            for(;made<need;made++){
                if(!cuda_ok(cudaMalloc(&nl[made],(size_t)COLI_KV_PAGE_TOKENS*K*sizeof(float)),"ragged KV latent page")||
                   !cuda_ok(cudaMalloc(&nr[made],(size_t)COLI_KV_PAGE_TOKENS*R*sizeof(float)),"ragged KV rope page"))break;
            }
            if(made<need){
                if(nl[made])cudaFree(nl[made]);if(nr[made])cudaFree(nr[made]);
                for(int i=e->page_count;i<made;i++){cudaFree(nl[i]);cudaFree(nr[i]);}
                std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;
            }
            std::free(e->latent_pages);std::free(e->rope_pages);
            e->latent_pages=nl;e->rope_pages=nr;e->page_count=need;
        }
        if(e->page_count>page_stride)page_stride=e->page_count;
        old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t table_n=(size_t)S*page_stride;
    float **dl=(float**)std::calloc(table_n,sizeof(*dl));
    float **dr=(float**)std::calloc(table_n,sizeof(*dr));
    if(!dl||!dr){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++)for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){
        for(int p=0;p<w->ragged[i].page_count;p++){
            dl[(size_t)s*page_stride+p]=w->ragged[i].latent_pages[p];
            dr[(size_t)s*page_stride+p]=w->ragged[i].rope_pages[p];
        }
        break;
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=2*table_n*sizeof(float*)+(size_t)S*4*sizeof(int);
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+table_n;
    int *dn=(int*)(ddr+table_n),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok=cuda_ok(cudaMemcpyAsync(dc->al,dc->host_kv,pb,cudaMemcpyHostToDevice,dc->stream),"ragged KV append upload");
    }
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(ddl,dl,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged latent page table")&&
             cuda_ok(cudaMemcpyAsync(ddr,dr,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged rope page table")&&
             cuda_ok(cudaMemcpyAsync(dn,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload")&&
             cuda_ok(cudaMemcpyAsync(dold,old,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged old lengths")&&
             cuda_ok(cudaMemcpyAsync(dadd,add,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append lengths")&&
             cuda_ok(cudaMemcpyAsync(doff,off,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append offsets");
    if(ok&&pb)ragged_kv_append<<<S,256,0,dc->stream>>>(ddl,ddr,dc->al,dold,dadd,doff,K,R,page_stride);
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,ddl,ddr,
        dn,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,page_stride,scale,w->gs,w->ng);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        /* Must mirror the upload's accounting exactly -- literally the same
         * expression upload uses to charge (scale_count * sizeof(float), gated
         * on fmt=6 never having a separate scale buffer), so the two can no
         * longer drift independently. Over-subtracting here trips the >= guard
         * below, which silently leaves the tensor's bytes on the device counter
         * forever. */
        size_t storage_bytes =
#ifdef COLI_ANS
            tensor->compressed ? tensor->archive_bytes :
#endif
            tensor->weight_bytes;
        size_t bytes = storage_bytes +
            ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights&&tensor->weights_owned) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    for(int i=0;i<tensor->ragged_count;i++)ragged_kv_clear(&tensor->ragged[i]);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    /* Must mirror upload's and free's accounting exactly -- literally the same
     * expression they use (scale_count * sizeof(float), gated on fmt=6 never
     * having a separate scale buffer) -- so all three can no longer drift
     * independently. The prior `O * ng` shape over-reported for fmt=8 (real
     * footprint is (O+127)/128 * ng block scales, not O * ng) and for fmt=6
     * (which has no separate scale buffer at all). */
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    return storage_bytes +
        ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
}

/* What a cudaMalloc of `bytes` actually takes off the card.
 *
 * coli_cuda_tensor_bytes() above is the LOGICAL size and must stay that way -
 * it mirrors upload and free so the three cannot drift. This is a different
 * question: the allocator rounds a request up, and nothing was charging the
 * difference to the expert budget.
 *
 * Measured on an RTX 3090 (sm_86, CUDA 13.1):
 *
 *     request              actual      overhead
 *     0.75 MiB          1.00 MiB        +0.25     <- int4-g64 scale array
 *     1.00 MiB          1.00 MiB        +0.00
 *     1.00 MiB + 1 B    2.00 MiB        +1.00
 *     1.50 MiB          2.00 MiB        +0.50
 *     2.00 MiB          2.00 MiB        +0.00
 *     2.00 MiB + 1 B    4.00 MiB        +2.00
 *     6.00 MiB          6.00 MiB        +0.00     <- int4-g64 weight tensor
 *
 * A GLM-5.2 int4-g64 expert is three 6 MiB weight arrays, which are already on
 * a boundary and cost nothing extra, and three 0.75 MiB scale arrays, each
 * padded to 1 MiB. 3 x 0.25 = 0.75 MiB per expert unaccounted - which is the
 * 0.741 MiB/expert (sigma 0.019) measured independently on H100 and H200 in
 * #687, from the other direction.
 *
 * PROBED, not modelled. The table above is not a rule this hardcodes: the
 * rounding is a driver and architecture property, and a formula fitted to one
 * card would be silently wrong on the next. Probed once per distinct size
 * and cached, so the cost does not scale with the tier.
 *
 * cudaMemGetInfo is the obvious alternative and it does not survive contact
 * with the numbers: it is O(live allocations), measured 0.52 us empty and
 * 68.2 us with 12,000 live, so charging it per expert would be quadratic
 * across a tier that places thousands.
 *
 * Returns `bytes` unchanged if the probe cannot run. Under-reporting is the
 * pre-existing behaviour and degrades to exactly what this replaced; refusing
 * to size at all would be worse.
 */
#define COLI_CUDA_FOOTPRINT_CACHE 16
static struct { size_t req, real; } g_fp_cache[COLI_CUDA_FOOTPRINT_CACHE];
static int g_fp_cache_n = 0;

/* The probe MUST allocate several buffers and divide, not one and measure it.
 *
 * A single cudaMalloc reserves a whole 2 MiB VMM page, so one allocation of
 * anything smaller reads as a flat 2 MiB and the answer is the page size
 * rather than the per-allocation cost. Later allocations suballocate from
 * pages already reserved, so the real amortised figure only appears once
 * enough of them are live to fill a page. Measured both ways on sm_86, same
 * 786,432-byte request:
 *
 *     1 allocation    2.00 MiB   <- the page, not the allocation
 *     64 allocations  1.00 MiB   <- the truth, and what the tier will pay
 *
 * The single-shot version of this function shipped in an earlier draft and its
 * own test caught it: it over-reported every expert by 3x and would have
 * shrunk the tier far more than the bug it fixes.
 */
#define COLI_CUDA_FOOTPRINT_PROBE_BYTES (24u << 20)   /* transient probe budget */
#define COLI_CUDA_FOOTPRINT_PROBE_MAX   64

extern "C" size_t coli_cuda_alloc_footprint(size_t bytes) {
    if (!bytes) return 0;
    for (int i = 0; i < g_fp_cache_n; i++)
        if (g_fp_cache[i].req == bytes) return g_fp_cache[i].real;

    /* Enough allocations to cross a page boundary, capped so the probe never
     * takes a meaningful bite out of a card that is about to be filled. */
    int want = (int)(COLI_CUDA_FOOTPRINT_PROBE_BYTES / bytes);
    if (want < 8) want = 8;
    if (want > COLI_CUDA_FOOTPRINT_PROBE_MAX) want = COLI_CUDA_FOOTPRINT_PROBE_MAX;

    size_t free_before = 0, free_after = 0, total = 0, real = bytes;
    void *p[COLI_CUDA_FOOTPRINT_PROBE_MAX];
    int made = 0;
    if (cudaMemGetInfo(&free_before, &total) == cudaSuccess) {
        for (int i = 0; i < want; i++) {
            if (cudaMalloc(&p[i], bytes) != cudaSuccess) break;
            made++;
        }
        if (made > 0 && cudaMemGetInfo(&free_after, &total) == cudaSuccess &&
            free_before > free_after) {
            size_t avg = (free_before - free_after) / (size_t)made;
            /* Only ever round UP. A concurrent free elsewhere on the device
             * can make the delta read small, and charging less than the
             * logical size is the failure this exists to fix. */
            if (avg > real) real = avg;
        }
        for (int i = 0; i < made; i++) cudaFree(p[i]);
    }
    /* Cache even a failed probe: `real` is then the logical size, which is the
     * pre-existing behaviour, and retrying per expert on a card that just
     * refused 8 allocations would be the worst possible time to try again. */
    if (g_fp_cache_n < COLI_CUDA_FOOTPRINT_CACHE)
        g_fp_cache[g_fp_cache_n++] = { bytes, real };
    return real;
}

/* Same split as coli_cuda_tensor_bytes, but per ALLOCATION rather than summed
 * first: the weights and the scales are two separate cudaMallocs and each is
 * rounded on its own. Summing then rounding once would miss the scale array's
 * padding entirely, which is the whole term. */
extern "C" size_t coli_cuda_tensor_vram(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    size_t total = coli_cuda_alloc_footprint(storage_bytes);
    if (tensor->fmt && tensor->fmt != 6)
        total += coli_cuda_alloc_footprint(tensor->scale_count * sizeof(float));
    return total;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=27||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection runs single-thread so the
 * argmax order, tie-breaking (strict >, lowest index wins) and weight math
 * match the CPU reference exactly; only the dot/expf rounding can differ,
 * which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
__global__ void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, float *logit, float *choice){
    int e = blockIdx.x;
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    for(int i=threadIdx.x; i<D; i+=blockDim.x) acc += x[i]*w[i];
    __shared__ float sh[128];
    sh[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x>>1; s>0; s>>=1){
        if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
        __syncthreads();
    }
    if(!threadIdx.x){
        float lg = 1.f/(1.f+expf(-sh[0]));
        logit[e]=lg; choice[e]=lg+bias[e];
    }
}
__global__ void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    if(threadIdx.x||blockIdx.x) return;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best];
    }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    pipe_router_logits<<<E,128>>>(x_dev,(const float*)rw_dev,(const float*)rb_dev,D,logit,chc);
    pipe_router_select<<<1,1>>>(logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(!cuda_ok(cudaGetLastError(),"pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    if(!cuda_ok(cudaMemcpy(buf,out,pack,cudaMemcpyDeviceToHost),"pipe router readback")) return 0;
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
__global__ static void bcast_row(float *dst,const float *src,int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
__global__ static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
__global__ static void sum_slots(float *dst,const float *slots,int n,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
extern "C" int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev){
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs,total};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total++;
    }
    if(!all_s4) return 0;                       /* resident path: per-row int4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    if(!ctx->ev_done_ok){
        if(!cuda_ok(cudaEventCreateWithFlags(&ctx->ev_done,cudaEventDisableTiming),
                    "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group desc")||
       !cuda_ok(cudaMemcpyAsync(w_dev,weights,(size_t)count*sizeof(float),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group weights"))
        return 0;
    /* input row: P2P from the home device. The caller guarantees x_src_dev is
     * materialized (the pre-moe nrm download already synced the home stream). */
    if(!cuda_ok(cudaMemcpyPeerAsync(ctx->x,device,x_src_dev,home_device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident group x p2p"))
        return 0;
    bcast_row<<<64,256,0,ctx->stream>>>(ctx->x,ctx->x,count,D);   /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself) */
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dim3 hg((unsigned)I,1,(unsigned)count),og((unsigned)D,1,(unsigned)count);
    grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);  /* silu fused in epilogue */
    grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
    if(!cuda_ok(cudaMemcpyPeerAsync(partial_slot_dev,home_device,partial_local,device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident partial p2p"))
        return 0;
    if(!cuda_ok(cudaEventRecord(ctx->ev_done,ctx->stream),"resident event record")) return 0;
    return cuda_ok(cudaGetLastError(),"resident group launch");
}
extern "C" int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,
                                           float *slots_dev,float *acc_dev,int D){
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if(!cuda_ok(cudaStreamWaitEvent(0,src->ev_done,0),"resident take wait")) return 0;
    }
    sum_slots<<<48,256>>>(acc_dev,slots_dev,n_issued,D);          /* legacy stream: ordered with pipe_* */
    return cuda_ok(cudaGetLastError(),"resident take reduce");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if (fault_injected()) return 0;
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I),t->gs,t->ng);
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
