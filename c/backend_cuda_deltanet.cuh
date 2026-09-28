/* Qwen3.6 DeltaNet decode on the GPU: the kernels between the two placed
 * GEMVs (docs/qwen36-deltanet-gpu-plan.md, stage 1).
 *
 * One decode step of one DeltaNet layer, S = 1, everything already on the
 * device:
 *
 *   qkvz  [conv_dim + value_dim]   the dnproj GEMV's output (qkv ++ z)
 *   x     [hidden]                 the layer's normed input (for a and b)
 *
 *   dn_conv_silu   depthwise causal conv against the ring, SiLU, ring advance
 *   dn_head        one block per value head:
 *                  b, a = the head's two rows of dn_b . x (b rows first, then
 *                  a rows); beta = sigmoid(b), g = -exp(A_log) *
 *                  softplus(a + dt_bias);
 *                  repeat-interleave q/k (value head h takes key head
 *                  h / rep), l2norm, q scaled by 1/sqrt(kdim);
 *                  decay, kv = k S, delta = (v - kv) beta, S += k delta^T,
 *                  o = q S;
 *                  the head's gated RMSNorm times silu(z)
 *
 *   outr  [value_dim]              the dnout GEMV's input
 *
 * Two launches per layer. The first version had three (a separate kernel
 * for a, b and the gates) and gave each value column of the state to one
 * thread: 32 blocks of 128 threads on the 35B, 29 us per layer for the
 * recurrence alone, and ~6-7 us for each small kernel, whose work is tiny:
 * most likely the cost of a launch on the reference Windows machine, not
 * measured then (docs/qwen36-deltanet-gpu-plan.md, stage 1, first run; the
 * test's bench now has an empty kernel for it). Now the gates live in dn_head, whose blocks are already one per
 * head, and each column's key rows are split across DN_SPLIT threads.
 *
 * The CPU reference is deltanet() in qwen36.c. Where the order is the CPU's:
 *   - the conv is the same loop per channel;
 *   - the l2norm sums, its sqrt and division, and the RMSNorm sum are double.
 * Where it is not: the a/b dot products, kv and o are sums in a fixed order
 * of partial sums (slices of the key rows, then the slices in order), and
 * the double sums are fixed-order trees; nvcc also contracts multiply-adds
 * into FMAs. Deterministic, not bit-identical. The tests
 * (tests/test_qwen36_deltanet_cuda.cu) bound the difference.
 * No atomics: the same inputs give the same bytes, run after run.
 *
 * Stage 1 has no engine or backend entry point: this header is included by
 * the test alone, so coli_cuda.dll does not change. Stage 2 includes it in
 * backend_cuda.cu behind coli_cuda_deltanet_decode(). */
#pragma once
#include "backend_gpu_compat.h"   /* the CUDA runtime, or HIP mapped onto it */
#include <math.h>
#include <stddef.h>

typedef struct {
    int hidden;     /* H */
    int vh, vk;     /* value heads, key heads (vh % vk == 0) */
    int kdim, vdim; /* key and value head dims */
    int convk;      /* conv kernel width (>= 2) */
    float eps;      /* the gated RMSNorm's eps (config rms_norm_eps) */
} DnShape;

static inline int dn_conv_dim(const DnShape *s)  { return 2 * s->vk * s->kdim + s->vh * s->vdim; }
static inline int dn_value_dim(const DnShape *s) { return s->vh * s->vdim; }

/* Smallest power of two >= n (n <= 1024 here). */
static inline __host__ __device__ int dn_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

/* Key-row slices per value column: up to 4, as many as fit 1024 threads,
 * never more than kdim. dn_head's block is vdim * dn_split threads. */
static inline int dn_split(const DnShape *s) {
    int p = s->vdim > 0 ? 1024 / s->vdim : 1;
    if (p > 4) p = 4;
    if (p > s->kdim) p = s->kdim;
    return p < 1 ? 1 : p;
}
static inline int dn_head_threads(const DnShape *s) { return s->vdim * dn_split(s); }

/* Dynamic shared memory of dn_head: q and k of one head, the partial sums
 * of the slices, then the double reduction scratch (8-byte aligned). */
static inline size_t dn_head_smem(const DnShape *s) {
    size_t f = ((size_t)2 * s->kdim + (size_t)dn_head_threads(s)) * sizeof(float);
    f = (f + 7) & ~(size_t)7;
    return f + (size_t)dn_pow2(dn_head_threads(s)) * sizeof(double);
}

/* NULL when the shape is supported, else why not. */
/* The plan's stage 1 named four kernels (dn_ab_gates, dn_conv_silu,
 * dn_l2norm_rep, dn_recurrence_norm); here they are two: dn_conv_silu, and
 * dn_head for the other three. */
static inline const char *dn_shape_check(const DnShape *s) {
    if (!s) return "no shape";
    if (s->hidden <= 0 || s->vh <= 0 || s->vk <= 0 || s->kdim <= 0 || s->vdim <= 0)
        return "a dimension is not positive";
    if (s->vh % s->vk) return "value heads are not a multiple of key heads";
    if (s->convk < 2) return "conv kernel narrower than 2";
    if (s->vdim > 1024) return "value head dim above 1024 (at least one thread per value column)";
    if (dn_head_smem(s) > 48 * 1024) return "q, k, the partial sums and the reduction scratch exceed 48 KiB of shared memory";
    return NULL;
}

/* Sum over the block, fixed order: every thread passes its value, every
 * thread gets the total. red holds pow2(blockDim.x) elements. Ends with a
 * barrier, so red can be reused by the next call. */
template <typename T>
__device__ static inline T dn_block_sum(T v, T *red) {
    const int P = dn_pow2(blockDim.x);
    for (int i = threadIdx.x; i < P; i += blockDim.x) red[i] = i == (int)threadIdx.x ? v : (T)0;
    __syncthreads();
    for (int stride = P >> 1; stride > 0; stride >>= 1) {
        for (int i = threadIdx.x; i < stride; i += blockDim.x) red[i] += red[i + stride];
        __syncthreads();
    }
    T total = red[0];
    __syncthreads();
    return total;
}

/* grid ceil(conv_dim / 256), block 256. ring: [conv_dim][convk-1], oldest
 * first, the host's layout; qkv: the raw projected channels. */
__global__ static void dn_conv_silu(const float *__restrict__ qkv, const float *__restrict__ wconv,
                                    float *__restrict__ ring, int conv_dim, int convk,
                                    float *__restrict__ conv_out) {
    const int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= conv_dim) return;
    const float *w = wconv + (size_t)cc * convk;
    float *rg = ring + (size_t)cc * (convk - 1);
    float acc = 0.f;
    for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
    const float cur = qkv[cc];
    acc += w[convk - 1] * cur;
    conv_out[cc] = acc / (1.f + expf(-acc));
    for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
    rg[convk - 2] = cur;
}

/* grid vh, block dn_head_threads() = vdim * split, dynamic shared
 * dn_head_smem(). Thread t is value column t % vdim, key-row slice
 * t / vdim, so a warp reads consecutive columns of one state row.
 * wab: [2*vh][H], b rows first; rec: [vh][kdim][vdim]; z: the z half of
 * qkvz; beta, g: [vh], written for the tests; outr: [vh][vdim]. */
__global__ static void dn_head(const float *__restrict__ x, const float *__restrict__ wab,
                               const float *__restrict__ alog, const float *__restrict__ dtbias,
                               const float *__restrict__ conv_out, const float *__restrict__ z,
                               const float *__restrict__ normw, float *__restrict__ rec,
                               int H, int vh, int rep, int kdim, int vdim, int split, int key_dim_tot,
                               float scale, float eps,
                               float *__restrict__ beta_out, float *__restrict__ g_out,
                               float *__restrict__ outr) {
    extern __shared__ unsigned char dn_smem[];
    const int nt = blockDim.x;
    float *qs = (float *)dn_smem, *ks = qs + kdim, *part = ks + kdim;
    double *red = (double *)(dn_smem + ((((size_t)2 * kdim + nt) * sizeof(float) + 7) & ~(size_t)7));
    float *redf = (float *)red;
    const int h = blockIdx.x, t = threadIdx.x, vv = t % vdim, sl = t / vdim;

    /* the gates: this head's b and a rows against x */
    const float *wb = wab + (size_t)h * H, *wa = wab + (size_t)(vh + h) * H;
    float sb = 0.f, sa = 0.f;
    for (int i = t; i < H; i += nt) { float xi = x[i]; sb += xi * wb[i]; sa += xi * wa[i]; }
    sb = dn_block_sum(sb, redf);
    sa = dn_block_sum(sa, redf);
    const float bh = 1.f / (1.f + expf(-sb));
    const float za = sa + dtbias[h];
    const float gh = -expf(alog[h]) * (za > 20.f ? za : log1pf(expf(za)));   /* softplus_f */
    if (t == 0) { beta_out[h] = bh; g_out[h] = gh; }

    /* per-head l2norm, eps 1e-6 inside the sqrt, in double */
    const int kh = h / rep;                       /* repeat_interleave: NOT h % vk */
    const float *q_in = conv_out + (size_t)kh * kdim;
    const float *k_in = conv_out + key_dim_tot + (size_t)kh * kdim;
    double pq = 0.0, pk = 0.0;
    for (int d = t; d < kdim; d += nt) { double a = q_in[d], b = k_in[d]; pq += a * a; pk += b * b; }
    pq = dn_block_sum(pq, red);
    pk = dn_block_sum(pk, red);
    const double nq = sqrt(1e-6 + pq), nk = sqrt(1e-6 + pk);
    for (int d = t; d < kdim; d += nt) {
        qs[d] = (float)((double)q_in[d] / nq * (double)scale);
        ks[d] = (float)((double)k_in[d] / nk);
    }
    __syncthreads();

    /* gated delta rule: column vv, key rows [k0, k1) of this head's state */
    const int chunk = (kdim + split - 1) / split;
    const int k0 = sl * chunk, k1 = k0 + chunk < kdim ? k0 + chunk : kdim;
    float *Sh = rec + (size_t)h * kdim * vdim;
    const float egh = expf(gh);
    float kvp = 0.f;
    for (int kk = k0; kk < k1; kk++) kvp += ks[kk] * (Sh[(size_t)kk * vdim + vv] * egh);
    part[t] = kvp;
    __syncthreads();
    float kv = 0.f;
    for (int j = 0; j < split; j++) kv += part[j * vdim + vv];
    const float v = conv_out[2 * key_dim_tot + (size_t)h * vdim + vv];
    const float dl = (v - kv) * bh;
    float op = 0.f;
    for (int kk = k0; kk < k1; kk++) {
        float s = Sh[(size_t)kk * vdim + vv] * egh;
        s += ks[kk] * dl;
        Sh[(size_t)kk * vdim + vv] = s;
        op += qs[kk] * s;
    }
    __syncthreads();                 /* every slice has read its kv from part */
    part[t] = op;
    __syncthreads();
    float o = 0.f;
    for (int j = 0; j < split; j++) o += part[j * vdim + vv];

    /* gated RMSNorm over the head (slice 0 holds the column), then silu(z) */
    const double ms = dn_block_sum(sl == 0 ? (double)o * o : 0.0, red);
    if (sl == 0) {
        const float r = 1.f / sqrtf((float)(ms / vdim) + eps);
        const float zz = z[(size_t)h * vdim + vv];
        outr[(size_t)h * vdim + vv] = (o * r * normw[vv]) * zz / (1.f + expf(-zz));
    }
}

/* Device pointers of one layer, for dn_decode_middle. */
typedef struct {
    const float *x;       /* [H] */
    const float *qkvz;    /* [conv_dim + value_dim] */
    const float *wab;     /* [2*vh][H] */
    const float *alog, *dtbias;   /* [vh] */
    const float *wconv;   /* [conv_dim][convk] */
    const float *normw;   /* [vdim] */
    float *ring;          /* [conv_dim][convk-1], updated */
    float *rec;           /* [vh][kdim][vdim], updated */
    float *beta, *g;      /* [vh], written for the tests */
    float *conv_out;      /* [conv_dim] scratch */
    float *outr;          /* [value_dim] output */
} DnLayerDev;

/* The two kernels in order on one stream. Returns the launch error (the
 * kernels run asynchronously; a fault shows at the next synchronisation). */
static inline cudaError_t dn_decode_middle(cudaStream_t st, const DnShape *s, const DnLayerDev *d) {
    const int conv_dim = dn_conv_dim(s), key_dim_tot = s->vk * s->kdim;
    dn_conv_silu<<<(conv_dim + 255) / 256, 256, 0, st>>>(d->qkvz, d->wconv, d->ring, conv_dim, s->convk, d->conv_out);
    dn_head<<<s->vh, dn_head_threads(s), dn_head_smem(s), st>>>(d->x, d->wab, d->alog, d->dtbias,
        d->conv_out, d->qkvz + conv_dim, d->normw, d->rec,
        s->hidden, s->vh, s->vh / s->vk, s->kdim, s->vdim, dn_split(s), key_dim_tot,
        1.f / sqrtf((float)s->kdim), s->eps, d->beta, d->g, d->outr);
    return cudaGetLastError();
}
