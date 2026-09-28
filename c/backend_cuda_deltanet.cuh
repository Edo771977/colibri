/* Qwen3.6 DeltaNet decode on the GPU: the kernels between the two placed
 * GEMVs (docs/qwen36-deltanet-gpu-plan.md, stage 1).
 *
 * One decode step of one DeltaNet layer, S = 1, everything already on the
 * device:
 *
 *   qkvz  [conv_dim + value_dim]   the dnproj GEMV's output (qkv ++ z)
 *   x     [hidden]                 the layer's normed input (for a and b)
 *
 *   dn_ab_gates    b, a = rows of dn_b . x (b rows first, then a rows);
 *                  beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
 *   dn_conv_silu   depthwise causal conv against the ring, SiLU, ring advance
 *   dn_rec_norm    per value head: repeat-interleave q/k (value head h takes
 *                  key head h / rep), l2norm, q scaled by 1/sqrt(kdim), decay,
 *                  kv = k S, delta = (v - kv) beta, S += k delta^T, o = q S,
 *                  then that head's gated RMSNorm times silu(z)
 *
 *   outr  [value_dim]              the dnout GEMV's input
 *
 * The CPU reference is deltanet() in qwen36.c; the operations and their
 * order follow it where that is cheap:
 *   - the conv is the same loop per channel;
 *   - the recurrence gives each value column to one thread, which walks the
 *     key rows in the CPU's order, so kv and o are the same sequential sums;
 *   - the l2norm sums, its sqrt and division, and the RMSNorm sum are double;
 *   - the reductions that the CPU does sequentially (the a/b dot products,
 *     the three double sums) are fixed-order trees here: deterministic, not
 *     bit-identical; nvcc also contracts multiply-adds into FMAs. The tests
 *     (tests/test_qwen36_deltanet_cuda.cu) bound the difference.
 * No atomics: the same inputs give the same bytes, run after run.
 *
 * The plan listed four kernels; the l2norm and the repeat-interleave are
 * folded into dn_rec_norm, which is the only reader of q and k: one launch
 * less per layer, no q/k buffers.
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

/* Dynamic shared memory of dn_rec_norm: q and k of one head, then the double
 * reduction scratch (8-byte aligned). */
static inline size_t dn_rec_smem(const DnShape *s) {
    size_t f = (size_t)2 * s->kdim * sizeof(float);
    f = (f + 7) & ~(size_t)7;
    return f + (size_t)dn_pow2(s->vdim) * sizeof(double);
}

/* NULL when the shape is supported, else why not. */
static inline const char *dn_shape_check(const DnShape *s) {
    if (!s) return "no shape";
    if (s->hidden <= 0 || s->vh <= 0 || s->vk <= 0 || s->kdim <= 0 || s->vdim <= 0)
        return "a dimension is not positive";
    if (s->vh % s->vk) return "value heads are not a multiple of key heads";
    if (s->convk < 2) return "conv kernel narrower than 2";
    if (s->vdim > 1024) return "value head dim above 1024 (one thread per value column)";
    if (dn_rec_smem(s) > 48 * 1024) return "q, k and the reduction scratch exceed 48 KiB of shared memory";
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

/* grid vh, block 256 (a power of two <= 1024). wab: [2*vh][H], b rows first. */
__global__ static void dn_ab_gates(const float *__restrict__ x, const float *__restrict__ wab,
                                   const float *__restrict__ alog, const float *__restrict__ dtbias,
                                   int H, int vh, float *__restrict__ beta, float *__restrict__ g) {
    __shared__ float red[1024];
    const int h = blockIdx.x;
    const float *wb = wab + (size_t)h * H, *wa = wab + (size_t)(vh + h) * H;
    float sb = 0.f, sa = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { float xi = x[i]; sb += xi * wb[i]; sa += xi * wa[i]; }
    sb = dn_block_sum(sb, red);
    sa = dn_block_sum(sa, red);
    if (threadIdx.x == 0) {
        beta[h] = 1.f / (1.f + expf(-sb));
        float z = sa + dtbias[h];
        float sp = z > 20.f ? z : log1pf(expf(z));    /* softplus_f */
        g[h] = -expf(alog[h]) * sp;
    }
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

/* grid vh, block vdim, dynamic shared dn_rec_smem(). rec: [vh][kdim][vdim];
 * z: the z half of qkvz; outr: [vh][vdim]. */
__global__ static void dn_rec_norm(const float *__restrict__ conv_out, const float *__restrict__ z,
                                   const float *__restrict__ beta, const float *__restrict__ g,
                                   const float *__restrict__ normw, float *__restrict__ rec,
                                   int rep, int kdim, int vdim, int key_dim_tot,
                                   float scale, float eps, float *__restrict__ outr) {
    extern __shared__ unsigned char dn_smem[];
    float *qs = (float *)dn_smem, *ks = qs + kdim;
    double *red = (double *)(dn_smem + (((size_t)2 * kdim * sizeof(float) + 7) & ~(size_t)7));
    const int h = blockIdx.x, vv = threadIdx.x;
    const int kh = h / rep;                       /* repeat_interleave: NOT h % vk */
    const float *q_in = conv_out + (size_t)kh * kdim;
    const float *k_in = conv_out + key_dim_tot + (size_t)kh * kdim;

    /* per-head l2norm, eps 1e-6 inside the sqrt, in double */
    double pq = 0.0, pk = 0.0;
    for (int d = vv; d < kdim; d += blockDim.x) {
        double a = q_in[d], b = k_in[d]; pq += a * a; pk += b * b;
    }
    pq = dn_block_sum(pq, red);
    pk = dn_block_sum(pk, red);
    const double nq = sqrt(1e-6 + pq), nk = sqrt(1e-6 + pk);
    for (int d = vv; d < kdim; d += blockDim.x) {
        qs[d] = (float)((double)q_in[d] / nq * (double)scale);
        ks[d] = (float)((double)k_in[d] / nk);
    }
    __syncthreads();

    /* gated delta rule, column vv of this head's state */
    float *Sh = rec + (size_t)h * kdim * vdim;
    const float egh = expf(g[h]);
    const float v = conv_out[2 * key_dim_tot + (size_t)h * vdim + vv];
    float kv = 0.f;
    for (int kk = 0; kk < kdim; kk++) kv += ks[kk] * (Sh[(size_t)kk * vdim + vv] * egh);
    const float dl = (v - kv) * beta[h];
    float o = 0.f;
    for (int kk = 0; kk < kdim; kk++) {
        float s = Sh[(size_t)kk * vdim + vv] * egh;
        s += ks[kk] * dl;
        Sh[(size_t)kk * vdim + vv] = s;
        o += qs[kk] * s;
    }

    /* gated RMSNorm over the head, then silu(z) */
    const double ms = dn_block_sum((double)o * o, red);
    const float r = 1.f / sqrtf((float)(ms / vdim) + eps);
    const float zz = z[(size_t)h * vdim + vv];
    outr[(size_t)h * vdim + vv] = (o * r * normw[vv]) * zz / (1.f + expf(-zz));
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
    float *beta, *g;      /* [vh] scratch */
    float *conv_out;      /* [conv_dim] scratch */
    float *outr;          /* [value_dim] output */
} DnLayerDev;

/* The three kernels in order on one stream. Returns the launch error (the
 * kernels run asynchronously; a fault shows at the next synchronisation). */
static inline cudaError_t dn_decode_middle(cudaStream_t st, const DnShape *s, const DnLayerDev *d) {
    const int conv_dim = dn_conv_dim(s), key_dim_tot = s->vk * s->kdim;
    dn_ab_gates<<<s->vh, 256, 0, st>>>(d->x, d->wab, d->alog, d->dtbias, s->hidden, s->vh, d->beta, d->g);
    dn_conv_silu<<<(conv_dim + 255) / 256, 256, 0, st>>>(d->qkvz, d->wconv, d->ring, conv_dim, s->convk, d->conv_out);
    dn_rec_norm<<<s->vh, s->vdim, dn_rec_smem(s), st>>>(d->conv_out, d->qkvz + conv_dim, d->beta, d->g,
        d->normw, d->rec, s->vh / s->vk, s->kdim, s->vdim, key_dim_tot,
        1.f / sqrtf((float)s->kdim), s->eps, d->outr);
    return cudaGetLastError();
}
