/* The CPU reference for the Qwen3.6 DeltaNet decode kernels, shared by
 * tests/test_qwen36_deltanet_cuda.cu (the kernels) and
 * tests/test_qwen36_deltanet_decode_cuda.cu (the backend entry point).
 *
 * dn_ref_step is a copy of deltanet()'s middle in qwen36.c -- from the a/b
 * matmul to the gated RMSNorm -- for one token: an nvcc test cannot include
 * qwen36.c. Keep it in step with qwen36.c. Also the test data: weights and
 * per-token inputs in the ranges described at make_layer. */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../backend_cuda_deltanet.cuh"

/* ---- the CPU reference: deltanet() in qwen36.c, one token --------------- */
static inline float softplus_f(float z) { return z > 20.f ? z : log1pf(expf(z)); }

static inline void dn_ref_step(const DnShape *s, const float *x, const float *qkvz, const float *wab,
                        const float *alog, const float *dtbias, const float *wconv, const float *normw,
                        float *ring, float *rec,
                        float *beta, float *gg, float *conv_out, float *outr) {
    const int vh = s->vh, vk = s->vk, kdim = s->kdim, vdim = s->vdim, convk = s->convk, H = s->hidden;
    const int conv_dim = dn_conv_dim(s), rep = vh / vk, key_dim_tot = vk * kdim;
    const float scale = 1.f / sqrtf((float)kdim);
    const float *qkv = qkvz, *z = qkvz + conv_dim;
    std::vector<float> ba(2 * vh), q((size_t)vh * kdim), k((size_t)vh * kdim), outv((size_t)vh * vdim);
    /* matmul(b, xs, l->dn_b, 1, H, 2*vh) */
    for (int o = 0; o < 2 * vh; o++) {
        const float *w = wab + (size_t)o * H; float acc = 0.f;
        for (int i = 0; i < H; i++) acc += x[i] * w[i];
        ba[o] = acc;
    }
    const float *b = ba.data(), *a = ba.data() + vh;
    for (int h = 0; h < vh; h++) {
        beta[h] = 1.f / (1.f + expf(-b[h]));
        gg[h] = -expf(alog[h]) * softplus_f(a[h] + dtbias[h]);
    }
    for (int cc = 0; cc < conv_dim; cc++) {
        const float *w = wconv + (size_t)cc * convk;
        float *rg = ring + (size_t)cc * (convk - 1);
        float acc = 0.f;
        for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
        acc += w[convk - 1] * qkv[cc];
        conv_out[cc] = acc / (1.f + expf(-acc));
        for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
        rg[convk - 2] = qkv[cc];
    }
    const float *q_in = conv_out, *k_in = conv_out + key_dim_tot, *v_in = conv_out + 2 * key_dim_tot;
    for (int h = 0; h < vh; h++) {
        int vk_idx = h / rep;
        memcpy(&q[(size_t)h * kdim], q_in + (size_t)vk_idx * kdim, kdim * sizeof(float));
        memcpy(&k[(size_t)h * kdim], k_in + (size_t)vk_idx * kdim, kdim * sizeof(float));
    }
    for (int oh = 0; oh < vh; oh++) {
        float *qh = &q[(size_t)oh * kdim];
        double sq = 1e-6; for (int d = 0; d < kdim; d++) sq += (double)qh[d] * qh[d];
        double nq = sqrt(sq);
        for (int d = 0; d < kdim; d++) qh[d] = (float)((double)qh[d] / nq * scale);
        float *kh = &k[(size_t)oh * kdim];
        double sk = 1e-6; for (int d = 0; d < kdim; d++) sk += (double)kh[d] * kh[d];
        double nk = sqrt(sk);
        for (int d = 0; d < kdim; d++) kh[d] = (float)((double)kh[d] / nk);
    }
    std::vector<float> kvl(vdim), dl(vdim);
    for (int h = 0; h < vh; h++) {
        float *Sh = rec + (size_t)h * kdim * vdim;
        float egh = expf(gg[h]);
        for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;
        const float *kd = &k[(size_t)h * kdim];
        const float *vd = v_in + (size_t)h * vdim;
        for (int vv = 0; vv < vdim; vv++) kvl[vv] = 0.f;
        for (int kk = 0; kk < kdim; kk++) {
            float kkd = kd[kk]; const float *Sr = Sh + (size_t)kk * vdim;
            for (int vv = 0; vv < vdim; vv++) kvl[vv] += kkd * Sr[vv];
        }
        for (int vv = 0; vv < vdim; vv++) dl[vv] = (vd[vv] - kvl[vv]) * beta[h];
        for (int kk = 0; kk < kdim; kk++) {
            float kkd = kd[kk]; float *Sr = Sh + (size_t)kk * vdim;
            for (int vv = 0; vv < vdim; vv++) Sr[vv] += kkd * dl[vv];
        }
        const float *qd = &q[(size_t)h * kdim];
        float *ov = &outv[(size_t)h * vdim];
        for (int vv = 0; vv < vdim; vv++) ov[vv] = 0.f;
        for (int kk = 0; kk < kdim; kk++) {
            float qkd = qd[kk]; const float *Sr = Sh + (size_t)kk * vdim;
            for (int vv = 0; vv < vdim; vv++) ov[vv] += qkd * Sr[vv];
        }
    }
    for (int h = 0; h < vh; h++) {
        const float *o = &outv[(size_t)h * vdim];
        const float *zr = z + (size_t)h * vdim;
        double ms = 0; for (int d = 0; d < vdim; d++) ms += (double)o[d] * o[d];
        float r = 1.f / sqrtf((float)(ms / vdim) + s->eps);
        for (int d = 0; d < vdim; d++) {
            float val = o[d] * r * normw[d];
            outr[(size_t)h * vdim + d] = val * zr[d] / (1.f + expf(-zr[d]));
        }
    }
}

/* ---- data ------------------------------------------------------------------ */
static unsigned long long g_rng = 0x9E3779B97F4A7C15ull;
static inline float urand(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float)((g_rng >> 11) * (1.0 / 9007199254740992.0));
}
static inline void fill(std::vector<float> &v, float lo, float hi) { for (auto &e : v) e = urand(lo, hi); }

/* Weights and per-token inputs: x in the range of an RMS-normed vector,
 * A = exp(A_log) in 1..16, and dt_bias such that softplus(dt_bias) spans
 * 0.001..0.1 (a Mamba2-style dt range, chosen here, not read from the
 * checkpoint). With a of std ~0.6 that puts exp(g) between about 0.01 and
 * 1, most heads near 1: long memory, the harder case for an error carried in
 * the state. Head 0 gets dt_bias 25, above softplus's threshold of 20. */
struct Layer {
    std::vector<float> wab, alog, dtbias, wconv, normw;
};
static inline Layer make_layer(const DnShape *s) {
    Layer L;
    const int conv_dim = dn_conv_dim(s);
    L.wab.resize((size_t)2 * s->vh * s->hidden); fill(L.wab, -1.f, 1.f);
    const float ws = 1.f / sqrtf((float)s->hidden);
    for (auto &e : L.wab) e *= ws;
    L.alog.resize(s->vh); for (auto &e : L.alog) e = logf(urand(1.f, 16.f));
    L.dtbias.resize(s->vh); for (auto &e : L.dtbias) { float dt = expf(urand(logf(0.001f), logf(0.1f))); e = logf(expm1f(dt)); }
    L.dtbias[0] = 25.f;
    L.wconv.resize((size_t)conv_dim * s->convk); fill(L.wconv, -0.6f, 0.6f);
    L.normw.resize(s->vdim); fill(L.normw, 0.5f, 1.5f);
    return L;
}
static inline void make_token(const DnShape *s, std::vector<float> &x, std::vector<float> &qkvz) {
    x.resize(s->hidden); fill(x, -1.7f, 1.7f);
    qkvz.resize((size_t)dn_conv_dim(s) + dn_value_dim(s)); fill(qkvz, -2.f, 2.f);
}

/* max |got - ref| over max |ref| (with a floor of 1e-30); the scale-free error */
static inline double rel_err(const std::vector<float> &got, const std::vector<float> &ref) {
    double num = 0, den = 1e-30;
    for (size_t i = 0; i < ref.size(); i++) {
        num = std::max(num, (double)fabsf(got[i] - ref[i]));
        den = std::max(den, (double)fabsf(ref[i]));
    }
    return num / den;
}
