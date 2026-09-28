/* The DeltaNet decode kernels (backend_cuda_deltanet.cuh) against a CPU
 * reference copied from deltanet() in qwen36.c, on real silicon.
 * docs/qwen36-deltanet-gpu-plan.md, stage 1.
 *
 * An nvcc test cannot include qwen36.c, so dn_ref_step below is a copy of
 * deltanet()'s middle -- from the a/b matmul to the gated RMSNorm -- for one
 * token, compiled by nvcc's host compiler (part of what the tolerances cover).
 * Keep it in step with qwen36.c.
 *
 * For the 35B shapes (hidden 2048, 32 value heads, 16 key heads, head dims
 * 128, conv width 4), the CI tiny fixture's (hidden 64, 8 and 4 heads, head
 * dims 8), and three made-up shapes for the paths those two do not reach
 * (kdim > vdim, kdim above the block size, a vdim that is not a power of
 * two, above and below a warp, three value heads per key head):
 *   1. one step from a random state: beta and g within an ELEMENTWISE
 *      relative tolerance (a long-memory head has a g a thousand times
 *      smaller than the others, and g is the error that compounds),
 *      conv_out, outr and the state within tolerance of the largest value;
 *      the ring advanced byte for byte; head 0 has a dt_bias above 20, the
 *      softplus branch that returns its argument;
 *   2. the same step run twice from the same state, the outputs poisoned
 *      with NaN in between: the same bytes in every output and in the ring
 *      and the state (no atomics, fixed-order reductions);
 *   3. 512 steps from a zero state, each input fresh, both sides carrying
 *      their own state: every step's outr and the final state within a
 *      bound, and the error of the last 128 steps no more than twice that of
 *      steps 128-255, on the mean error per window (the decay makes the
 *      recurrence contractive; an error that grows means it is not carried
 *      the same way). An emulation of the kernels' arithmetic on the host
 *      (not the device expf) gave errors of 1e-7 to 2e-6, and so did the
 *      first run on an RTX 4070 Ti SUPER (1e-7 to 2.5e-6, three kernels,
 *      one thread per state column): the bounds, 2e-6 to 2e-5, leave a
 *      margin of 4 to 20 over those.
 * The errors are printed, so the tolerances can be tightened with data.
 *
 * --bench: the 35B shape, 30 layers of state (60 MiB, more than the 48 MB
 * L2 of the reference RTX 4070 Ti SUPER; a card with a larger L2 holds it),
 * per-layer time of the two kernels back to back and with a synchronize
 * after every layer, each kernel alone back to back (for dn_conv_silu that
 * is launch throughput as much as kernel time), and an empty kernel, the
 * launch rate by itself.
 *
 * Build: nvcc -O3 -std=c++17 -arch=native tests/test_qwen36_deltanet_cuda.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../backend_cuda_deltanet.cuh"

static int g_fail = 0;
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

/* ---- the CPU reference: deltanet() in qwen36.c, one token --------------- */
static float softplus_f(float z) { return z > 20.f ? z : log1pf(expf(z)); }

static void dn_ref_step(const DnShape *s, const float *x, const float *qkvz, const float *wab,
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
static float urand(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float)((g_rng >> 11) * (1.0 / 9007199254740992.0));
}
static void fill(std::vector<float> &v, float lo, float hi) { for (auto &e : v) e = urand(lo, hi); }

/* Weights and per-token inputs: x in the range of an RMS-normed vector,
 * A = exp(A_log) in 1..16, and dt_bias such that softplus(dt_bias) spans
 * 0.001..0.1 (a Mamba2-style dt range, chosen here, not read from the
 * checkpoint). With a of std ~0.6 that puts exp(g) between about 0.01 and
 * 1, most heads near 1: long memory, the harder case for an error carried in
 * the state. Head 0 gets dt_bias 25, above softplus's threshold of 20. */
struct Layer {
    std::vector<float> wab, alog, dtbias, wconv, normw;
};
static Layer make_layer(const DnShape *s) {
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
static void make_token(const DnShape *s, std::vector<float> &x, std::vector<float> &qkvz) {
    x.resize(s->hidden); fill(x, -1.7f, 1.7f);
    qkvz.resize((size_t)dn_conv_dim(s) + dn_value_dim(s)); fill(qkvz, -2.f, 2.f);
}

/* ---- device buffers -------------------------------------------------------- */
struct Dev {
    float *x, *qkvz, *wab, *alog, *dtbias, *wconv, *normw, *ring, *rec, *beta, *g, *conv_out, *outr;
};
static float *dalloc(size_t n) { float *p; CK(cudaMalloc(&p, n * sizeof(float))); return p; }
static void up(float *d, const std::vector<float> &h) { CK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice)); }
static void down(std::vector<float> &h, const float *d, size_t n) { h.resize(n); CK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost)); }
static Dev dev_alloc(const DnShape *s) {
    const size_t conv_dim = dn_conv_dim(s), value_dim = dn_value_dim(s);
    Dev d;
    d.x = dalloc(s->hidden); d.qkvz = dalloc(conv_dim + value_dim);
    d.wab = dalloc((size_t)2 * s->vh * s->hidden); d.alog = dalloc(s->vh); d.dtbias = dalloc(s->vh);
    d.wconv = dalloc(conv_dim * s->convk); d.normw = dalloc(s->vdim);
    d.ring = dalloc(conv_dim * (s->convk - 1)); d.rec = dalloc((size_t)s->vh * s->kdim * s->vdim);
    d.beta = dalloc(s->vh); d.g = dalloc(s->vh); d.conv_out = dalloc(conv_dim); d.outr = dalloc(value_dim);
    return d;
}
static void dev_free(Dev &d) {
    float *all[] = {d.x, d.qkvz, d.wab, d.alog, d.dtbias, d.wconv, d.normw, d.ring, d.rec, d.beta, d.g, d.conv_out, d.outr};
    for (float *p : all) cudaFree(p);
}
static DnLayerDev view(const Dev &d) {
    DnLayerDev v;
    v.x = d.x; v.qkvz = d.qkvz; v.wab = d.wab; v.alog = d.alog; v.dtbias = d.dtbias; v.wconv = d.wconv;
    v.normw = d.normw; v.ring = d.ring; v.rec = d.rec; v.beta = d.beta; v.g = d.g; v.conv_out = d.conv_out; v.outr = d.outr;
    return v;
}

/* max |got - ref| over max |ref| (with a floor of 1e-30); the scale-free error */
static double rel_err(const std::vector<float> &got, const std::vector<float> &ref) {
    double num = 0, den = 1e-30;
    for (size_t i = 0; i < ref.size(); i++) {
        num = std::max(num, (double)fabsf(got[i] - ref[i]));
        den = std::max(den, (double)fabsf(ref[i]));
    }
    return num / den;
}
/* max over elements of |got - ref| / max(|ref|, floor): every element
 * relative to itself, for the per-head gates */
static double rel_err_elem(const std::vector<float> &got, const std::vector<float> &ref, double floor) {
    double worst = 0;
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, (double)fabsf(got[i] - ref[i]) / std::max((double)fabsf(ref[i]), floor));
    return worst;
}
static void check(bool ok, const char *what, double v, double tol) {
    printf("  %-44s %.3e  (tol %.0e)  %s\n", what, v, tol, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

static void run_shape(const char *name, DnShape s) {
    printf("%s: hidden %d, heads %d/%d, dims %d/%d, conv %d x %d\n",
           name, s.hidden, s.vh, s.vk, s.kdim, s.vdim, dn_conv_dim(&s), s.convk);
    const char *why = dn_shape_check(&s);
    if (why) { printf("  shape refused: %s  FAIL\n", why); g_fail++; return; }
    const size_t conv_dim = dn_conv_dim(&s), value_dim = dn_value_dim(&s);
    const size_t nring = conv_dim * (s.convk - 1), nrec = (size_t)s.vh * s.kdim * s.vdim;
    Layer L = make_layer(&s);
    Dev d = dev_alloc(&s);
    up(d.wab, L.wab); up(d.alog, L.alog); up(d.dtbias, L.dtbias); up(d.wconv, L.wconv); up(d.normw, L.normw);
    DnLayerDev v = view(d);

    /* 1. one step from a random state */
    std::vector<float> x, qkvz, ring(nring), rec(nrec);
    make_token(&s, x, qkvz); fill(ring, -2.f, 2.f); fill(rec, -0.5f, 0.5f);
    std::vector<float> ring0 = ring, rec0 = rec;
    std::vector<float> beta(s.vh), g(s.vh), conv_out(conv_dim), outr(value_dim);
    dn_ref_step(&s, x.data(), qkvz.data(), L.wab.data(), L.alog.data(), L.dtbias.data(), L.wconv.data(),
                L.normw.data(), ring.data(), rec.data(), beta.data(), g.data(), conv_out.data(), outr.data());
    up(d.x, x); up(d.qkvz, qkvz); up(d.ring, ring0); up(d.rec, rec0);
    CK(dn_decode_middle(0, &s, &v)); CK(cudaDeviceSynchronize());
    std::vector<float> gb, gg, gc, go, gring, grec;
    down(gb, d.beta, s.vh); down(gg, d.g, s.vh); down(gc, d.conv_out, conv_dim); down(go, d.outr, value_dim);
    down(gring, d.ring, nring); down(grec, d.rec, nrec);
    check(rel_err_elem(gb, beta, 1e-30) < 1e-5, "one step: beta, per element", rel_err_elem(gb, beta, 1e-30), 1e-5);
    check(rel_err_elem(gg, g, 1e-30) < 1e-5, "one step: g, per element", rel_err_elem(gg, g, 1e-30), 1e-5);
    check(rel_err(gc, conv_out) < 2e-6, "one step: conv_out", rel_err(gc, conv_out), 2e-6);
    check(rel_err(go, outr) < 1e-5, "one step: outr", rel_err(go, outr), 1e-5);
    check(rel_err(grec, rec) < 1e-5, "one step: state", rel_err(grec, rec), 1e-5);
    bool ring_same = memcmp(gring.data(), ring.data(), nring * sizeof(float)) == 0;
    check(ring_same, "one step: ring, byte for byte (0 = same)", ring_same ? 0.0 : 1.0, 0.0);

    /* 2. the same step again from the same state, outputs poisoned: the same bytes */
    up(d.ring, ring0); up(d.rec, rec0);
    CK(cudaMemset(d.beta, 0xFF, s.vh * sizeof(float))); CK(cudaMemset(d.g, 0xFF, s.vh * sizeof(float)));
    CK(cudaMemset(d.conv_out, 0xFF, conv_dim * sizeof(float))); CK(cudaMemset(d.outr, 0xFF, value_dim * sizeof(float)));
    CK(dn_decode_middle(0, &s, &v)); CK(cudaDeviceSynchronize());
    std::vector<float> gb2, gg2, gc2, go2, gring2, grec2;
    down(gb2, d.beta, s.vh); down(gg2, d.g, s.vh); down(gc2, d.conv_out, conv_dim); down(go2, d.outr, value_dim);
    down(gring2, d.ring, nring); down(grec2, d.rec, nrec);
    bool det = gb2 == gb && gg2 == gg && gc2 == gc && go2 == go && gring2 == gring && grec2 == grec;
    check(det, "repeat: every output, ring, state byte for byte (0 = same)", det ? 0.0 : 1.0, 0.0);

    /* 3. 512 steps from a zero state */
    std::fill(ring.begin(), ring.end(), 0.f); std::fill(rec.begin(), rec.end(), 0.f);
    up(d.ring, ring); up(d.rec, rec);
    double worst = 0, mid = 0, last = 0, mid_sum = 0, last_sum = 0;
    for (int t = 0; t < 512; t++) {
        make_token(&s, x, qkvz);
        dn_ref_step(&s, x.data(), qkvz.data(), L.wab.data(), L.alog.data(), L.dtbias.data(), L.wconv.data(),
                    L.normw.data(), ring.data(), rec.data(), beta.data(), g.data(), conv_out.data(), outr.data());
        up(d.x, x); up(d.qkvz, qkvz);
        CK(dn_decode_middle(0, &s, &v)); CK(cudaDeviceSynchronize());
        down(go, d.outr, value_dim);
        double e = rel_err(go, outr);
        worst = std::max(worst, e);
        if (t >= 128 && t < 256) { mid = std::max(mid, e); mid_sum += e; }
        if (t >= 384) { last = std::max(last, e); last_sum += e; }
    }
    down(grec, d.rec, nrec);
    check(worst < 2e-5, "512 steps: worst outr", worst, 2e-5);
    check(rel_err(grec, rec) < 2e-5, "512 steps: final state", rel_err(grec, rec), 2e-5);
    const double mid_mean = mid_sum / 128, last_mean = last_sum / 128;
    printf("  512 steps: outr error max / mean, steps 128-255 %.3e / %.3e, steps 384-511 %.3e / %.3e\n",
           mid, mid_mean, last, last_mean);
    const double grow = 2.0 * std::max(mid_mean, 1e-8);
    check(last_mean <= grow, "512 steps: mean of 384-511 <= 2 x mean of 128-255", last_mean, grow);
    dev_free(d);
}

/* ---- bench ----------------------------------------------------------------- */
/* The launch rate alone, for the per-kernel lines to be read against. */
__global__ static void dn_bench_empty(void) {}
static void bench(void) {
    DnShape s = {2048, 32, 16, 128, 128, 4, 1e-6f};
    const int NL = 30, TOK = 200;
    printf("bench: 35B shape, %d layers of state, %d tokens\n", NL, TOK);
    std::vector<Dev> ds(NL);
    std::vector<DnLayerDev> vs(NL);
    std::vector<float> x, qkvz, ring((size_t)dn_conv_dim(&s) * (s.convk - 1), 0.f), rec((size_t)s.vh * s.kdim * s.vdim, 0.f);
    for (int l = 0; l < NL; l++) {
        Layer L = make_layer(&s);
        ds[l] = dev_alloc(&s);
        up(ds[l].wab, L.wab); up(ds[l].alog, L.alog); up(ds[l].dtbias, L.dtbias);
        up(ds[l].wconv, L.wconv); up(ds[l].normw, L.normw); up(ds[l].ring, ring); up(ds[l].rec, rec);
        make_token(&s, x, qkvz); up(ds[l].x, x); up(ds[l].qkvz, qkvz);
        vs[l] = view(ds[l]);
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    /* warm-up: as long as a timed pass, so the clocks have ramped */
    for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++) CK(dn_decode_middle(0, &s, &vs[l]));
    CK(cudaDeviceSynchronize());
    float ms = 0;
    CK(cudaEventRecord(e0));
    for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++) CK(dn_decode_middle(0, &s, &vs[l]));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("  back to back:          %7.2f us per layer, %.3f ms per token (30 layers)\n",
           1000.0 * ms / (TOK * NL), ms / TOK);
    CK(cudaEventRecord(e0));
    for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++) { CK(dn_decode_middle(0, &s, &vs[l])); CK(cudaStreamSynchronize(0)); }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("  synchronize per layer: %7.2f us per layer, %.3f ms per token (30 layers)\n",
           1000.0 * ms / (TOK * NL), ms / TOK);
    const int conv_dim = dn_conv_dim(&s), key_dim_tot = s.vk * s.kdim;
    for (int which = 0; which < 3; which++) {
        CK(cudaEventRecord(e0));
        for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++) {
            const DnLayerDev &d = vs[l];
            if (which == 0) dn_conv_silu<<<(conv_dim + 255) / 256, 256>>>(d.qkvz, d.wconv, d.ring, conv_dim, s.convk, d.conv_out);
            else if (which == 1) dn_head<<<s.vh, dn_head_threads(&s), dn_head_smem(&s)>>>(d.x, d.wab, d.alog, d.dtbias,
                     d.conv_out, d.qkvz + conv_dim, d.normw, d.rec, s.hidden, s.vh, s.vh / s.vk, s.kdim, s.vdim,
                     dn_split(&s), key_dim_tot, 1.f / sqrtf((float)s.kdim), s.eps, d.beta, d.g, d.outr);
            else dn_bench_empty<<<1, 32>>>();
        }
        CK(cudaGetLastError());
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
        static const char *nm[3] = {"dn_conv_silu", "dn_head", "empty kernel"};
        printf("  %-22s %7.2f us per layer, back to back (kernel or launch rate, whichever is slower)\n", nm[which], 1000.0 * ms / (TOK * NL));
    }
    for (auto &d : ds) dev_free(d);
}

int main(int argc, char **argv) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) { fprintf(stderr, "no CUDA device\n"); return 2; }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s, sm_%d%d\n", p.name, p.major, p.minor);
    if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(); return 0; }
    run_shape("35B", DnShape{2048, 32, 16, 128, 128, 4, 1e-6f});
    run_shape("tiny", DnShape{64, 8, 4, 8, 8, 4, 1e-6f});
    run_shape("kdim > vdim, vdim 24", DnShape{96, 6, 3, 40, 24, 4, 1e-6f});
    run_shape("vdim 48", DnShape{128, 4, 2, 16, 48, 3, 1e-6f});
    run_shape("rep 3, kdim above the block", DnShape{96, 6, 2, 160, 96, 4, 1e-6f});
    printf("deltanet cuda: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}
