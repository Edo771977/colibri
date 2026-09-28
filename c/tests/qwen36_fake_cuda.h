/* qwen36_fake_cuda.h -- fake CUDA backend shared by the qwen36 tier tests.
 *
 * Defines every coli_cuda_* symbol qwen36_tier.c links against (its
 * signatures come from backend_cuda.h, which the tier includes on its own)
 * and RECORDS what it receives, so a test can assert on real upload/issue
 * traffic without a GPU or the CUDA toolkit. A test that only checked
 * "qt_init returns 1" would pass even with the tier fully broken.
 *
 * Three settable hooks beyond plain recording:
 *   fake_ndev        - device count returned by coli_cuda_available_device_count
 *                       and coli_cuda_device_count (default 1).
 *   fake_issue_hook   - called by coli_cuda_expert_group_issue with the issuing
 *                       device (taken from g[0]->device), the row count and the
 *                       input pointer; its return value is what issue returns.
 *                       NULL (the default) reproduces the old always-0 stub.
 *   fake_upload_hook  - called at the start of every tensor upload, on the
 *                       uploader thread, with the tensor's fmt. A test that
 *                       needs an upload to take TIME (a real cudaMemcpy does)
 *                       sleeps here; NULL (the default) uploads instantly. */
#ifndef QWEN36_FAKE_CUDA_H
#define QWEN36_FAKE_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../backend_cuda.h"

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; const float *sc; };

static int fake_uploads;
static int last_fmt = -1;
static size_t last_bytes;
static unsigned char captured[4096];
static size_t captured_len;

static int fake_ndev = 1;
static size_t fake_free_bytes = 2ull << 30;    /* what coli_cuda_mem_info reports as free */
static int (*fake_issue_hook)(int device, int count, const float *x) = NULL;
static void (*fake_upload_hook)(int fmt) = NULL;

/* fake_dense_compute=1: coli_cuda_matmul really computes fmt 1 (int8 per
 * row) from the uploaded bytes, so an engine test can put a trunk on the fake
 * tier and demand the same tokens as the CPU int8 reference. The engine
 * frees its int8 rows right after qt_dense_init, so the upload keeps a copy. */
static int fake_dense_compute;
static int upload_common(ColiCudaTensor **t, const void *w, const float *sc, int fmt,
                         int I, int O, int device, int gs) {
    if (fake_upload_hook) fake_upload_hook(fmt);
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w; n->sc = sc;
    if (fake_dense_compute && fmt == 1 && w && sc) {
        size_t ng = gs > 0 ? ((size_t)I + gs - 1) / gs : 1;   /* grouped int8: [O][ng] scales */
        int8_t *q = (int8_t *)malloc((size_t)I * O); float *s = (float *)malloc((size_t)O * ng * sizeof(float));
        if (q && s) { memcpy(q, w, (size_t)I * O); memcpy(s, sc, (size_t)O * ng * sizeof(float)); n->w = q; n->sc = s; }
        else { free(q); free(s); n->w = NULL; n->sc = NULL; }
    }
    *t = n;
    fake_uploads++;
    last_fmt = fmt;
    last_bytes = (size_t)I * O / ((fmt == 1 || fmt == 8) ? 1 : 2);
    if (fake_uploads == 1) {
        captured_len = last_bytes < sizeof captured ? last_bytes : sizeof captured;
        memcpy(captured, w, captured_len);
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    return upload_common(t, w, s, fmt, I, O, device, 0);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                              int fmt, int I, int O, int device, int gs) {
    return upload_common(t, w, s, fmt, I, O, device, gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if (t && fake_dense_compute && t->fmt == 1) { free((void *)t->w); free((void *)t->sc); }
    free(t);
}
int coli_cuda_available_device_count(void) { return fake_ndev; }
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
static int fake_lut_published;
int coli_cuda_fp8_set_lut(const float *lut) { fake_lut_published = lut != NULL; return lut != NULL; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = fake_free_bytes; *total = 4ull << 30;   /* 2 GiB liberi by default */
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    (void)u; (void)d; (void)rows;
    if (fake_issue_hook) return fake_issue_hook(count > 0 ? g[0]->device : -1, count, x);
    return 0;
}
/* The fake backend answers "no broadcast", so qt_issue exercises the
 * duplicating path here -- which is the path a pre-#1602 DLL takes, and the
 * one that would otherwise go untested. */
int coli_cuda_expert_group_issue_x(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                   ColiCudaTensor *const *d, const int *rows,
                                   int count, const float *x, int x_rows) {
    (void)g; (void)u; (void)d; (void)rows; (void)count; (void)x; (void)x_rows;
    return 0;
}
int coli_cuda_has_group_x_broadcast(void) { return 0; }
const float *coli_cuda_expert_group_take(int device) { (void)device; return NULL; }
/* The resident dense GEMV counters. The fake records no timings -- it has no
 * GPU timeline to record -- so this reports zero calls and qt_stats prints
 * nothing, which is the same thing a real backend does with the profiling flag
 * off. It exists so the tier tests link. */
void coli_cuda_dense_stats(int device,
                           uint64_t *calls, uint64_t *weight_bytes,
                           double *h2d_ms, double *kernel_ms,
                           double *d2h_ms, double *wall_ms) {
    (void)device;
    if (calls) *calls = 0;              if (weight_bytes) *weight_bytes = 0;
    if (h2d_ms) *h2d_ms = 0;            if (kernel_ms) *kernel_ms = 0;
    if (d2h_ms) *d2h_ms = 0;            if (wall_ms) *wall_ms = 0;
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}
void coli_cuda_stats(int device, size_t *count, size_t *bytes) {
    (void)device; if (count) *count = 0; if (bytes) *bytes = 0;
}

/* dense GEMV on a resident tensor (lm_head / DeltaNet projections placed on a
 * device). Counted, never computed: the placement tests check WHERE work
 * went; the arithmetic has its own oracle in the CUDA build. Parameters are
 * unused on purpose (CFLAGS carry -Wno-unused-parameter). */
static int fake_matmuls;
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device, int gs) {
    ColiCudaTensor *t = tensor ? *tensor : NULL;
    /* The real backend re-validates a cached tensor against the call's format
     * and group size and refuses a mismatch; a grouped upload answered with
     * gs 0 (or the other way round) is a matmul that never happens. Mirror
     * that here so a caller that drops the group size fails on the fake too. */
    if (t && (t->fmt != fmt || t->gs != (gs > 0 ? gs : 0))) return 0;
    fake_matmuls++;
    if (fake_dense_compute && t && t->fmt == 1 && t->w && t->sc && t->I == I && t->O == O) {
        const int8_t *q = (const int8_t *)t->w; const int g = t->gs; const int ng = g > 0 ? (I + g - 1) / g : 1;
        for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
            const int8_t *w = q + (size_t)o * I; const float *xs = x + (size_t)s * I; float a = 0.f;
            if (g > 0) {   /* grouped: the engine's Q38_TRUNK_CPU_INT8 loop, group subtotal times its scale */
                const float *sc = t->sc + (size_t)o * ng;
                for (int gi = 0; gi < ng; gi++) { int i0 = gi * g, i1 = i0 + g < I ? i0 + g : I; float ag = 0.f;
                    for (int i = i0; i < i1; i++) ag += xs[i] * (float)w[i]; a += ag * sc[gi]; }
                y[(size_t)s * O + o] = a;
            } else {
                for (int i = 0; i < I; i++) a += xs[i] * (float)w[i];
                y[(size_t)s * O + o] = a * t->sc[o];
            }
        }
    }
    return 1;
}

/* coli_cuda_deltanet_* (COLI_DN_GPU): the handle keeps its own state buffers,
 * the fake "device", and checks what the real create checks. decode calls
 * fake_dn_decode_hook: a test that includes qwen36.c points it at the
 * engine's own CPU deltanet() run on those buffers, so a GPU-path decode is
 * the CPU path on the device copy of the state and must match it byte for
 * byte. NULL (the default) makes decode fail. fake_dn_fail_decode fails it
 * without the hook, fake_dn_fail_upload fails state_upload. */
struct ColiCudaDeltaNet {
    int H, vh, vk, kdim, vdim, convk, device;
    const float *wab;
    float *rec, *ring; size_t nrec, nring;
};
static int fake_dn_creates, fake_dn_decodes, fake_dn_uploads, fake_dn_downloads, fake_dn_frees;
static int fake_dn_fail_decode, fake_dn_fail_upload, fake_dn_absent;
static int (*fake_dn_decode_hook)(ColiCudaDeltaNet *h, const float *x, float *out) = NULL;
int coli_cuda_has_deltanet(void) { return !fake_dn_absent; }
int coli_cuda_deltanet_create(ColiCudaDeltaNet **handle, ColiCudaTensor *dnproj, ColiCudaTensor *dnout,
                              int H, int vh, int vk, int kdim, int vdim, int convk, float eps,
                              const float *wab, const float *alog, const float *dtbias,
                              const float *wconv, const float *normw) {
    (void)eps;
    if (!handle) return 0;
    *handle = NULL;
    const int conv_dim = 2 * vk * kdim + vh * vdim, value_dim = vh * vdim;
    if (!dnproj || !dnout || dnproj->device != dnout->device) return 0;
    if (dnproj->I != H || dnproj->O != conv_dim + value_dim || dnout->I != value_dim || dnout->O != H) return 0;
    if (!wab || !alog || !dtbias || !wconv || !normw || vh % vk || convk < 2) return 0;
    ColiCudaDeltaNet *h = (ColiCudaDeltaNet *)calloc(1, sizeof *h);
    if (!h) return 0;
    h->H = H; h->vh = vh; h->vk = vk; h->kdim = kdim; h->vdim = vdim; h->convk = convk;
    h->device = dnproj->device; h->wab = wab;
    h->nrec = (size_t)vh * kdim * vdim; h->nring = (size_t)conv_dim * (convk - 1);
    h->rec = (float *)calloc(h->nrec, sizeof(float)); h->ring = (float *)calloc(h->nring, sizeof(float));
    if (!h->rec || !h->ring) { free(h->rec); free(h->ring); free(h); return 0; }
    fake_dn_creates++;
    *handle = h;
    return 1;
}
int coli_cuda_deltanet_decode(ColiCudaDeltaNet *h, const float *x, float *out) {
    if (!h || fake_dn_fail_decode || !fake_dn_decode_hook) return 0;
    fake_dn_decodes++;
    return fake_dn_decode_hook(h, x, out);
}
int coli_cuda_deltanet_state_upload(ColiCudaDeltaNet *h, const float *rec, const float *ring) {
    if (!h || !rec || !ring || fake_dn_fail_upload) return 0;
    memcpy(h->rec, rec, h->nrec * sizeof(float)); memcpy(h->ring, ring, h->nring * sizeof(float));
    fake_dn_uploads++;
    return 1;
}
int coli_cuda_deltanet_state_download(ColiCudaDeltaNet *h, float *rec, float *ring) {
    if (!h || !rec || !ring) return 0;
    memcpy(rec, h->rec, h->nrec * sizeof(float)); memcpy(ring, h->ring, h->nring * sizeof(float));
    fake_dn_downloads++;
    return 1;
}
int coli_cuda_deltanet_state_zero(ColiCudaDeltaNet *h) {
    if (!h) return 0;
    memset(h->rec, 0, h->nrec * sizeof(float)); memset(h->ring, 0, h->nring * sizeof(float));
    return 1;
}
size_t coli_cuda_deltanet_bytes(const ColiCudaDeltaNet *h) {
    return h ? (h->nrec + h->nring) * sizeof(float) : 0;
}
void coli_cuda_deltanet_free(ColiCudaDeltaNet *h) {
    if (!h) return;
    free(h->rec); free(h->ring); free(h);
    fake_dn_frees++;
}

#endif /* QWEN36_FAKE_CUDA_H */
