/* bench_qwen36_decode_omp — where the CPU half of a Qwen3.6 decode token goes.
 *
 * The profile on a 7950X + 4070 Ti SUPER (COLI_TIMERS=1, 128 tokens, warm
 * cache, 56.4 ms/token) charges 12.5 ms to the shared expert and 9.0-10.0 ms to
 * the DeltaNet output projection.  Two readings explain that, and they call for
 * OPPOSITE fixes:
 *
 *   sync-bound    those kernels are ~126 MB and ~252 MB of int8 per token but
 *                 only ~126 M and ~250 M MACs, and they are cut into 120 and 30
 *                 OpenMP regions per token.  If a region entry costs tens of
 *                 microseconds (a sleeping team, which is what a CUDA build
 *                 gets), the regions ARE the time.  Fix: fewer, bigger regions.
 *
 *   stream-bound  400 MB of weights per token has to cross the memory bus no
 *                 matter how it is cut up.  At 60 GB/s that is 6.7 ms, at
 *                 15 GB/s it is 27 ms.  Fix: read fewer bytes (lower precision,
 *                 or move the matrix to the GPU); region count is irrelevant.
 *
 * A MAC count alone cannot tell these apart -- it ignores memory entirely,
 * which is how "6-15 ms of headroom" was once read off one.  This benchmark
 * measures both ends on the machine in front of it, with no model file:
 *
 *   - the same GEMVs at the shipping shapes, in the engine's own kernels,
 *     cut the way the engine cuts them and cut into one region per layer;
 *   - the achieved GB/s of each, against a plain streaming read of the same
 *     bytes, which is this machine's ceiling for the access pattern;
 *   - the fork/join floor of one empty region, for the team as configured.
 *
 * If fused ~= unfused and both sit near the streaming ceiling, the kernels are
 * stream-bound and no amount of OpenMP surgery will move them.  If fused is
 * well ahead of unfused, or both are far below the ceiling, the regions are.
 *
 *   make -C c tests/bench_qwen36_decode_omp ARCH=native
 *   OMP_NUM_THREADS=16 ./c/tests/bench_qwen36_decode_omp
 *
 * Run it twice with OMP_WAIT_POLICY unset and set to active to see what a
 * spinning team is worth here before changing the engine for it.
 * Args: [reps] [samples] [layers] [hidden] [shared_inter] [value_dim].
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static float wvalue(int64_t i, int salt) {
    int v = (int)((i * 31 + salt * 17) % 251);
    return (float)(v - 125) / (float)(109 + salt);
}

/* A dense matrix the way the engine holds one after quantize-at-load: the int8
 * copy in the registry, the f32 original shrunk to a live one-float key. */
static float *dense_i8(int I, int O, int salt) {
    float *W = falloc((int64_t)O * I);
    for (int64_t i = 0; i < (int64_t)O * I; i++) W[i] = wvalue(i, salt);
    int before = g_qdw_n;
    qdw_register(W, I, O);
    if (g_qdw_n != before + 1) { fprintf(stderr, "registry refused %dx%d\n", O, I); exit(2); }
    float *t = realloc(W, sizeof(float));
    if (t) { W = t; g_qdw[g_qdw_n - 1].w = t; }
    return W;
}

static int8_t *raw_i8(int I, int O, int salt, float **scale) {
    int8_t *q = malloc((size_t)O * I);
    float *sc = falloc(O);
    if (!q) { fprintf(stderr, "OOM %d x %d\n", O, I); exit(2); }
    for (int64_t i = 0; i < (int64_t)O * I; i++) q[i] = (int8_t)((i * 29 + salt * 7) % 255 - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.001f * (float)(1 + (o * 11) % 31);
    *scale = sc;
    return q;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_d);
    return v[n / 2];
}

/* The three-call shared expert, exactly as moe() spells it at S=1. */
static void shared_three(Layer *l, const float *x, float *out, int D, int Ish,
                         float *sh, float *shu, float *shd) {
    matmul_d(sh, x, l->sh_g, 1, D, Ish);
    matmul_d(shu, x, l->sh_u, 1, D, Ish);
    for (int i = 0; i < Ish; i++) { float sv = sh[i]; sh[i] = (sv / (1.f + expf(-sv))) * shu[i]; }
    matmul_d(shd, sh, l->sh_d, 1, Ish, D);
    float sgate = 1.f;
    if (l->sh_gate) {
        float sg = 0.f; const float *wg = l->sh_gate;
        for (int i = 0; i < D; i++) sg += x[i] * wg[i];
        sgate = 1.f / (1.f + expf(-sg));
    }
    for (int d = 0; d < D; d++) out[d] += sgate * shd[d];
}

static volatile long g_sink;

/* One empty region, the team as configured: the cost every #pragma pays before
 * it does any work.  bench_omp_grain.c measures the same floor standalone. */
static double forkjoin_ns(long reps) {
    long acc = 0;
    double t0 = now_s();
    for (long r = 0; r < reps; r++) {
        #pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < 1; i++) acc += i + r;
    }
    double par = now_s() - t0;
    t0 = now_s();
    for (long r = 0; r < reps; r++)
        for (int i = 0; i < 1; i++) acc += i + r;
    double seq = now_s() - t0;
    g_sink = acc;
    return (par - seq) / (double)reps * 1e9;
}

/* What this machine can pull through the same bytes with nothing in the way. */
static double stream_read(const int8_t **blocks, const size_t *sizes, int n) {
    double t0 = now_s();
    long total = 0;
    for (int b = 0; b < n; b++) {
        const int8_t *p = blocks[b];
        size_t len = sizes[b];
        long acc = 0;
        #pragma omp parallel for schedule(static) reduction(+ : acc)
        for (int64_t i = 0; i < (int64_t)len; i += 64) acc += p[i];
        total += acc;
    }
    g_sink = total;
    return now_s() - t0;
}

static void row(const char *name, double ms, double mb, int regions) {
    printf("  %-26s %8.2f ms   %7.1f MB   %6.1f GB/s   %5d regions\n",
           name, ms, mb, mb / 1024.0 / (ms / 1000.0), regions);
}

int main(int argc, char **argv) {
    unsetenv("COLI_DENSE_I8");
    unsetenv("QWEN36_SHARED_FUSE");
    long reps   = argc > 1 ? atol(argv[1]) : 8;
    int samples = argc > 2 ? atoi(argv[2]) : 5;
    int L       = argc > 3 ? atoi(argv[3]) : 40;    /* layers (all carry an MoE block) */
    int D       = argc > 4 ? atoi(argv[4]) : 2048;  /* hidden_size */
    int Ish     = argc > 5 ? atoi(argv[5]) : 512;   /* shared_expert_intermediate_size */
    int VD      = argc > 6 ? atoi(argv[6]) : 4096;  /* dn_vheads * dn_vdim */
    int E       = 256;                              /* router outputs */
    int LDN     = L - L / 4;                        /* DeltaNet layers: 3 of every 4 */
    if (reps < 1) reps = 1;
    if (samples < 1) samples = 1;

    int nthreads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    { 
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    double mb_shared = 3.0 * Ish * D * L / 1048576.0;
    double mb_dn     = (double)VD * D * LDN / 1048576.0;
    double mb_router = (double)E * D * L / 1048576.0;
    printf("bench_qwen36_decode_omp: L=%d D=%d shared_inter=%d value_dim=%d, team of %d\n",
           L, D, Ish, VD, nthreads);
    printf("per token: shared %.0f MB, dn_out %.0f MB, router %.0f MB -> %.0f MB of int8 weights\n\n",
           mb_shared, mb_dn, mb_router, mb_shared + mb_dn + mb_router);

    /* --- the working set: distinct memory per layer, so nothing stays in L3 --- */
    Layer *ls = calloc((size_t)L, sizeof(Layer));
    if (!ls) { fputs("OOM layers\n", stderr); return 2; }
    for (int i = 0; i < L; i++) {
        ls[i].sh_g = dense_i8(D, Ish, 2 + i);
        ls[i].sh_u = dense_i8(D, Ish, 3 + i);
        ls[i].sh_d = dense_i8(Ish, D, 4 + i);
        ls[i].sh_gate = falloc(D);
        for (int k = 0; k < D; k++) ls[i].sh_gate[k] = wvalue(k, 5 + i);
    }
    int8_t **dnq = malloc(sizeof(int8_t *) * (size_t)LDN);
    float  **dns = malloc(sizeof(float *)  * (size_t)LDN);
    for (int i = 0; i < LDN; i++) dnq[i] = raw_i8(VD, D, 11 + i, &dns[i]);
    int8_t **rtq = malloc(sizeof(int8_t *) * (size_t)L);
    float  **rts = malloc(sizeof(float *)  * (size_t)L);
    for (int i = 0; i < L; i++) rtq[i] = raw_i8(D, E, 23 + i, &rts[i]);

    float *x = falloc(D), *xv = falloc(VD);
    for (int i = 0; i < D; i++)  x[i]  = wvalue(i, 6);
    for (int i = 0; i < VD; i++) xv[i] = wvalue(i, 7);
    float *sh = falloc(Ish), *shu = falloc(Ish), *shd = falloc(D);
    float *out_a = falloc(D), *out_b = falloc(D);
    float *dny = falloc(D), *rty = falloc(E);

    /* --- the two shapes must agree to the bit before either is timed --- */
    memset(out_a, 0, (size_t)D * sizeof(float));
    memset(out_b, 0, (size_t)D * sizeof(float));
    for (int i = 0; i < L; i++) shared_three(&ls[i], x, out_a, D, Ish, sh, shu, shd);
    for (int i = 0; i < L; i++)
        if (!qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh)) {
            fputs("FAIL: the fused shared expert declined int8 weights\n", stderr);
            return 1;
        }
    if (memcmp(out_a, out_b, (size_t)D * sizeof(float))) {
        fputs("FAIL: fused and three-call shared expert differ\n", stderr);
        return 1;
    }

    double *t3 = malloc(sizeof(double) * (size_t)samples);
    double *t1 = malloc(sizeof(double) * (size_t)samples);
    double *td = malloc(sizeof(double) * (size_t)samples);
    double *tr = malloc(sizeof(double) * (size_t)samples);
    double *tstream = malloc(sizeof(double) * (size_t)samples);

    const int8_t **blocks = malloc(sizeof(int8_t *) * (size_t)(3 * L + LDN + L));
    size_t *sizes = malloc(sizeof(size_t) * (size_t)(3 * L + LDN + L));
    int nb = 0;
    for (int i = 0; i < g_qdw_n; i++) {
        blocks[nb] = g_qdw[i].q; sizes[nb++] = (size_t)g_qdw[i].I * g_qdw[i].O;
    }
    for (int i = 0; i < LDN; i++) { blocks[nb] = dnq[i]; sizes[nb++] = (size_t)VD * D; }
    for (int i = 0; i < L;   i++) { blocks[nb] = rtq[i]; sizes[nb++] = (size_t)D * E; }

    for (int s = 0; s < samples; s++) {
        double a = 0, b = 0;
        /* alternate the order every sample: a machine that drifts mid-run must
         * not be able to hand the win to whichever arm ran first. */
        for (long r = 0; r < reps; r++) {
            if ((s + r) & 1) {
                double t = now_s();
                for (int i = 0; i < L; i++) qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh);
                b += now_s() - t;
                t = now_s();
                for (int i = 0; i < L; i++) shared_three(&ls[i], x, out_a, D, Ish, sh, shu, shd);
                a += now_s() - t;
            } else {
                double t = now_s();
                for (int i = 0; i < L; i++) shared_three(&ls[i], x, out_a, D, Ish, sh, shu, shd);
                a += now_s() - t;
                t = now_s();
                for (int i = 0; i < L; i++) qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh);
                b += now_s() - t;
            }
        }
        t3[s] = a / (double)reps * 1000.0;
        t1[s] = b / (double)reps * 1000.0;

        double t = now_s();
        for (long r = 0; r < reps; r++)
            for (int i = 0; i < LDN; i++) matmul_q(dny, xv, dnq[i], dns[i], VD, D);
        td[s] = (now_s() - t) / (double)reps * 1000.0;

        t = now_s();
        for (long r = 0; r < reps; r++)
            for (int i = 0; i < L; i++) matmul_q(rty, x, rtq[i], rts[i], D, E);
        tr[s] = (now_s() - t) / (double)reps * 1000.0;

        tstream[s] = stream_read(blocks, sizes, nb) * 1000.0;
    }

    double mb_all = mb_shared + mb_dn + mb_router;
    printf("  %-26s %11s %13s %13s %13s\n", "", "per token", "weights", "achieved", "OpenMP");
    row("shared expert, 3 calls", median(t3, samples), mb_shared, 3 * L);
    row("shared expert, fused",   median(t1, samples), mb_shared, L);
    row("dn out_proj",            median(td, samples), mb_dn,     LDN);
    row("router",                 median(tr, samples), mb_router, L);
    row("all three, streamed",    median(tstream, samples), mb_all, nb);

    double fj = forkjoin_ns(20000);
    double saved = median(t3, samples) - median(t1, samples);
    printf("\n  fork/join floor: %.0f ns per region (team of %d)\n", fj, nthreads);
    printf("  the %d regions the fusion removes, at that floor: %.2f ms/token\n",
           2 * L, fj * 2.0 * L / 1e6);
    printf("  measured saving: %.2f ms/token\n", saved);
    printf("  streaming ceiling says the shared expert cannot go below %.2f ms/token\n",
           median(tstream, samples) * mb_shared / mb_all);
    puts("\n  fused ~= 3 calls and both near the streaming line -> stream-bound,");
    puts("  region surgery will not pay. A large gap -> the regions are the cost.");

    for (int i = 0; i < g_qdw_n; i++) { free(g_qdw[i].q); free(g_qdw[i].sc); }
    g_qdw_n = 0;
    for (int i = 0; i < L; i++) { free(ls[i].sh_g); free(ls[i].sh_u); free(ls[i].sh_d); free(ls[i].sh_gate); }
    for (int i = 0; i < LDN; i++) { free(dnq[i]); free(dns[i]); }
    for (int i = 0; i < L; i++) { free(rtq[i]); free(rts[i]); }
    free(ls); free(dnq); free(dns); free(rtq); free(rts); free(blocks); free(sizes);
    free(x); free(xv); free(sh); free(shu); free(shd); free(out_a); free(out_b);
    free(dny); free(rty); free(t3); free(t1); free(td); free(tr); free(tstream);
    return 0;
}
