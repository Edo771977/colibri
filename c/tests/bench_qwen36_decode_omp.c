/* bench_qwen36_decode_omp — where the CPU half of a Qwen3.6 decode token goes.
 *
 * The profile on a 7950X + 4070 Ti SUPER (COLI_TIMERS=1, 128 tokens, warm
 * cache, 56.4 ms/token) charges 9.3 ms to the shared expert, 9.0 to the
 * DeltaNet output projection and 2.7 to the router.  Two readings explain that,
 * and they call for OPPOSITE fixes:
 *
 *   sync-bound    those kernels are cut into 120, 30 and 40 OpenMP regions per
 *                 token.  A region entry costs what it costs however small the
 *                 body is, and at decode these bodies are microseconds.
 *                 Fix: fewer, bigger regions.
 *
 *   stream-bound  they are also 126 MB, 252 MB and 21 MB of int8 weights per
 *                 token, and those bytes have to cross the memory bus however
 *                 the work is cut up.  Fix: read fewer bytes; region count is
 *                 irrelevant.
 *
 * A MAC count cannot tell them apart -- it does not model memory at all, which
 * is how a "6-15 ms of headroom" estimate gets written down for a kernel that
 * might already be at its DRAM floor.  This benchmark measures both ends on the
 * machine in front of it, with no model file and no GPU:
 *
 *   - the shipping shapes through the engine's own kernels, cut the way the
 *     engine cuts them and cut into one region per layer;
 *   - the cost of an empty parallel region, and of a barrier INSIDE one, for
 *     the team as configured -- the two prices region surgery trades between;
 *   - a streaming read of the same bytes in ONE region, which is this machine's
 *     ceiling for the access pattern.
 *
 * Each row is then split: regions x the measured floor, and the rest, with the
 * bandwidth that rest implies.  If the split leaves a sane GB/s the model
 * holds and the region column is the bill; if the regions account for almost
 * nothing, the kernels are stream-bound and no OpenMP surgery will pay.
 *
 *   make -C c tests/bench_qwen36_decode_omp ARCH=native
 *   OMP_NUM_THREADS=<physical cores> ./c/tests/bench_qwen36_decode_omp
 *
 * Worth running at several team sizes: the floor is not linear in threads, and
 * a runtime whose floor is tens of microseconds is itself the finding.
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

/* One empty region, the team as configured: what every #pragma pays before it
 * does any work. bench_omp_grain.c measures the same floor standalone. */
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

/* A barrier INSIDE a region that is already open. Fusing three regions into one
 * trades two region entries for two of these, so it only pays if this is the
 * cheaper price. Nothing guarantees that it is. */
static double barrier_ns(long reps) {
    double t0 = now_s();
    #pragma omp parallel
    {
        for (long r = 0; r < reps; r++) {
            #pragma omp barrier
        }
    }
    return (now_s() - t0) / (double)reps * 1e9;
}

typedef struct { const int8_t *p; size_t len; } Chunk;

/* Every byte the three groups read, in ONE region: this machine's ceiling for
 * the access pattern. Cutting it into one region per matrix would measure the
 * regions again and call the result a bandwidth ceiling -- which is exactly the
 * mistake this line exists to avoid. */
static double stream_read(const Chunk *ch, int n) {
    long acc = 0;
    double t0 = now_s();
    #pragma omp parallel for schedule(static) reduction(+ : acc)
    for (int c = 0; c < n; c++) {
        const int8_t *p = ch[c].p;
        size_t len = ch[c].len;
        for (size_t i = 0; i < len; i += 64) acc += p[i];
    }
    g_sink = acc;
    return now_s() - t0;
}

/* Charge each row its SYNC POINTS, not just its regions: fusing three regions
 * into one does not remove the synchronisation, it converts region entries into
 * barriers, and nothing says a barrier is the cheaper of the two. Then charge
 * the bytes at the rate the one-region line actually achieved. What is left is
 * what the model does not explain, and a fused arm whose residual is large is
 * telling you it broke the access pattern. */
static void row(const char *name, double ms, double mb, int regions, int barriers,
                double fj_ms, double ba_ms, double gbs) {
    double sync = regions * fj_ms + barriers * ba_ms;
    double dram = gbs > 0 ? mb / 1024.0 / gbs * 1000.0 : 0;
    printf("  %-22s %7.2f ms %6.1f MB | %6.2f ms sync (%3dr+%3db) | %6.2f ms dram | %+6.2f ms left\n",
           name, ms, mb, sync, regions, barriers, dram, ms - sync - dram);
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
        if (!qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh, shu)) {
            fputs("FAIL: the fused shared expert declined int8 weights\n", stderr);
            return 1;
        }
    if (memcmp(out_a, out_b, (size_t)D * sizeof(float))) {
        fputs("FAIL: fused and three-call shared expert differ\n", stderr);
        return 1;
    }

    /* --- the two sync prices, measured before anything is attributed to them --- */
    double fj_ns = forkjoin_ns(20000);
    double ba_ns = barrier_ns(20000);
    double fj_ms = fj_ns / 1e6;

    /* --- every weight byte, cut into ~1 MB chunks for one balanced region --- */
    int nblocks = g_qdw_n + LDN + L;
    const int8_t **bp = malloc(sizeof(int8_t *) * (size_t)nblocks);
    size_t *bl = malloc(sizeof(size_t) * (size_t)nblocks);
    int nb = 0;
    for (int i = 0; i < g_qdw_n; i++) { bp[nb] = g_qdw[i].q; bl[nb++] = (size_t)g_qdw[i].I * g_qdw[i].O; }
    for (int i = 0; i < LDN; i++) { bp[nb] = dnq[i]; bl[nb++] = (size_t)VD * D; }
    for (int i = 0; i < L;   i++) { bp[nb] = rtq[i]; bl[nb++] = (size_t)D * E; }
    size_t grain = 1u << 20;
    int nch = 0;
    for (int b = 0; b < nb; b++) nch += (int)((bl[b] + grain - 1) / grain);
    Chunk *ch = malloc(sizeof(Chunk) * (size_t)nch);
    int k = 0;
    for (int b = 0; b < nb; b++)
        for (size_t off = 0; off < bl[b]; off += grain) {
            size_t len = bl[b] - off; if (len > grain) len = grain;
            ch[k].p = bp[b] + off; ch[k].len = len; k++;
        }

    double *t3 = malloc(sizeof(double) * (size_t)samples);
    double *t1 = malloc(sizeof(double) * (size_t)samples);
    double *td = malloc(sizeof(double) * (size_t)samples);
    double *tr = malloc(sizeof(double) * (size_t)samples);
    double *ts = malloc(sizeof(double) * (size_t)samples);

    for (int s = 0; s < samples; s++) {
        double a = 0, b = 0;
        /* alternate the order every sample: a machine that drifts mid-run must
         * not be able to hand the win to whichever arm ran first. */
        for (long r = 0; r < reps; r++) {
            if ((s + r) & 1) {
                double t = now_s();
                for (int i = 0; i < L; i++) qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh, shu);
                b += now_s() - t;
                t = now_s();
                for (int i = 0; i < L; i++) shared_three(&ls[i], x, out_a, D, Ish, sh, shu, shd);
                a += now_s() - t;
            } else {
                double t = now_s();
                for (int i = 0; i < L; i++) shared_three(&ls[i], x, out_a, D, Ish, sh, shu, shd);
                a += now_s() - t;
                t = now_s();
                for (int i = 0; i < L; i++) qwen_shared_fused_row(&ls[i], x, out_b, D, Ish, sh, shu);
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

        ts[s] = stream_read(ch, nch) * 1000.0;
    }

    double mb_all = mb_shared + mb_dn + mb_router;
    double ms_stream = median(ts, samples);
    double gbs = mb_all / 1024.0 / (ms_stream / 1000.0);   /* one region: the real ceiling */
    double ba_ms = ba_ns / 1e6;

    printf("  one empty parallel region:   %7.2f us   (team of %d)\n", fj_ns / 1000.0, nthreads);
    printf("  one barrier inside a region: %7.2f us   %s\n",
           ba_ns / 1000.0,
           ba_ns < fj_ns ? "<- cheaper than a region: fusing can pay"
                         : "<- NOT cheaper than a region: fusing trades down");
    printf("  streaming read, one region:  %7.2f GB/s\n\n", gbs);

    row("shared expert, 3 calls", median(t3, samples), mb_shared, 3 * L, 0,   fj_ms, ba_ms, gbs);
    row("shared expert, fused",   median(t1, samples), mb_shared, L,     L,   fj_ms, ba_ms, gbs);
    row("dn out_proj",            median(td, samples), mb_dn,     LDN,   0,   fj_ms, ba_ms, gbs);
    row("router",                 median(tr, samples), mb_router, L,     0,   fj_ms, ba_ms, gbs);
    row("all bytes, one region",  ms_stream,           mb_all,    1,     0,   fj_ms, ba_ms, gbs);

    double sync3 = 3.0 * L * fj_ms, sync1 = L * fj_ms + L * ba_ms;
    printf("\n  fusing trades %.2f ms of region entries for %.2f ms of barriers: %+.2f ms predicted\n",
           sync3, sync1, sync1 - sync3);
    printf("  measured: %+.2f ms/token\n", median(t1, samples) - median(t3, samples));

    if (fj_ns > 10000.0 || ba_ns > 10000.0) {
        puts("\n  *** The sync prices above are tens of microseconds. A healthy OpenMP");
        puts("  *** runtime charges single digits, so on this host the RUNTIME is the");
        puts("  *** dominant cost of a decode token, not any kernel in it. A decode");
        printf("  *** token crosses roughly 250-300 regions: %.0f-%.0f ms at this floor.\n",
               250 * fj_ms, 300 * fj_ms);
        puts("  *** Before reshaping kernels, try another OpenMP runtime (clang/libomp");
        puts("  *** rather than MinGW libgomp on Windows) and re-run this. No kernel");
        puts("  *** change can recover what the runtime is spending.");
    } else {
        puts("\n  Read the last column: a residual near zero on every row means the model");
        puts("  holds, and the sync column is then the bill worth attacking.");
    }

    for (int i = 0; i < g_qdw_n; i++) { free(g_qdw[i].q); free(g_qdw[i].sc); }
    g_qdw_n = 0;
    for (int i = 0; i < L; i++) { free(ls[i].sh_g); free(ls[i].sh_u); free(ls[i].sh_d); free(ls[i].sh_gate); }
    for (int i = 0; i < LDN; i++) { free(dnq[i]); free(dns[i]); }
    for (int i = 0; i < L; i++) { free(rtq[i]); free(rts[i]); }
    free(ls); free(dnq); free(dns); free(rtq); free(rts); free(bp); free(bl); free(ch);
    free(x); free(xv); free(sh); free(shu); free(shd); free(out_a); free(out_b);
    free(dny); free(rty); free(t3); free(t1); free(td); free(tr); free(ts);
    return 0;
}
