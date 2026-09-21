/* Per-site attribution of the resident dense GEMV profile, on the fake
 * backend. No GPU, no toolkit.
 *
 * WHY THIS EXISTS. `dense_stats` is one aggregate over every placed GEMV in
 * the engine, and the question it cannot answer is which phase is spending the
 * time. On qwen36, DeltaNet makes 60 of the ~70 dense calls a decode token
 * makes -- 30 dnproj plus 30 dnout -- so most of that aggregate belongs to one
 * phase and nothing printed said so.
 *
 * That question got answered wrongly once already, by dividing the aggregate
 * by a throughput measured in a DIFFERENT session: the same 5224 calls
 * reported 372 GB/s one day and 175 GB/s the next, and any arithmetic that
 * divides by that inherits the factor of two. So the split is measured inside
 * one run, by reading the backend's own accumulators either side of each call.
 *
 * WHAT IS ASSERTED, and why each one is here:
 *
 *   1. Calls land on the site that owns the handle, and dnproj/lmhead on
 *      theirs -- the basic contract.
 *   2. The per-site totals SUM to the aggregate. A split that does not add up
 *      to the number it splits is worse than no split: it looks authoritative
 *      and is wrong. This is the assertion that would catch a double charge or
 *      a dropped sample.
 *   3. An UNLABELLED handle lands in `other` rather than being charged to
 *      whichever site happened to be last. qwen38_core.h allocates dense
 *      handles through the same entry point and labels none of them, so this
 *      is the normal case for another engine, not an edge case.
 *   4. The `calls`-delta guard fires. The whole attribution rests on nothing
 *      else reaching coli_cuda_matmul between the two reads; the code checks
 *      that rather than trusting it, and a violated sample must land in
 *      `unattributed` instead of being charged to the wrong site. Tested by
 *      making the backend advance its own counter behind the tier's back.
 *   5. With profiling off, nothing is attributed and nothing is spent.
 *
 * Build: part of the CPU suite (`make check`).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2 };

/* One int8 matrix small enough to be uninteresting; the arithmetic is not
 * what is under test here, the bookkeeping is. */
static int make_handle(int I, int O, int8_t **qout, float **scout) {
    int8_t *q = malloc((size_t)I * O);
    float *sc = malloc((size_t)O * sizeof(float));
    if (!q || !sc) { free(q); free(sc); return -1; }
    memset(q, 1, (size_t)I * O);
    for (int o = 0; o < O; o++) sc[o] = 0.01f;
    int h = qt_dense_init(q, sc, I, O, 0, 0);
    *qout = q; *scout = sc;
    return h;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", "", 1);
    setenv("COLI_CUDA_PROFILE", "1", 1);      /* the per-site table is opt-in */
    fake_ndev = 1; fake_uploads = 0;
    fake_dense_profile = 1;

    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2)
                     + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
    char gb[64];
    snprintf(gb, sizeof gb, "%.15f", (double)(8 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    qt_trunk_offer("lmhead", 0, exp_bytes);
    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0, 1)) { printf("FAIL tier did not start\n"); return 1; }

    const int I = IH, O = D;
    int8_t *q1, *q2, *q3; float *s1, *s2, *s3;
    int h_dnout   = make_handle(I, O, &q1, &s1);
    int h_attnout = make_handle(I, O, &q2, &s2);
    int h_plain   = make_handle(I, O, &q3, &s3);
    if (h_dnout < 0 || h_attnout < 0 || h_plain < 0) { printf("FAIL handles\n"); return 1; }
    qt_dense_site(h_dnout,   QT_SITE_DNOUT);
    qt_dense_site(h_attnout, QT_SITE_ATTNOUT);
    /* h_plain deliberately unlabelled: that is what another engine does. */

    float *x = calloc((size_t)I, sizeof(float));
    float *y = calloc((size_t)O, sizeof(float));
    if (!x || !y) { printf("FAIL scratch\n"); return 1; }

    printf(" 1. calls land on the site that owns the handle\n");
    for (int i = 0; i < 5; i++) check(qt_dense_matmul(h_dnout,   y, x, I, O), "dnout call ran");
    for (int i = 0; i < 3; i++) check(qt_dense_matmul(h_attnout, y, x, I, O), "attnout call ran");
    for (int i = 0; i < 2; i++) check(qt_dense_matmul(h_plain,   y, x, I, O), "unlabelled call ran");
    check(g_site_calls[QT_SITE_DNOUT]   == 5, "5 calls charged to dnout");
    check(g_site_calls[QT_SITE_ATTNOUT] == 3, "3 calls charged to attnout");
    check(g_site_calls[QT_SITE_OTHER]   == 2, "an unlabelled handle charges `other`, not a neighbour");
    check(g_site_calls[QT_SITE_DNPROJ]  == 0 && g_site_calls[QT_SITE_LMHEAD] == 0,
          "sites that never ran stay at zero");

    printf(" 2. the milliseconds follow the calls, exactly\n");
    /* Constants, not tolerances: the fake advances by a known amount per call,
     * so 5 dnout calls must be 5x that amount. A tolerance here would pass a
     * delta computed against the wrong baseline. */
    check(g_site_wall[QT_SITE_DNOUT]   == 5 * fake_ms_wall,   "dnout wall is 5 calls' worth");
    check(g_site_kernel[QT_SITE_DNOUT] == 5 * fake_ms_kernel, "dnout kernel is 5 calls' worth");
    check(g_site_h2d[QT_SITE_ATTNOUT]  == 3 * fake_ms_h2d,    "attnout h2d is 3 calls' worth");
    check(g_site_d2h[QT_SITE_OTHER]    == 2 * fake_ms_d2h,    "other d2h is 2 calls' worth");

    printf(" 3. the split adds up to the thing it splits\n");
    {
        uint64_t agg_calls = 0, agg_bytes = 0;
        double a_h2d = 0, a_ker = 0, a_d2h = 0, a_wall = 0;
        coli_cuda_dense_stats(-1, &agg_calls, &agg_bytes, &a_h2d, &a_ker, &a_d2h, &a_wall);
        uint64_t sc = 0, sb = 0; double sh = 0, sk = 0, sd = 0, sw = 0;
        for (int i = 0; i < QT_SITE_N; i++) {
            sc += g_site_calls[i]; sb += g_site_bytes[i];
            sh += g_site_h2d[i];   sk += g_site_kernel[i];
            sd += g_site_d2h[i];   sw += g_site_wall[i];
        }
        check(sc == agg_calls, "per-site calls sum to dense_stats calls");
        check(sb == agg_bytes, "per-site bytes sum to dense_stats bytes");
        check(sh == a_h2d && sk == a_ker && sd == a_d2h && sw == a_wall,
              "per-site milliseconds sum to dense_stats milliseconds");
        check(g_site_unattributed == 0, "nothing was lost along the way");
    }

    printf(" 4. the guard: a call from somewhere else is NOT charged to a site\n");
    {
        /* The one assumption the attribution makes is that nothing else
         * reaches coli_cuda_matmul between the two reads. Break it on purpose:
         * a second call INSIDE that window makes the `calls` delta 2, and the
         * sample must be refused rather than charged at double value.
         *
         * Inside the window, not before it. A first draft bumped the counter
         * before qt_dense_matmul and the guard stayed silent -- correctly: the
         * tier's first read already included it, so the delta was still 1.
         * The window is what is under test, not the counter. */
        uint64_t before_site = g_site_calls[QT_SITE_DNOUT];
        uint64_t before_unatt = g_site_unattributed;
        fake_dense_interlopers = 1;
        check(qt_dense_matmul(h_dnout, y, x, I, O), "call ran during the interference");
        fake_dense_interlopers = 0;
        check(g_site_calls[QT_SITE_DNOUT] == before_site,
              "a sample with a calls-delta of 2 is NOT charged to the site");
        check(g_site_unattributed == before_unatt + 1,
              "it is counted as unattributed, not dropped in silence");
    }

    printf(" 5. profiling off: nothing measured, nothing spent\n");
    {
        /* site_profile_on() caches, so the flag cannot be flipped mid-process
         * the way COLI_CUDA_GRAPH can -- it is read ~70 times a token and the
         * getenv is not free. What IS checkable here is the other half of the
         * contract: with the backend's accumulators frozen, a call adds
         * nothing to any site. */
        uint64_t before[QT_SITE_N];
        for (int i = 0; i < QT_SITE_N; i++) before[i] = g_site_calls[i];
        uint64_t unatt = g_site_unattributed;
        fake_dense_profile = 0;                  /* accumulators stop moving */
        check(qt_dense_matmul(h_dnout, y, x, I, O), "call still runs");
        int moved = 0;
        for (int i = 0; i < QT_SITE_N; i++) if (g_site_calls[i] != before[i]) moved = 1;
        check(!moved, "a call the backend did not record charges no site");
        check(g_site_unattributed == unatt + 1,
              "and it is visible as unattributed rather than invented");
        fake_dense_profile = 1;
    }

    /* The table itself must render without tripping over a site that never
     * ran; printing it is the only way to find a format string that lies. */
    printf(" 6. the table prints\n");
    dense_site_print();

    qt_shutdown();
    free(q1); free(s1); free(q2); free(s2); free(q3); free(s3);
    free(x); free(y);
    if (fails) { printf("test_qwen36_tier_dense_sites: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_dense_sites: ok\n");
    return 0;
}
