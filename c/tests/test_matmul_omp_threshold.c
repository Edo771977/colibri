/* matmul()'s OpenMP work threshold (COLI_MATMUL_OMP_MIN).
 *
 * WHY IT EXISTS. qwen36's two DeltaNet gate projections are [32 x 2048] --
 * 65k FMAs, roughly 4 us on one core -- and matmul() was opening an OpenMP
 * team for each, twice per DeltaNet layer, sixty regions a decode token. The
 * per-site dense profile put the pair at 0.86 ms/token, about four times the
 * work they contain; this project has priced a region at up to 53 us on the
 * same CPU under MinGW libgomp (tests/bench_qwen36_decode_omp).
 *
 * WHAT IS ASSERTED, and what deliberately is not:
 *
 *   1. BIT-IDENTITY between the two arms. The change is an `if` clause on the
 *      existing pragma, not a second implementation, so the same loop runs in
 *      the same per-row order either way and no result may move. This is the
 *      safety property, and it is the reason the change needs no token oracle.
 *   2. A double-precision reference underneath, so "identical" cannot mean
 *      "both equally wrong".
 *   3. The threshold sits where it was meant to: the DeltaNet gate shape falls
 *      BELOW it and an ordinary trunk matmul falls above. This is the decision
 *      property.
 *
 * What this file does NOT assert is that a team was really opened or really
 * skipped. Every other threshold in this repo gets a dispatch counter, because
 * there identity would also be satisfied by a branch that never fires. Here
 * the situation is inverted: identity holds in BOTH arms by construction, so
 * it cannot be used to prove dispatch, and matmul() has no seam to observe the
 * team through without changing the loop under test. The dispatch is therefore
 * checked at the CONDITION (assertion 3) and settled by measurement, not here.
 * Saying so is worth more than a counter that would only be testing itself.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../quant.h"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

static uint32_t rs = 0x12345678u;
static uint32_t rnd(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static float rndf(void){ return (float)((int)(rnd()%2001)-1000)/1000.0f; }

/* The same accumulation order matmul uses, spelled out in double: per output
 * row, sequential over I. If matmul ever reassociated that sum -- which is
 * what a vectorising compiler would do under -ffast-math -- this is what would
 * catch it. */
static void ref(double *y, const float *x, const float *W, int S, int I, int O){
    for (int o = 0; o < O; o++)
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I; const float *w = W + (size_t)o * I;
            double a = 0; for (int i = 0; i < I; i++) a += (double)xs[i] * w[i];
            y[(size_t)s * O + o] = a;
        }
}

static double rel_rms(const float *got, const double *want, size_t n){
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) { double d = got[i] - want[i]; se += d*d; sr += want[i]*want[i]; }
    return sr > 0 ? sqrt(se/sr) : sqrt(se);
}

static void shape(int S, int I, int O, const char *why){
    size_t wn = (size_t)I*O, yn = (size_t)S*O;
    float *W = malloc(wn*sizeof(float)), *x = malloc((size_t)S*I*sizeof(float));
    float *y_team = malloc(yn*sizeof(float)), *y_solo = malloc(yn*sizeof(float));
    double *want = malloc(yn*sizeof(double));
    char name[160];
    if (!W || !x || !y_team || !y_solo || !want) { printf("FAIL alloc\n"); fails++; return; }
    for (size_t i = 0; i < wn; i++) W[i] = rndf();
    for (size_t i = 0; i < (size_t)S*I; i++) x[i] = rndf();
    ref(want, x, W, S, I, O);

    g_matmul_omp_min = 0;                 /* a team for every shape */
    matmul(y_team, x, W, S, I, O);
    g_matmul_omp_min = INT64_MAX;         /* never a team */
    matmul(y_solo, x, W, S, I, O);

    int same = !memcmp(y_team, y_solo, yn*sizeof(float));
    double r = rel_rms(y_team, want, yn);
    printf("  S=%d [%5d x %5d]  ref rms %.2e |  %s   %s\n", S, O, I, r,
           same ? "bitwise" : "DIFFERS", why);
    snprintf(name, sizeof name, "team and no-team are bitwise identical S=%d [%d x %d]", S, O, I);
    check(same, name);
    snprintf(name, sizeof name, "within the double reference S=%d [%d x %d]", S, O, I);
    check(r < 1e-5, name);
    free(W); free(x); free(y_team); free(y_solo); free(want);
}

int main(void){
    printf("matmul OpenMP threshold: the two arms must not move a single bit\n");
    shape(1,  2048,   32, "qwen36 DeltaNet gate projection (in_proj_a / in_proj_b)");
    shape(1,  2048, 2048, "an ordinary trunk matmul");
    shape(4,  2048,   64, "S > 1");
    shape(1,   333,   17, "neither dimension a nice number");
    shape(1,     1,    1, "degenerate");
    shape(1,  2048,    1, "one output row: nothing for a team to split");

    printf("the knob parses, and ships OFF\n");
    {
        /* It ships off: measured worth nothing on an LLVM libomp host, where a
         * region costs single-digit us and the DeltaNet gate shape sits ON the
         * break-even point. See quant.h. So the first assertion here is that
         * the default changes NOTHING -- a threshold of 0 means every shape
         * still gets a team, exactly as before. */
        g_matmul_omp_min = -1;
        unsetenv("COLI_MATMUL_OMP_MIN");
        check(matmul_omp_min() == 0, "unset means a team for every shape (no default change)");
        check(COLI_MATMUL_OMP_MIN_DEFAULT == 0, "the compiled default is off");

        /* And that when someone DOES set it -- on a host where a region really
         * costs 53 us -- it lands between the work it was aimed at and the work
         * it must not touch. Checked at the condition, not the dispatch: see
         * the header. */
        g_matmul_omp_min = -1;
        setenv("COLI_MATMUL_OMP_MIN", "262144", 1);
        int64_t lim = matmul_omp_min();
        check(lim == 262144, "an explicit value is honoured");
        check((int64_t)1*2048*32   <  lim, "the DeltaNet gate shape falls below it");
        check((int64_t)1*2048*2048 >= lim, "an ordinary trunk matmul stays above it");

        g_matmul_omp_min = -1;
        setenv("COLI_MATMUL_OMP_MIN", "zzz", 1);
        check(matmul_omp_min() == COLI_MATMUL_OMP_MIN_DEFAULT,
              "garbage reads as the default, not as some other number");
        g_matmul_omp_min = -1;
        setenv("COLI_MATMUL_OMP_MIN", "-5", 1);
        check(matmul_omp_min() == COLI_MATMUL_OMP_MIN_DEFAULT,
              "a negative value reads as the default: -1 is the not-read-yet marker");
        unsetenv("COLI_MATMUL_OMP_MIN");
        g_matmul_omp_min = -1;
    }

    if (fails) { printf("test_matmul_omp_threshold: %d failure(s)\n", fails); return 1; }
    printf("test_matmul_omp_threshold: ok\n");
    return 0;
}
