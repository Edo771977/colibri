/* fmt=1 per-row int8, R rows per block (COLI_CUDA_I8_ROWS).
 *
 * quant_matmul_i8r changes ONE thing about the generic branch: a block carries
 * R output rows and reads each activation once for all of them, instead of one
 * row re-reading the whole vector. The per-thread partition, the byte-at-a-time
 * weight read, the 256-wide reduction tree and the trailing f32 scale are
 * untouched, so every addition happens in the same order on the same operands.
 *
 * That makes the contract BITWISE IDENTITY, not a tolerance -- and identity is
 * the whole point. #28's rework reordered the sums, which moved every logit's
 * last bits and put a token-exact oracle between the patch and any default
 * change. This one cannot move a logit, so it cannot flip a token, so the only
 * open question it leaves is whether it is faster.
 *
 *   1. memcmp against the original kernel, for R in {2,4,8}, on shapes that
 *      exercise every edge the blocking introduces: O divisible by R, O with a
 *      remainder (short last block), O SMALLER than R, I not a multiple of the
 *      block width (strided tail), I smaller than one thread per element, and
 *      S > 1.
 *   2. A double-precision CPU reference underneath, so "identical" cannot mean
 *      "both equally wrong".
 *   3. Proof of bite: a mutated weight must move the output. Identity between
 *      two kernels that read nothing would otherwise pass.
 *   4. Launch-to-launch determinism, bitwise, for the blocked kernel.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_int8_rows_cuda.cu -o tests/test_int8_rows
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#include "../backend_cuda.cu"

#ifdef _WIN32
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
#endif

static void cpu_gemv_i8(const int8_t *w, const float *sc, int I, int O,
                        const float *x, float *y) {
    for (int o = 0; o < O; o++) {
        double a = 0.0;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)w[(size_t)o * I + i];
        y[o] = (float)(a * (double)sc[o]);
    }
}

static uint32_t rng_state = 0x243f6a88u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
                            rng_state ^= rng_state << 5; return rng_state; }
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL %s\n", what); fails++; }
}

static double rel_rms(const float *got, const float *want, size_t n) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        se += d * d; sr += (double)want[i] * (double)want[i];
    }
    return sr > 0 ? sqrt(se / sr) : sqrt(se);
}

static void shape(int I, int O, int S) {
    char name[160];
    size_t wn = (size_t)I * O, yn = (size_t)S * O;
    int8_t *w = (int8_t *)malloc(wn);
    float *sc = (float *)malloc((size_t)O * sizeof(float));
    float *x = (float *)malloc((size_t)S * I * sizeof(float));
    float *y0 = (float *)malloc(yn * sizeof(float));
    float *yr = (float *)malloc(yn * sizeof(float));
    float *want = (float *)malloc(yn * sizeof(float));
    if (!w || !sc || !x || !y0 || !yr || !want) { printf("FAIL alloc %dx%d\n", O, I); fails++; return; }

    for (size_t i = 0; i < wn; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.001f + (float)(rnd() % 100) / 50000.0f;
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();
    for (int s = 0; s < S; s++) cpu_gemv_i8(w, sc, I, O, x + (size_t)s * I, want + (size_t)s * O);

    ColiCudaTensor *t = NULL;
    setenv("COLI_CUDA_I8_ROWS", "0", 1);
    snprintf(name, sizeof name, "original ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, y0, x, w, sc, 1, S, I, O, 0, 0), name);

    /* The reference underneath: identity is worthless if both are wrong. */
    double r0 = rel_rms(y0, want, yn);
    snprintf(name, sizeof name, "original within reference bound [%d x %d]", O, I);
    check(r0 < 1e-5, name);

    printf("  [%5d x %5d] S=%d  ref rms %.2e |", O, I, S, r0);
    for (int R = 2; R <= 8; R *= 2) {
        char v[4]; snprintf(v, sizeof v, "%d", R);
        setenv("COLI_CUDA_I8_ROWS", v, 1);
        snprintf(name, sizeof name, "R=%d ran [%d x %d] S=%d", R, O, I, S);
        check(coli_cuda_matmul(&t, yr, x, w, sc, 1, S, I, O, 0, 0), name);
        int same = !memcmp(y0, yr, yn * sizeof(float));
        printf("  R=%d %s", R, same ? "bitwise" : "DIFFERS");
        snprintf(name, sizeof name, "R=%d bitwise identical to the original [%d x %d] S=%d", R, O, I, S);
        check(same, name);
    }
    printf("\n");

    /* Proof of bite, on the blocked kernel: identity between two kernels that
     * read nothing would pass every check above. */
    setenv("COLI_CUDA_I8_ROWS", "8", 1);
    int probe = I > 3 ? 3 : 0;
    w[probe] = (int8_t)(w[probe] == 127 ? -127 : w[probe] + 1);
    coli_cuda_tensor_free(t); t = NULL;
    float *ym = (float *)malloc(yn * sizeof(float));
    if (ym) {
        check(coli_cuda_matmul(&t, ym, x, w, sc, 1, S, I, O, 0, 0), "mutated run");
        snprintf(name, sizeof name, "a mutated weight moves the output [%d x %d]", O, I);
        check(memcmp(ym, yr, yn * sizeof(float)) != 0, name);
        /* and twice in a row must agree with itself */
        float *again = (float *)malloc(yn * sizeof(float));
        if (again) {
            check(coli_cuda_matmul(&t, again, x, w, sc, 1, S, I, O, 0, 0), "determinism run");
            snprintf(name, sizeof name, "R=8 determinism bitwise [%d x %d]", O, I);
            check(!memcmp(again, ym, yn * sizeof(float)), name);
            free(again);
        }
        free(ym);
    }
    coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(y0); free(yr); free(want);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("FAIL cuda init\n"); return 1; }

    /* FIRST, and before any identity is claimed: does the branch fire at all?
     * Bitwise identity is satisfied by a dispatch that never fires -- the same
     * kernel twice is trivially identical -- so every check below is worthless
     * without this one. It cost a whole measurement session to learn that the
     * hard way: a four-point sweep came back flat, and nothing in this file
     * could say whether that meant the hypothesis was wrong or the kernel had
     * never run. */
    {
        int I = 2048, O = 64;
        int8_t *w = (int8_t *)malloc((size_t)I * O);
        float *sc = (float *)malloc((size_t)O * sizeof(float));
        float *x = (float *)malloc((size_t)I * sizeof(float));
        float *y = (float *)malloc((size_t)O * sizeof(float));
        if (w && sc && x && y) {
            for (size_t i = 0; i < (size_t)I * O; i++) w[i] = (int8_t)(rnd() % 127);
            for (int o = 0; o < O; o++) sc[o] = 0.01f;
            for (int i = 0; i < I; i++) x[i] = rndf();
            ColiCudaTensor *t = NULL;

            setenv("COLI_CUDA_I8_ROWS", "0", 1);
            check(i8_rows_mode() == 0, "ROWS=0 parses as off");
            uint64_t before = g_i8r_launches;
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, ROWS=0");
            check(g_i8r_launches == before, "ROWS=0 must NOT take the row-blocked branch");

            setenv("COLI_CUDA_I8_ROWS", "8", 1);
            check(i8_rows_mode() == 8, "ROWS=8 parses as 8");
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, ROWS=8");
            check(g_i8r_launches == before + 1, "ROWS=8 MUST take the row-blocked branch");

            /* The default is ON at R=COLI_I8_ROWS_DEFAULT, so "off" now has to
             * be asked for: an unset variable must dispatch, or the shipped
             * build is quietly running the old kernel. */
            setenv("COLI_CUDA_I8_ROWS", "", 1);
            check(i8_rows_mode() == COLI_I8_ROWS_DEFAULT, "unset reads as the compiled default");
            before = g_i8r_launches;
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, unset");
            check(g_i8r_launches == before + 1, "unset MUST take the row-blocked branch");

            setenv("COLI_CUDA_I8_ROWS", "3", 1);   /* not instantiated */
            check(i8_rows_mode() == COLI_I8_ROWS_DEFAULT,
                  "an uninstantiated R falls back to the default, not to a neighbour");
            coli_cuda_tensor_free(t);
        } else { printf("FAIL alloc dispatch probe\n"); fails++; }
        free(w); free(sc); free(x); free(y);
    }
    printf("dispatch: the row-blocked branch fires on ROWS=8 and unset, not on ROWS=0\n");

    printf("fmt=1 per-row int8: R rows per block must be BITWISE identical\n");
    shape(2048, 512, 1);   /* trunk geometry, O divisible by 2/4/8 */
    shape(2048, 64, 4);    /* S > 1 */
    shape(2048, 13, 1);    /* O % R != 0 for every R: short last block */
    shape(2048, 5, 1);     /* O < R at R=8: one short block only */
    shape(2050, 17, 1);    /* I not a multiple of the block width, O with a tail */
    shape(96, 3, 1);       /* I < 256: most threads idle, O < R */
    shape(4, 1, 1);        /* degenerate */

    if (fails) { printf("test_int8_rows_cuda: %d failure(s)\n", fails); return 1; }
    printf("test_int8_rows_cuda: ok\n");
    return 0;
}
