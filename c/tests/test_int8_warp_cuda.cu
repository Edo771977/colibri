/* fmt=1 per-row int8 warp GEMV oracle (COLI_CUDA_I8_WARP rework).
 *
 * quant_matmul_i8w is quant_matmul_f8w's shape applied to the format the
 * qwen3.6 dense trunk actually uses: fmt=1 uploaded with gs=0, one f32 scale
 * per row. It is opt-in and OFF by default because it changes the summation
 * order; this test is what a default flip would have to survive first.
 *
 *   1. Both kernels against a double-precision CPU reference, on the trunk's
 *      real shapes and on tails that defeat the vectorised path -- I not a
 *      multiple of 4 (char4 refused), I not a multiple of 128 (partial last
 *      block), I < 128 (one partial block only).
 *   2. Old vs new directly. They are NOT expected to be bit-identical -- the
 *      order change is the point -- so the contract is that both sit inside
 *      the same CPU-reference bound, and their difference is reported so a
 *      regression that widens it is visible rather than merely tolerated.
 *   3. Proof of bite: one mutated weight must move the output. A test that
 *      passes on a kernel that never read the weights is worth nothing.
 *   4. S-invariance, per kernel, BITWISE: rows=8 in one call against 8 calls
 *      of rows=1. The s-tile changes concurrency, never the order within a
 *      dot, so anything else means a lane is reading another lane's block.
 *   5. Determinism: the same launch twice, bitwise. The cross-warp fold runs
 *      in fixed warp order with no atomics, so two identical launches that
 *      disagree have read dsum[] while it was still being written.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_int8_warp_cuda.cu -o tests/test_int8_warp
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

/* Reference in double: the question is whether the kernels agree with the
 * arithmetic, not with each other's rounding. */
static void cpu_gemv_i8(const int8_t *w, const float *sc, int I, int O,
                        const float *x, float *y) {
    for (int o = 0; o < O; o++) {
        double a = 0.0;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)w[(size_t)o * I + i];
        y[o] = (float)(a * (double)sc[o]);
    }
}

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
                            rng_state ^= rng_state << 5; return rng_state; }
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL %s\n", what); fails++; }
}

/* Relative RMS against the reference, with the worst single element named:
 * an RMS alone hides one catastrophically wrong row among thousands. */
static double rel_rms(const float *got, const float *want, size_t n, double *worst) {
    double se = 0, sr = 0; *worst = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        double rel = fabs((double)want[i]) > 1e-6 ? fabs(d) / fabs((double)want[i]) : fabs(d);
        if (rel > *worst) *worst = rel;
        se += d * d; sr += (double)want[i] * (double)want[i];
    }
    return sr > 0 ? sqrt(se / sr) : sqrt(se);
}

/* One shape, both kernels, against the reference and against each other. */
static void shape(int I, int O, int S, double bound) {
    char name[128];
    int8_t *w = (int8_t *)malloc((size_t)I * O);
    float *sc = (float *)malloc((size_t)O * sizeof(float));
    float *x = (float *)malloc((size_t)S * I * sizeof(float));
    float *yo = (float *)malloc((size_t)S * O * sizeof(float));
    float *yn = (float *)malloc((size_t)S * O * sizeof(float));
    float *want = (float *)malloc((size_t)S * O * sizeof(float));
    if (!w || !sc || !x || !yo || !yn || !want) { printf("FAIL alloc %dx%d\n", O, I); fails++; return; }

    for (size_t i = 0; i < (size_t)I * O; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.001f + (float)(rnd() % 100) / 50000.0f;
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();

    for (int s = 0; s < S; s++) cpu_gemv_i8(w, sc, I, O, x + (size_t)s * I, want + (size_t)s * O);

    ColiCudaTensor *t = NULL;
    setenv("COLI_CUDA_I8_WARP", "0", 1);
    snprintf(name, sizeof name, "original kernel ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, yo, x, w, sc, 1, S, I, O, 0, 0), name);
    coli_cuda_tensor_free(t); t = NULL;

    setenv("COLI_CUDA_I8_WARP", "1", 1);
    snprintf(name, sizeof name, "warp kernel ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, yn, x, w, sc, 1, S, I, O, 0, 0), name);

    double wo, wn, wc;
    double ro = rel_rms(yo, want, (size_t)S * O, &wo);
    double rn = rel_rms(yn, want, (size_t)S * O, &wn);
    double rc = rel_rms(yn, yo, (size_t)S * O, &wc);
    printf("  [%6d x %6d] S=%d  original rms %.3e (worst %.3e) | warp rms %.3e (worst %.3e) | between %.3e\n",
           O, I, S, ro, wo, rn, wn, rc);
    snprintf(name, sizeof name, "original within bound [%d x %d]", O, I);
    check(ro < bound, name);
    snprintf(name, sizeof name, "warp within bound [%d x %d]", O, I);
    check(rn < bound, name);

    /* Proof of bite: move one weight the warp path must have read. */
    int probe = I > 5 ? 5 : 0;
    w[probe] = (int8_t)(w[probe] == 127 ? -127 : w[probe] + 1);
    coli_cuda_tensor_free(t); t = NULL;
    float *ym = (float *)malloc((size_t)S * O * sizeof(float));
    if (ym) {
        check(coli_cuda_matmul(&t, ym, x, w, sc, 1, S, I, O, 0, 0), "mutated run");
        int moved = 0;
        for (int s = 0; s < S && !moved; s++) if (ym[(size_t)s * O] != yn[(size_t)s * O]) moved = 1;
        snprintf(name, sizeof name, "one mutated weight moves the output [%d x %d]", O, I);
        check(moved, name);
        free(ym);
    }
    coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(yo); free(yn); free(want);
}

/* rows=S in one call against S calls of rows=1, bitwise, for one kernel. */
static void s_invariance(const char *mode, int I, int O, int S) {
    char name[128];
    int8_t *w = (int8_t *)malloc((size_t)I * O);
    float *sc = (float *)malloc((size_t)O * sizeof(float));
    float *x = (float *)malloc((size_t)S * I * sizeof(float));
    float *ybatch = (float *)malloc((size_t)S * O * sizeof(float));
    float *yone = (float *)malloc((size_t)S * O * sizeof(float));
    if (!w || !sc || !x || !ybatch || !yone) { printf("FAIL alloc s-invariance\n"); fails++; return; }
    for (size_t i = 0; i < (size_t)I * O; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.002f + (float)(rnd() % 50) / 25000.0f;
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();

    setenv("COLI_CUDA_I8_WARP", mode, 1);
    ColiCudaTensor *t = NULL;
    check(coli_cuda_matmul(&t, ybatch, x, w, sc, 1, S, I, O, 0, 0), "s-invariance batched call");
    for (int s = 0; s < S; s++)
        check(coli_cuda_matmul(&t, yone + (size_t)s * O, x + (size_t)s * I, w, sc, 1, 1, I, O, 0, 0),
              "s-invariance single-row call");
    int same = !memcmp(ybatch, yone, (size_t)S * O * sizeof(float));
    snprintf(name, sizeof name, "S-invariance bitwise, COLI_CUDA_I8_WARP=%s", mode);
    check(same, name);

    /* Same launch twice: the fold has fixed warp order and no atomics. */
    float *again = (float *)malloc((size_t)S * O * sizeof(float));
    if (again) {
        check(coli_cuda_matmul(&t, again, x, w, sc, 1, S, I, O, 0, 0), "determinism call");
        snprintf(name, sizeof name, "determinism bitwise, COLI_CUDA_I8_WARP=%s", mode);
        check(!memcmp(again, ybatch, (size_t)S * O * sizeof(float)), name);
        free(again);
    }
    coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(ybatch); free(yone);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("FAIL cuda init\n"); return 1; }

    printf("fmt=1 per-row int8: original vs COLI_CUDA_I8_WARP\n");
    /* The qwen3.6 trunk's own geometry (hidden 2048), then the shapes that
     * defeat the vectorised path one precondition at a time. */
    shape(2048, 512, 1, 1e-5);     /* aligned, full 128-blocks */
    shape(2048, 64, 4, 1e-5);      /* same, S > 1 */
    shape(2050, 32, 1, 1e-5);      /* I % 4 != 0 -> char4 refused everywhere */
    shape(2176, 32, 1, 1e-5);      /* I % 128 != 0 -> vectorised body, guarded tail */
    shape(96, 32, 1, 1e-5);        /* I < 128 -> single partial block */
    shape(4, 8, 1, 1e-5);          /* degenerate: fewer elements than one lane's quad */

    s_invariance("0", 2048, 64, 8);
    s_invariance("1", 2048, 64, 8);

    if (fails) { printf("test_int8_warp_cuda: %d failure(s)\n", fails); return 1; }
    printf("test_int8_warp_cuda: ok\n");
    return 0;
}
