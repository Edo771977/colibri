/* COLI_CUDA_DENSE_PINNED: pinned host staging for the resident dense GEMV.
 *
 * The toggle changes WHERE the bytes come from and HOW MANY TIMES the calling
 * thread blocks, and nothing else:
 *
 *   0  two synchronous cudaMemcpy straight off the caller's pageable buffers
 *   1  the same two synchronous copies, through pinned staging
 *   2  the same two copies async on stream 0, ONE cudaStreamSynchronize
 *
 * The kernel, its arguments, its grid and the order of every addition inside
 * it are untouched. So the contract is BITWISE IDENTITY, not a tolerance -- a
 * staging buffer that changed a logit's last bit would be a bug, not a
 * rounding difference, and memcmp is the only gate that says so.
 *
 * What this file is really guarding, though, is mode 2. Async copies plus a
 * buffer the caller reads afterwards is exactly the shape of bug that does not
 * fail loudly: miss the synchronize and the output is whatever host_dy held
 * from the PREVIOUS call, which for a repeated shape is usually almost right.
 * Hence the interleaving in mixed_order(): every mode-2 result is compared
 * against a mode-0 result for a DIFFERENT input than the one before it, so a
 * missing sync shows up as the previous answer rather than passing unnoticed.
 *
 * The same class of bug already bit this codebase once, in the expert group:
 * a readback sized with xb instead of yb. shape() therefore includes both a
 * wide-in/narrow-out and a narrow-in/wide-out geometry, where confusing the
 * two lengths cannot go unnoticed.
 *
 *   1. memcmp between modes 0, 1 and 2, on shapes that separate xb from yb.
 *   2. A double-precision CPU reference underneath, so "identical" cannot
 *      mean "all three equally wrong".
 *   3. Proof of bite: a mutated weight must move the output.
 *   4. The parse table, and a dispatch counter -- identity is trivially
 *      satisfied by a toggle that never fires.
 *   5. Descending capacities, so a small call after a large one proves the
 *      staging is used by LENGTH and not by capacity.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_dense_pinned_cuda.cu \
 *            -o tests/test_dense_pinned
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

/* No direct <cuda_runtime.h>: the backend include below pulls the right
 * runtime header for the target, and a direct include breaks the hipcc build
 * that `make gpu-compile` runs this file through. */
#include "../backend_cuda.cu"

#ifdef _WIN32
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
#endif

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL %s\n", what); fails++; }
}

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
                            rng_state ^= rng_state << 5; return rng_state; }
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

static void set_mode(int m) {
    char v[4]; snprintf(v, sizeof v, "%d", m);
    setenv("COLI_CUDA_DENSE_PINNED", v, 1);
}

static void cpu_gemv_i8(const int8_t *w, const float *sc, int I, int O,
                        const float *x, float *y) {
    for (int o = 0; o < O; o++) {
        double a = 0.0;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)w[(size_t)o * I + i];
        y[o] = (float)(a * (double)sc[o]);
    }
}

static double rel_rms(const float *got, const float *want, size_t n) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        se += d * d; sr += (double)want[i] * (double)want[i];
    }
    return sr > 0 ? sqrt(se / sr) : sqrt(se);
}

static void shape(int I, int O, int S, const char *why) {
    char name[192];
    size_t wn = (size_t)I * O, yn = (size_t)S * O;
    int8_t *w = (int8_t *)malloc(wn);
    float *sc = (float *)malloc((size_t)O * sizeof(float));
    float *x  = (float *)malloc((size_t)S * I * sizeof(float));
    float *y0 = (float *)malloc(yn * sizeof(float));
    float *y1 = (float *)malloc(yn * sizeof(float));
    float *y2 = (float *)malloc(yn * sizeof(float));
    float *want = (float *)malloc(yn * sizeof(float));
    if (!w || !sc || !x || !y0 || !y1 || !y2 || !want) {
        printf("FAIL alloc %dx%d\n", O, I); fails++; return; }

    for (size_t i = 0; i < wn; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.001f + (float)(rnd() % 100) / 50000.0f;
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();
    for (int s = 0; s < S; s++)
        cpu_gemv_i8(w, sc, I, O, x + (size_t)s * I, want + (size_t)s * O);

    /* Poison the destinations with a sentinel no GEMV would produce. A mode
     * that writes only part of y -- a readback sized with xb when yb is
     * larger, the bug that already bit the expert group -- then leaves the
     * sentinel in the tail instead of whatever malloc happened to hand over,
     * and memcmp fails on it deterministically rather than by luck. */
    for (size_t i = 0; i < yn; i++) y0[i] = y1[i] = y2[i] = -12345.0f;

    ColiCudaTensor *t = NULL;
    set_mode(0);
    snprintf(name, sizeof name, "pageable ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, y0, x, w, sc, 1, S, I, O, 0, 0), name);

    double r0 = rel_rms(y0, want, yn);
    snprintf(name, sizeof name, "pageable within reference bound [%d x %d]", O, I);
    check(r0 < 1e-5, name);

    uint64_t before = g_dense_pinned_calls;
    set_mode(1);
    snprintf(name, sizeof name, "pinned sync ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, y1, x, w, sc, 1, S, I, O, 0, 0), name);
    set_mode(2);
    snprintf(name, sizeof name, "pinned async ran [%d x %d] S=%d", O, I, S);
    check(coli_cuda_matmul(&t, y2, x, w, sc, 1, S, I, O, 0, 0), name);
    snprintf(name, sizeof name, "both pinned modes dispatched [%d x %d]", O, I);
    check(g_dense_pinned_calls == before + 2, name);

    int s1 = !memcmp(y0, y1, yn * sizeof(float));
    int s2 = !memcmp(y0, y2, yn * sizeof(float));
    printf("  [%5d x %5d] S=%d  xb %7zu  yb %7zu  ref rms %.2e |  mode1 %s  mode2 %s   %s\n",
           O, I, S, (size_t)S * I * sizeof(float), yn * sizeof(float), r0,
           s1 ? "bitwise" : "DIFFERS", s2 ? "bitwise" : "DIFFERS", why);
    snprintf(name, sizeof name, "mode 1 bitwise identical to pageable [%d x %d] S=%d", O, I, S);
    check(s1, name);
    snprintf(name, sizeof name, "mode 2 bitwise identical to pageable [%d x %d] S=%d", O, I, S);
    check(s2, name);

    /* Proof of bite: identity between three paths that read nothing would pass
     * everything above. */
    int probe = I > 3 ? 3 : 0;
    w[probe] = (int8_t)(w[probe] == 127 ? -127 : w[probe] + 1);
    coli_cuda_tensor_free(t); t = NULL;
    float *ym = (float *)malloc(yn * sizeof(float));
    if (ym) {
        set_mode(2);
        check(coli_cuda_matmul(&t, ym, x, w, sc, 1, S, I, O, 0, 0), "mutated run");
        snprintf(name, sizeof name, "a mutated weight moves the output [%d x %d]", O, I);
        check(memcmp(ym, y2, yn * sizeof(float)) != 0, name);
        free(ym);
    }
    coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(y0); free(y1); free(y2); free(want);
}

/* The mode-2 trap, spelled out.
 *
 * Every call reuses the same pinned host_dy. Drop the synchronize and the
 * memcpy back to the caller reads whatever the PREVIOUS call left there --
 * which, for a loop that keeps asking the same question, is the right answer.
 * So this asks a different question every time: the input vector changes on
 * each iteration, the expected answer changes with it, and a stale buffer
 * shows up as the previous iteration's output instead of hiding in agreement.
 *
 * The device work is also made long enough (a wide I) that a genuinely async
 * D2H cannot have completed by the time the host reaches the memcpy without
 * being waited on. */
static void mixed_order(void) {
    const int I = 8192, O = 96, reps = 12;
    int8_t *w = (int8_t *)malloc((size_t)I * O);
    float *sc = (float *)malloc((size_t)O * sizeof(float));
    float *x  = (float *)malloc((size_t)I * sizeof(float));
    float *ref = (float *)malloc((size_t)O * sizeof(float));
    float *got = (float *)malloc((size_t)O * sizeof(float));
    if (!w || !sc || !x || !ref || !got) { printf("FAIL alloc mixed_order\n"); fails++; return; }
    for (size_t i = 0; i < (size_t)I * O; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.002f;
    for (int o = 0; o < O; o++) got[o] = 0.0f;   /* prev[] reads it on rep 0 */

    ColiCudaTensor *t = NULL;
    int stale = 0, wrong = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < I; i++) x[i] = rndf();          /* a NEW question */
        set_mode(0);
        check(coli_cuda_matmul(&t, ref, x, w, sc, 1, 1, I, O, 0, 0), "mixed: pageable");
        float prev[1]; prev[0] = got[0];
        set_mode(2);
        check(coli_cuda_matmul(&t, got, x, w, sc, 1, 1, I, O, 0, 0), "mixed: pinned async");
        if (memcmp(ref, got, (size_t)O * sizeof(float))) {
            wrong++;
            if (r > 0 && got[0] == prev[0]) stale++;
        }
    }
    printf("  interleaved %d x [%d x %d]: %d mismatched%s\n", reps, O, I, wrong,
           stale ? "  <-- and the value was the PREVIOUS call's: missing synchronize" : "");
    check(wrong == 0, "mode 2 answers the question it was asked, every time");
    coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(ref); free(got);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("FAIL cuda init\n"); return 1; }

    /* The parse table first. An unrecognised value must read as OFF rather
     * than rounding to a neighbour: a typo in a measurement script must give
     * the baseline arm, not a third arm nobody meant to run. */
    setenv("COLI_CUDA_DENSE_PINNED", "0", 1);  check(dense_pinned_mode() == 0, "\"0\" is off");
    setenv("COLI_CUDA_DENSE_PINNED", "", 1);   check(dense_pinned_mode() == 0, "unset is off");
    setenv("COLI_CUDA_DENSE_PINNED", "1", 1);  check(dense_pinned_mode() == 1, "\"1\" is pinned sync");
    setenv("COLI_CUDA_DENSE_PINNED", "2", 1);  check(dense_pinned_mode() == 2, "\"2\" is pinned async");
    setenv("COLI_CUDA_DENSE_PINNED", "3", 1);  check(dense_pinned_mode() == 0, "\"3\" is off, not a neighbour");
    setenv("COLI_CUDA_DENSE_PINNED", "2x", 1); check(dense_pinned_mode() == 0, "trailing junk is off");
    setenv("COLI_CUDA_DENSE_PINNED", "yes", 1);check(dense_pinned_mode() == 0, "\"yes\" is off");

    /* And that the branch fires at all. Every identity check below is
     * worthless without this one: the same path run three times is trivially
     * identical to itself, and a flat A/B from a toggle that never reached the
     * binary is indistinguishable from a hypothesis that was wrong. */
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

            set_mode(0);
            uint64_t before = g_dense_pinned_calls;
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, mode 0");
            check(g_dense_pinned_calls == before, "mode 0 must NOT stage through pinned memory");
            set_mode(1);
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, mode 1");
            check(g_dense_pinned_calls == before + 1, "mode 1 MUST stage through pinned memory");
            set_mode(2);
            check(coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0), "dispatch probe, mode 2");
            check(g_dense_pinned_calls == before + 2, "mode 2 MUST stage through pinned memory");
            coli_cuda_tensor_free(t);
        } else { printf("FAIL alloc dispatch probe\n"); fails++; }
        free(w); free(sc); free(x); free(y);
    }
    printf("dispatch: pinned staging fires on 1 and 2, not on 0 or unset\n");

    printf("COLI_CUDA_DENSE_PINNED: staging must not move a single bit\n");
    /* Descending, deliberately: the staging buffers only ever grow, so a small
     * call after a large one runs against oversized capacity. A copy that used
     * the CAPACITY instead of the length would read past the caller's buffer
     * here and nowhere else. */
    shape(4096, 4096, 1, "square, trunk-sized");
    shape(2048,  512, 1, "qwen36 trunk geometry");
    shape(2048,   13, 1, "wide in, narrow out: yb << xb");
    shape(  16, 4096, 1, "narrow in, wide out: yb >> xb");
    shape(2048,   64, 4, "S > 1");
    shape(2050,   17, 3, "I not a multiple of the block width, S > 1");
    shape(   4,    1, 1, "degenerate");

    printf("mode 2 is async: the synchronize is the only thing making it correct\n");
    mixed_order();

    setenv("COLI_CUDA_DENSE_PINNED", "0", 1);
    if (fails) { printf("test_dense_pinned_cuda: %d failure(s)\n", fails); return 1; }
    printf("test_dense_pinned_cuda: ok\n");
    return 0;
}
