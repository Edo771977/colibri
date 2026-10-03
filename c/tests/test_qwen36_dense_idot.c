/* The dense trunk's integer path (COLI_DENSE_IDOT) and the experts'
 * activation mode (QWEN_EXPERT_ACT), ported from upstream dfec3a4b without
 * its COLI_DENSE_BITS=4 option.
 *
 * Pins, with no model:
 *  - dense_act_i8's vector path equals the scalar qrow_i8 contract bit for
 *    bit (same rounding, same scale) and its block sums are exact;
 *  - with COLI_DENSE_IDOT unset, matmul_d answers from the integer kernel:
 *    byte-identical to matmul_q_idot on the registry's own int8 rows, within
 *    3% of the f32 reference, and every row of a prompt batch equals the
 *    same row computed alone (prefill and decode see the same numbers);
 *  - with COLI_DENSE_IDOT=0 (a child process: the flag is read once) matmul_d
 *    is the path before the port, byte-identical to matmul_q per row and to
 *    matmul_q_batch for a batch;
 *  - xf_act_mode is 1 by default and 0 under QWEN_EXPERT_ACT=f32. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include "../compat.h"   /* setenv/unsetenv: MinGW has none */

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }

static unsigned g_seed = 4242;
static float rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return ((g_seed >> 8) & 0xFFFF) / 32768.f - 1.f; }

static void ref_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0; for (int i = 0; i < I; i++) a += (double)x[(size_t)s * I + i] * W[(size_t)o * I + i];
        y[(size_t)s * O + o] = (float)a;
    }
}
static double rel_gap(const float *a, const float *b, int n) {
    double worst = 0, scale = 1e-6;
    for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > worst) worst = d; if (fabs((double)b[i]) > scale) scale = fabs((double)b[i]); }
    return worst / scale;
}

/* I = 256 + 40: vector body plus a scalar tail in both the quantizer and
 * the dot; O odd so the last output row is not a multiple of anything. */
enum { I = 296, O = 97, S = 5 };

/* One registered matrix and a batch of activations, the same in every process
 * (fixed seed), so parent and children compare the same numbers. */
static float *g_W, *g_x;
static void setup(void) {
    g_seed = 4242;
    g_W = malloc((size_t)O * I * sizeof(float)); g_x = malloc((size_t)S * I * sizeof(float));
    for (size_t i = 0; i < (size_t)O * I; i++) g_W[i] = rnd();
    for (size_t i = 0; i < (size_t)S * I; i++) g_x[i] = rnd() * 2.f;
    qdw_register(g_W, I, O);
}

/* --child classic: COLI_DENSE_IDOT=0 is set by the parent. */
static int child_classic(void) {
    setup();
    if (g_qdw_n != 1) return 90;
    float *y = malloc((size_t)S * O * sizeof(float)), *r = malloc((size_t)S * O * sizeof(float));
    matmul_d(y, g_x, g_W, 1, I, O);
    matmul_q(r, g_x, g_qdw[0].q, g_qdw[0].sc, I, O);
    if (memcmp(y, r, (size_t)O * sizeof(float))) return 91;
    matmul_d(y, g_x, g_W, S, I, O);
    matmul_q_batch(r, g_x, g_qdw[0].q, g_qdw[0].sc, S, I, O);
    if (memcmp(y, r, (size_t)S * O * sizeof(float))) return 92;
    return 0;
}
/* --child expert-f32: QWEN_EXPERT_ACT=f32 is set by the parent. */
static int child_expert_f32(void) { return xf_act_mode() == 0 ? 0 : 93; }

static int child_status(const char *self, const char *arm) {
    char cmd[1536];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "call \"%s\" --child %s", self, arm);
#else
    snprintf(cmd, sizeof(cmd), "\"%s\" --child %s", self, arm);
#endif
    int status = system(cmd);
#ifdef _WIN32
    return status;
#else
    return status >= 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) {
        if (!strcmp(argv[2], "classic")) return child_classic();
        if (!strcmp(argv[2], "expert-f32")) return child_expert_f32();
        return 99;
    }
    /* This gate owns the flags regardless of the caller's shell. */
    unsetenv("COLI_DENSE_IDOT"); unsetenv("QWEN_EXPERT_ACT"); unsetenv("COLI_DENSE_I8");

    printf("activation quantizer\n");
    {
        float x[I]; int8_t a[I], b[I]; int32_t sums[I / 64];
        int bad = 0;
        for (int t = 0; t < 50 && !bad; t++) {
            float mag = (t % 5 == 0) ? 1e-3f : 3.f;
            for (int i = 0; i < I; i++) x[i] = rnd() * mag;
            float sa = dense_act_i8(x, I, a, sums);
            float sb = qrow_i8(x, b, I);
            if (sa != sb || memcmp(a, b, I)) bad = 1;
            for (int g = 0; g < I / 64 && !bad; g++) { int32_t s = 0; for (int k = 0; k < 64; k++) s += a[g * 64 + k]; if (s != sums[g]) bad = 2; }
        }
        ck(bad != 1, "vector quantizer equals the scalar qrow_i8 contract on 50 random rows");
        ck(bad != 2, "block sums are exact");
    }

    printf("matmul_d, default (integer dot)\n");
    {
        setup();
        ck(g_qdw_n == 1, "the matrix is in the int8 registry");
        float *ref = malloc((size_t)S * O * sizeof(float)), *y = malloc((size_t)S * O * sizeof(float));
        float *k = malloc((size_t)S * O * sizeof(float)), *y1 = malloc((size_t)O * sizeof(float));
        ref_matmul(ref, g_x, g_W, S, I, O);
        matmul_d(y, g_x, g_W, S, I, O);
        int8_t *xq = malloc((size_t)S * I); float sx[S];
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(g_x + (size_t)s * I, xq + (size_t)s * I, I);
        matmul_q_idot(k, xq, sx, g_qdw[0].q, g_qdw[0].sc, S, I, O);
        ck(!memcmp(y, k, (size_t)S * O * sizeof(float)), "matmul_d equals matmul_q_idot on the registry's rows, byte for byte");
        ck(rel_gap(y, ref, S * O) < 3e-2, "within 3% of the f32 reference (int8 weights x int8 activations), 5 rows");
        int same = 1;
        for (int s = 0; s < S; s++) {
            matmul_d(y1, g_x + (size_t)s * I, g_W, 1, I, O);
            if (memcmp(y1, y + (size_t)s * O, (size_t)O * sizeof(float))) same = 0;
        }
        ck(same, "every row of the batch equals the same row computed alone");
        ck(xf_act_mode() == 1, "experts default to int8 activations (xf_act_mode 1)");
        free(ref); free(y); free(k); free(y1); free(xq);
    }

    printf("the switches (child processes: each flag is read once)\n");
    setenv("COLI_DENSE_IDOT", "0", 1);
    int st = child_status(argv[0], "classic");
    unsetenv("COLI_DENSE_IDOT");
    if (st) printf("  child classic exit %d\n", st);
    ck(st == 0, "COLI_DENSE_IDOT=0: matmul_d equals matmul_q per row and matmul_q_batch for a batch, byte for byte");
    setenv("QWEN_EXPERT_ACT", "f32", 1);
    st = child_status(argv[0], "expert-f32");
    unsetenv("QWEN_EXPERT_ACT");
    if (st) printf("  child expert-f32 exit %d\n", st);
    ck(st == 0, "QWEN_EXPERT_ACT=f32: xf_act_mode 0");

    if (fails) { printf("test_qwen36_dense_idot: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_dense_idot: the dense trunk's integer path\n");
    return 0;
}
