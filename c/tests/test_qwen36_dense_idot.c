/* The dense trunk's integer path (COLI_DENSE_IDOT=1) and the experts'
 * activation mode (QWEN_EXPERT_ACT=int8), ported from upstream dfec3a4b
 * without its COLI_DENSE_BITS=4 option, and opt-in here.
 *
 * Pins, with no model:
 *  - dense_act_i8's vector path equals the scalar qrow_i8 contract bit for
 *    bit (same rounding, same scale) and its block sums are exact;
 *  - by default, and with COLI_DENSE_IDOT=0, matmul_d is the path before the
 *    port: byte-identical to matmul_q per row and to matmul_q_batch for a
 *    batch;
 *  - with COLI_DENSE_IDOT=1 matmul_d answers from the integer kernel:
 *    byte-identical to matmul_q_idot on the registry's own int8 rows, within
 *    3% of the f32 reference, and every row of a prompt batch equals the
 *    same row computed alone (prefill and decode see the same numbers);
 *  - xf_act_mode is 0 by default and 1 under QWEN_EXPERT_ACT=int8.
 * Every flag is read once per process, so each arm but the default runs in a
 * child. That moe_xf_run passes xf_act_mode() to the kernel is checked
 * end to end by CI (the tiny int4 gs64 fixture: marker line, logits move). */
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

/* The path before the port: matmul_q per row, matmul_q_batch for a batch. */
static int check_classic(void) {
    setup();
    if (g_qdw_n != 1) return 90;
    float *y = malloc((size_t)S * O * sizeof(float)), *r = malloc((size_t)S * O * sizeof(float));
    int rc = 0;
    matmul_d(y, g_x, g_W, 1, I, O);
    matmul_q(r, g_x, g_qdw[0].q, g_qdw[0].sc, I, O);
    if (memcmp(y, r, (size_t)O * sizeof(float))) rc = 91;
    matmul_d(y, g_x, g_W, S, I, O);
    matmul_q_batch(r, g_x, g_qdw[0].q, g_qdw[0].sc, S, I, O);
    if (!rc && memcmp(y, r, (size_t)S * O * sizeof(float))) rc = 92;
    free(y); free(r);
    return rc;
}
/* --child idot: COLI_DENSE_IDOT=1 is set by the parent. */
static int child_idot(void) {
    setup();
    if (g_qdw_n != 1) return 80;
    float *ref = malloc((size_t)S * O * sizeof(float)), *y = malloc((size_t)S * O * sizeof(float));
    float *k = malloc((size_t)S * O * sizeof(float)), *y1 = malloc((size_t)O * sizeof(float));
    ref_matmul(ref, g_x, g_W, S, I, O);
    matmul_d(y, g_x, g_W, S, I, O);
    int8_t *xq = malloc((size_t)S * I); float sx[S];
    for (int s = 0; s < S; s++) sx[s] = qrow_i8(g_x + (size_t)s * I, xq + (size_t)s * I, I);
    matmul_q_idot(k, xq, sx, g_qdw[0].q, g_qdw[0].sc, S, I, O);
    if (memcmp(y, k, (size_t)S * O * sizeof(float))) return 81;     /* not the integer kernel */
    if (!(rel_gap(y, ref, S * O) < 3e-2)) return 82;                  /* too far from f32 */
    for (int s = 0; s < S; s++) {
        matmul_d(y1, g_x + (size_t)s * I, g_W, 1, I, O);
        if (memcmp(y1, y + (size_t)s * O, (size_t)O * sizeof(float))) return 83;   /* batch row != row alone */
    }
    return 0;
}
/* --child classic: COLI_DENSE_IDOT=0 is set by the parent. */
static int child_classic(void) { return check_classic(); }
/* --child expert-int8: QWEN_EXPERT_ACT=int8 is set by the parent. */
static int child_expert_int8(void) { return xf_act_mode() == 1 ? 0 : 93; }

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

static int arm(const char *self, const char *var, const char *val, const char *child) {
    setenv(var, val, 1);
    int st = child_status(self, child);
    unsetenv(var);
    if (st) printf("  child %s exit %d\n", child, st);
    return st;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) {
        if (!strcmp(argv[2], "idot")) return child_idot();
        if (!strcmp(argv[2], "classic")) return child_classic();
        if (!strcmp(argv[2], "expert-int8")) return child_expert_int8();
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

    printf("defaults (no flag set)\n");
    ck(check_classic() == 0, "matmul_d is the f32-activation path: matmul_q per row, matmul_q_batch for a batch, byte for byte");
    ck(xf_act_mode() == 0, "experts keep f32 activations (xf_act_mode 0)");

    printf("the switches (child processes: each flag is read once)\n");
    ck(arm(argv[0], "COLI_DENSE_IDOT", "1", "idot") == 0,
       "COLI_DENSE_IDOT=1: matmul_d equals matmul_q_idot on the registry's rows byte for byte, within 3% of f32, batch rows equal rows alone");
    ck(arm(argv[0], "COLI_DENSE_IDOT", "0", "classic") == 0, "COLI_DENSE_IDOT=0: the f32-activation path, byte for byte");
    ck(arm(argv[0], "QWEN_EXPERT_ACT", "int8", "expert-int8") == 0, "QWEN_EXPERT_ACT=int8: xf_act_mode 1");

    if (fails) { printf("test_qwen36_dense_idot: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_dense_idot: the dense trunk's integer path\n");
    return 0;
}
