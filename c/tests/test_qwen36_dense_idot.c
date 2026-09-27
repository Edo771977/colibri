/* COLI_DENSE_IDOT: matmul_d with int8 activations. Model-free.
 *
 * Pins:
 *  - the flag is off unless COLI_DENSE_IDOT=1, and off means the dispatch of
 *    before, byte for byte (matmul_q per row, matmul_q_batch for a batch);
 *  - dense_act_i8 gives the scale and bytes of quant.h's qrow_i8 (its
 *    scalar contract is copied below: this file includes qwen36.c, which
 *    cannot include quant.h), with ties at .5, tails shorter than a vector
 *    and an all-zero row;
 *  - qwen_dot_i8i8 equals a plain int32 loop over [-127, 127], every length;
 *  - on, matmul_d gives the bytes of that quantizer + an int32 dot scaled as
 *    quant.h's matmul_q_idot does, one row and a batch, and each batch row
 *    equals that row alone;
 *  - on, it stays within 3% of a double-precision reference;
 *  - a matrix the registry does not hold still goes to f32 matmul. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }

static unsigned g_seed = 4242;
static float rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return ((g_seed >> 8) & 0xFFFF) / 32768.f - 1.f; }

/* quant.h's qrow_i8, verbatim */
static float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
static int32_t ref_dot(const int8_t *w, const int8_t *x, int I) {
    int32_t s = 0; for (int i = 0; i < I; i++) s += (int32_t)w[i] * x[i]; return s;
}
/* quant.h's matmul_q_idot, scalar: (float)dot * weight scale * activation scale */
static void ref_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q, const float *sc, int S, int I, int O) {
    for (int o = 0; o < O; o++) for (int s = 0; s < S; s++)
        y[(int64_t)s * O + o] = (float)ref_dot(q + (int64_t)o * I, xq + (int64_t)s * I, I) * sc[o] * sx[s];
}

static int flag_with(const char *v) {
    if (v) setenv("COLI_DENSE_IDOT", v, 1); else unsetenv("COLI_DENSE_IDOT");
    g_dense_idot = -1;
    return dense_idot_on();
}

static int same_quant(const float *x, int I) {
    int8_t *a = malloc((size_t)I + 1), *b = malloc((size_t)I + 1);
    float sa = dense_act_i8(x, I, a), sb = qrow_i8(x, b, I);
    int ok = sa == sb && !memcmp(a, b, (size_t)I);
    free(a); free(b);
    return ok;
}

static double rel_gap(const float *a, const double *ref, int n) {
    double worst = 0, scale = 1e-6;
    for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - ref[i]); if (d > worst) worst = d; if (fabs(ref[i]) > scale) scale = fabs(ref[i]); }
    return worst / scale;
}

static void one_shape(int S, int I, int O) {
    char what[160];
    float *W = falloc((int64_t)O * I), *x = falloc((int64_t)S * I);
    float *y = falloc((int64_t)S * O), *want = falloc((int64_t)S * O), *y1 = falloc(O);
    double *ref = malloc(sizeof(double) * (size_t)S * O);
    for (int64_t i = 0; i < (int64_t)O * I; i++) W[i] = rnd();
    for (int64_t i = 0; i < (int64_t)S * I; i++) x[i] = rnd() * 2.f;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0; for (int i = 0; i < I; i++) a += (double)x[(int64_t)s * I + i] * W[(int64_t)o * I + i];
        ref[(int64_t)s * O + o] = a;
    }
    g_qdw_n = 0;
    qdw_register(W, I, O);
    if (g_qdw_n != 1) { ck(0, "matrix registered"); return; }
    const int8_t *q = g_qdw[0].q; const float *sc = g_qdw[0].sc;

    /* off: the dispatch of before */
    flag_with(NULL);
    matmul_d(y, x, W, S, I, O);
    if (S > 1 && dense_batch_on()) matmul_q_batch(want, x, q, sc, S, I, O);
    else for (int s = 0; s < S; s++) matmul_q(want + (int64_t)s * O, x + (int64_t)s * I, q, sc, I, O);
    snprintf(what, sizeof what, "S=%d I=%d O=%d off: same bytes as the classic int8 path", S, I, O);
    ck(!memcmp(y, want, sizeof(float) * (size_t)S * O), what);

    /* on: qrow_i8 + matmul_q_idot, byte for byte */
    flag_with("1");
    matmul_d(y, x, W, S, I, O);
    int8_t *xq = malloc((size_t)S * I); float *sx = falloc(S);
    for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s * I, xq + (int64_t)s * I, I);
    ref_idot(want, xq, sx, q, sc, S, I, O);
    snprintf(what, sizeof what, "S=%d I=%d O=%d on: same bytes as qrow_i8 + an int32 dot, scaled", S, I, O);
    ck(!memcmp(y, want, sizeof(float) * (size_t)S * O), what);
    int rows_ok = 1;
    for (int s = 0; s < S; s++) {
        matmul_d(y1, x + (int64_t)s * I, W, 1, I, O);
        if (memcmp(y1, y + (int64_t)s * O, sizeof(float) * (size_t)O)) rows_ok = 0;
    }
    snprintf(what, sizeof what, "S=%d I=%d O=%d on: every batch row equals that row alone", S, I, O);
    ck(rows_ok, what);
    double gap = rel_gap(y, ref, S * O);
    snprintf(what, sizeof what, "S=%d I=%d O=%d on: within 3%% of the double reference (%.4f)", S, I, O, gap);
    ck(gap < 3e-2, what);

    free(W); free(x); free(y); free(want); free(y1); free(ref); free(xq); free(sx);
    free(g_qdw[0].q); free(g_qdw[0].sc); g_qdw_n = 0;
}

int main(void) {
    printf("flag\n");
    ck(flag_with(NULL) == 0, "unset: off");
    ck(flag_with("0") == 0, "COLI_DENSE_IDOT=0: off");
    ck(flag_with("true") == 0, "COLI_DENSE_IDOT=true: off (only 1 turns it on)");
    ck(flag_with("1") == 1, "COLI_DENSE_IDOT=1: on");

    printf("activation quantizer\n");
    {
        static const int lens[] = { 1, 7, 8, 31, 32, 33, 63, 255, 512, 2048, 2061 };
        int ok = 1;
        for (size_t li = 0; li < sizeof lens / sizeof *lens; li++) {
            int I = lens[li];
            float *x = falloc(I);
            for (int t = 0; t < 20; t++) {
                float mag = (t % 5 == 0) ? 1e-3f : (t % 5 == 1) ? 1e3f : 3.f;
                for (int i = 0; i < I; i++) x[i] = rnd() * mag;
                if (!same_quant(x, I)) ok = 0;
            }
            for (int i = 0; i < I; i++) x[i] = 0.f;
            if (!same_quant(x, I)) ok = 0;
            free(x);
        }
        ck(ok, "same scale and bytes as qrow_i8: 11 lengths, 3 magnitudes, all-zero rows");
        /* amax 127 makes the scale 1: x*inv is x itself, so the .5 values are
         * exact ties and round to even in both paths */
        float x[64]; x[0] = 127.f;
        for (int i = 1; i < 64; i++) x[i] = (float)((i % 9) - 4) + 0.5f;
        int8_t a[64], b[64];
        dense_act_i8(x, 64, a); qrow_i8(x, b, 64);
        /* x[1..6] = -2.5 -1.5 -0.5 0.5 1.5 2.5 */
        ck(!memcmp(a, b, 64) && a[1] == -2 && a[2] == -2 && a[3] == 0 && a[4] == 0 && a[5] == 2 && a[6] == 2,
           "ties at .5 round to even in both paths");
    }

    printf("integer dot\n");
    {
        int ok = 1;
        int8_t w[300], x[300];
        for (int I = 0; I <= 300; I++) {
            for (int i = 0; i < I; i++) { w[i] = (int8_t)(((int)(rnd() * 32768.f) % 255 + 255) % 255 - 127); x[i] = (int8_t)(((int)(rnd() * 32768.f) % 255 + 255) % 255 - 127); }
            if (qwen_dot_i8i8(w, x, I) != ref_dot(w, x, I)) ok = 0;
        }
        for (int i = 0; i < 256; i++) { w[i] = (i & 1) ? -127 : 127; x[i] = (i & 2) ? -127 : 127; }
        if (qwen_dot_i8i8(w, x, 256) != ref_dot(w, x, 256)) ok = 0;
        for (int i = 0; i < 256; i++) { w[i] = -127; x[i] = -127; }
        if (qwen_dot_i8i8(w, x, 256) != 256 * 127 * 127) ok = 0;
        ck(ok, "qwen_dot_i8i8 equals the int32 loop, lengths 0..300 and the +-127 extremes");
    }

    printf("matmul_d dispatch\n");
    one_shape(1, 2048, 512);
    one_shape(1, 512, 2048);
    one_shape(5, 2048, 96);
    one_shape(3, 257, 33);

    printf("unregistered matrix\n");
    {
        enum { I = 64, O = 16 };
        float W[I * O], x[I], y[O], want[O];
        for (int i = 0; i < I * O; i++) W[i] = rnd();
        for (int i = 0; i < I; i++) x[i] = rnd();
        g_qdw_n = 0;
        flag_with("1");
        matmul_d(y, x, W, 1, I, O);
        matmul(want, x, W, 1, I, O);
        ck(!memcmp(y, want, sizeof y), "on, not in the registry: f32 matmul, same bytes");
    }

    unsetenv("COLI_DENSE_IDOT");
    if (fails) { printf("test_qwen36_dense_idot: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_dense_idot: COLI_DENSE_IDOT off keeps the old bytes, on is qrow_i8 + an exact int32 dot\n");
    return 0;
}
