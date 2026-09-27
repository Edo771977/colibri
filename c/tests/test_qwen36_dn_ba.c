/* Model-free exactness gate for deltanet()'s fused a/b projection.
 *
 * The loader now keeps in_proj_b and in_proj_a in ONE [2*vh, H] buffer (b's
 * rows first) and deltanet() computes both with a single matmul into [b ++ a],
 * where it used to call matmul twice. matmul computes each output row as its
 * own dot product in a fixed order, so the fused call must give the same
 * bytes. This pins that: two separate calls against one call on the
 * concatenated weights, raw float bytes, across head counts (including odd
 * ones and fewer rows than threads), hidden sizes with and without vector
 * tails, and several OpenMP team sizes. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int failures;

static float val(int64_t i, int salt) {
    int v = (int)((i * 37 + salt * 19) % 251);
    return (float)(v - 125) / (float)(97 + salt);
}

static void one_shape(int vh, int H, int threads) {
    float *x = falloc(H), *wb = falloc((int64_t)vh * H), *wa = falloc((int64_t)vh * H);
    float *wba = falloc(2 * (int64_t)vh * H);
    float *b = falloc(vh), *a = falloc(vh), *ba = falloc(2 * (int64_t)vh);
    for (int i = 0; i < H; i++) x[i] = val(i, 1);
    for (int64_t i = 0; i < (int64_t)vh * H; i++) { wb[i] = val(i, 2) * 0.01f; wa[i] = val(i, 3) * 0.02f; }
    /* The loader's layout: b rows, then a rows. */
    memcpy(wba, wb, (size_t)vh * H * sizeof(float));
    memcpy(wba + (int64_t)vh * H, wa, (size_t)vh * H * sizeof(float));
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    matmul(b, x, wb, 1, H, vh);
    matmul(a, x, wa, 1, H, vh);
    matmul(ba, x, wba, 1, H, 2 * vh);
    if (memcmp(b, ba, (size_t)vh * sizeof(float)) || memcmp(a, ba + vh, (size_t)vh * sizeof(float))) {
        fprintf(stderr, "FAIL vh=%d H=%d threads=%d: fused a/b differs from two matmuls\n", vh, H, threads);
        failures++;
    }
    free(x); free(wb); free(wa); free(wba); free(b); free(a); free(ba);
}

int main(void) {
    static const int VH[] = { 1, 3, 16, 32, 48, 64 };
    static const int HS[] = { 7, 64, 2048, 2051 };
    static const int TH[] = { 1, 4, 16, 33 };
    int n = 0;
    for (size_t i = 0; i < sizeof VH / sizeof VH[0]; i++)
        for (size_t j = 0; j < sizeof HS / sizeof HS[0]; j++)
            for (size_t k = 0; k < sizeof TH / sizeof TH[0]; k++) { one_shape(VH[i], HS[j], TH[k]); n++; }
    if (failures) { fprintf(stderr, "test_qwen36_dn_ba: %d of %d shapes FAILED\n", failures, n); return 1; }
    printf("OK test_qwen36_dn_ba: fused a/b matmul byte-identical to two matmuls (%d shapes)\n", n);
    return 0;
}
