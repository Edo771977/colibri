/* simd_i8f.h — AVX-512 int8 x f32 dot product, the 512-bit twin of the AVX2
 * inner loop the int8 GEMV kernels have always had.
 *
 * Why it exists. The kernels that multiply an int8 weight row by a float
 * activation row -- qwen36's dense matmul_q, its grouped expert sibling
 * matmul_q_gs, and quant.h's matmul_q -- were written for AVX2 and never grew a
 * 512-bit form, so on an AVX-512 part (Zen 4/5, Xeon SP) they used half the
 * register width and half the FMA throughput. On the box this was written for
 * (Ryzen 9 7950X + RTX 4070 Ti SUPER, Qwen3.6-35B-A3B int4 with the CUDA expert
 * tier) those kernels carry most of a decode: of 56.4 ms/token, the shared
 * expert is 9.3, the DeltaNet output projection 9.0 and the attention core 10.5,
 * all of them CPU (COLI_TIMERS=1, 128-token run).
 *
 * Numerics. Two 512-bit accumulators reduced with _mm512_reduce_add_ps is a
 * different summation ORDER from the AVX2 kernel's four 256-bit accumulators and
 * its shuffle tree, so the last bits differ -- the same tradeoff I4_ACC512
 * documents for dot_i4f_avx512 in quant.h, and the reason every caller keeps a
 * runtime switch instead of silently changing what the oracle gates measure.
 * Both are float accumulation of the same products; neither is "the" reference.
 *
 * The scalar tail is inside the kernel on purpose: callers pass group lengths
 * (gs=64) and row lengths that are not always a multiple of 32, and a tail the
 * caller writes twice is a tail that drifts. */
#ifndef COLI_SIMD_I8F_H
#define COLI_SIMD_I8F_H

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static inline float dot_i8f_avx512(const int8_t *w, const float *x, int n)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m128i b0 = _mm_loadu_si128((const __m128i *)(w + i));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(w + i + 16));
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i),
                             _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0)), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16),
                             _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1)), a1);
    }
    float acc = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; i < n; i++) acc += x[i] * (float)w[i];
    return acc;
}

/* Same contract as quant.h's i4_acc512_selftest: every length the kernel can be
 * handed, against the scalar reference it replaces. Lengths below 32 and
 * non-multiples of 32 are included so the tail is covered, and the tolerance is
 * the reassociation budget, not an exactness claim. */
static int i8_acc512_selftest(void)
{
    enum { N = 200 };
    int8_t w[N];
    float x[N];
    for (int i = 0; i < N; i++) {
        w[i] = (int8_t)(((i * 37 + 11) % 255) - 127);
        x[i] = (float)(((i * 29 + 7) % 101) - 50) / 37.f;
    }
    for (int n = 1; n <= N; n++) {
        float ref = 0;
        for (int i = 0; i < n; i++) ref += x[i] * (float)w[i];
        float got = dot_i8f_avx512(w, x, n), tol = 2e-5f * (1.f + fabsf(ref));
        if (fabsf(got - ref) > tol) {
            fprintf(stderr, "AVX512 i8 selftest n=%d: %.9g != %.9g\n", n, got, ref);
            return 0;
        }
    }
    return 1;
}
#endif /* AVX-512 */

#endif /* COLI_SIMD_I8F_H */
