/* test_i8_acc512.c — the AVX-512 int8 x f32 dot against the scalar reference.
 *
 * The kernel (simd_i8f.h) replaces the AVX2 inner loop of qwen36's int8 GEMVs on
 * an AVX-512 host, so what has to hold is the arithmetic, not bit-identity: the
 * two summation orders differ by construction. This checks every length the
 * callers pass, including the ones below 32 and the non-multiples of 32 that
 * exercise the scalar tail.
 *
 * Two guards, both needed. The file is compiled with -mavx512f -mavx512bw on
 * x86-64 even when the engine is not, so the kernel is verified on every x86
 * builder instead of only on the ones whose default -march happens to include
 * AVX-512. And the binary must still RUN on a host without the instructions --
 * most CI runners -- so the selftest is entered only behind a CPUID check, and
 * reports a skip otherwise rather than dying on SIGILL. */
#include <stdio.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include "../simd_i8f.h"

int main(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw")) {
        puts("i8 acc512: skipped (host has no AVX-512)");
        return 0;
    }
#endif
    if (!i8_acc512_selftest()) {
        fprintf(stderr, "i8 acc512: FAILED\n");
        return 1;
    }
    puts("i8 acc512: ok");
    return 0;
}

#else
int main(void) {
    puts("i8 acc512: skipped (built without AVX-512)");
    return 0;
}
#endif
