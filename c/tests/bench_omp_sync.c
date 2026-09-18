/* bench_omp_sync — what your OpenMP runtime charges to synchronise a team.
 *
 * Two numbers, and nothing else in the way:
 *
 *   region   entering and leaving a `#pragma omp parallel` whose body is empty
 *   barrier  a `#pragma omp barrier` inside a region that is already open
 *
 * Every decode token of a streamed MoE engine crosses a few hundred of these.
 * At single-digit microseconds they are free; at tens of microseconds they are
 * the largest single item in the token, ahead of any kernel in it. Measured on
 * a Ryzen 9 7950X: 53 us and 70 us with MinGW libgomp at 16 threads, against
 * 1.3 us and 0.4 us for the same two prices under Linux glibc libgomp on a
 * 4-core VM -- a difference in the RUNTIME, not in the hardware.
 *
 * This file exists to compare runtimes, so it is deliberately portable: omp.h
 * and stdio, omp_get_wtime() rather than clock_gettime(), nothing from the
 * engine. It builds with all three toolchains that can produce a Windows
 * binary, and they do NOT all use the same OpenMP runtime:
 *
 *   gcc   -O2 -fopenmp        tests/bench_omp_sync.c -o bench_omp_sync      (libgomp)
 *   clang -O2 -fopenmp        tests/bench_omp_sync.c -o bench_omp_sync      (libomp, from
 *                                                      MSYS2 CLANG64 -- UCRT64's clang
 *                                                      links libgomp and measures the same
 *                                                      thing as gcc)
 *   cl /O2 /openmp:llvm       tests\bench_omp_sync.c                        (libomp, ships
 *                                                      with VS 2022 -- nothing to install)
 *
 * Run each at the same OMP_NUM_THREADS. The minimum of several samples is
 * printed beside the median: these prices are thread wake-up latency, so one
 * descheduled team member inflates a sample, and a minimum far below the median
 * means the machine was never quiet.
 *
 *   make -C c bench-omp-sync
 *
 * Args: [samples] [reps per sample].
 */
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
static volatile long g_sink;

static double region_us(long reps) {
    long acc = 0;
    double t0 = omp_get_wtime();
    for (long r = 0; r < reps; r++) {
        #pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < 1; i++) acc += i + r;
    }
    double par = omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (long r = 0; r < reps; r++)
        for (int i = 0; i < 1; i++) acc += i + r;
    double seq = omp_get_wtime() - t0;
    g_sink = acc;
    return (par - seq) / (double)reps * 1e6;
}

static double barrier_us(long reps) {
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        for (long r = 0; r < reps; r++) {
            #pragma omp barrier
        }
    }
    return (omp_get_wtime() - t0) / (double)reps * 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void price(const char *name, double (*fn)(long), long reps, int runs) {
    double *v = (double *)malloc(sizeof(double) * (size_t)runs);
    if (!v) { printf("  %-10s out of memory\n", name); return; }
    for (int i = 0; i < runs; i++) v[i] = fn(reps);
    qsort(v, (size_t)runs, sizeof(double), cmp_d);
    printf("  %-10s %8.2f us   (quietest of %d: %8.2f)\n", name, v[runs / 2], runs, v[0]);
    free(v);
}
#endif

int main(int argc, char **argv) {
#ifndef _OPENMP
    (void)argc; (void)argv;
    printf("built without OpenMP -- nothing to measure\n");
    return 0;
#else
    int runs = argc > 1 ? atoi(argv[1]) : 7;
    long reps = argc > 2 ? atol(argv[2]) : 4000;
    if (runs < 1) runs = 1;
    if (reps < 1) reps = 1;

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }

    printf("bench_omp_sync: team of %d, %d samples of %ld\n", nthreads, runs, reps);
    price("region", region_us, reps, runs);
    price("barrier", barrier_us, reps, runs);
    printf("\n  single-digit us: synchronisation is free, look at the kernels.\n");
    printf("  tens of us: the runtime is the bill. Try another one before\n");
    printf("  changing any code -- this file is here to compare them.\n");
    return 0;
#endif
}
