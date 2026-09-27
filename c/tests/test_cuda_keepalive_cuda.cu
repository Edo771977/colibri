/* COLI_CUDA_KEEPALIVE: the opt-in spin that keeps the card from dropping to
 * a low power state between qwen36's short GPU bursts.
 *
 * What it must never do is change an answer or outlive the backend, so this
 * pins, on real silicon:
 *   1. the parse: only "1" turns it on; unset, "0" and "true" leave it off;
 *   2. on, the thread actually launches (a counter, since a keep-alive that
 *      never runs would pass every other check);
 *   3. a dense GEMV computed while it runs equals, byte for byte, the same
 *      GEMV with it stopped;
 *   4. coli_cuda_shutdown stops and joins it, and a later init with the
 *      variable off does not restart it.
 * Whether the driver keeps its clocks up is not testable here: that is the
 * measurement in docs/experiments/qwen36-gpu-clocks-2026-09-27-raw.txt.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_cuda_keepalive_cuda.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>

/* No direct <cuda_runtime.h>: the backend include supplies the runtime. */
#include "../backend_cuda.cu"

#ifdef _WIN32
static int set_env(const char *name, const char *value) { return _putenv_s(name, value); }
#else
static int set_env(const char *name, const char *value) { return setenv(name, value, 1); }
#endif

static int fails;
static void check(int ok, const char *what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

static int devs[1] = { 0 };

static void gemv(float *y, const int8_t *w, const float *sc, const float *x, int I, int O) {
    ColiCudaTensor *t = NULL;
    if (!coli_cuda_matmul(&t, y, x, w, sc, 1, 1, I, O, 0, 0)) { printf("FATAL matmul\n"); exit(1); }
    coli_cuda_tensor_free(t);
}

int main(void) {
#if !COLI_HAS_KEEPALIVE
    printf("SKIP test_cuda_keepalive: CUDA only (the spin reads %%globaltimer)\n");
    return 0;
#else
    printf("parse\n");
    set_env("COLI_CUDA_KEEPALIVE", "");     check(!keepalive_mode(), "unset: off");
    set_env("COLI_CUDA_KEEPALIVE", "0");    check(!keepalive_mode(), "\"0\": off");
    set_env("COLI_CUDA_KEEPALIVE", "true"); check(!keepalive_mode(), "\"true\": off (only \"1\")");
    set_env("COLI_CUDA_KEEPALIVE", "1");    check(keepalive_mode(), "\"1\": on");

    printf("running\n");
    if (!coli_cuda_init(devs, 1)) { printf("FATAL cuda init\n"); return 1; }
    check(g_ka_run.load() == 1, "init with COLI_CUDA_KEEPALIVE=1 starts it");
    unsigned long long l0 = g_ka_launches.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    unsigned long long l1 = g_ka_launches.load();
    printf("  %llu spin kernels completed in 100 ms\n", l1 - l0);
    check(l1 - l0 >= 20, "it launches (>= 20 one-millisecond spins in 100 ms)");

    printf("results\n");
    const int I = 2048, O = 4096;
    int8_t *w = (int8_t *)malloc((size_t)I * O);
    float *sc = (float *)malloc(O * sizeof(float)), *x = (float *)malloc(I * sizeof(float));
    float *y_on = (float *)malloc(O * sizeof(float)), *y_off = (float *)malloc(O * sizeof(float));
    uint32_t r = 12345u;
    for (size_t i = 0; i < (size_t)I * O; i++) { r = r * 1103515245u + 12345u; w[i] = (int8_t)((int)(r >> 24) % 255 - 127); }
    for (int o = 0; o < O; o++) sc[o] = 0.001f + (float)(o % 7) * 1e-4f;
    for (int i = 0; i < I; i++) x[i] = (float)((i * 37) % 201 - 100) / 100.f;
    gemv(y_on, w, sc, x, I, O);
    keepalive_stop();
    check(g_ka_run.load() == 0 && !g_ka_thr.joinable(), "keepalive_stop stops and joins it");
    gemv(y_off, w, sc, x, I, O);
    check(!memcmp(y_on, y_off, O * sizeof(float)), "a GEMV while it runs equals the GEMV without it, byte for byte");

    printf("shutdown\n");
    coli_cuda_shutdown();
    set_env("COLI_CUDA_KEEPALIVE", "1");
    if (!coli_cuda_init(devs, 1)) { printf("FATAL cuda re-init\n"); return 1; }
    check(g_ka_run.load() == 1, "re-init with it on restarts it");
    coli_cuda_shutdown();
    check(g_ka_run.load() == 0 && !g_ka_thr.joinable(), "coli_cuda_shutdown stops and joins it");
    set_env("COLI_CUDA_KEEPALIVE", "0");
    if (!coli_cuda_init(devs, 1)) { printf("FATAL cuda re-init\n"); return 1; }
    check(g_ka_run.load() == 0, "init with it off does not start it");
    coli_cuda_shutdown();

    free(w); free(sc); free(x); free(y_on); free(y_off);
    set_env("COLI_CUDA_KEEPALIVE", "");
    if (fails) { printf("test_cuda_keepalive: %d failure(s)\n", fails); return 1; }
    printf("OK test_cuda_keepalive: off unless \"1\", launches when on, changes no byte, stops at shutdown\n");
    return 0;
#endif
}
