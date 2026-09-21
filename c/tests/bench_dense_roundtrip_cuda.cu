/* What a resident dense GEMV round-trip actually costs, and how much of it is
 * the transport rather than the kernel.
 *
 * WHY THIS EXISTS. The per-site dense profile measured, on qwen36:
 *
 *     dnproj   161.7 us kernel | 29.9 h2d | 35.6 d2h | 10.6 residue
 *     dnout     60.3 us kernel | 30.2 h2d | 27.5 d2h | 11.6 residue
 *
 * ~30 us per transfer on payloads of 8, 16 and 48 KiB -- flat in size, so
 * latency rather than bandwidth -- and only ~11 us of launch/sync residue.
 * DeltaNet makes 60 such calls a decode token, so the transports alone are
 * ~4 ms/token.
 *
 * What that measurement CANNOT say is whether 30 us is the driver's floor or
 * something the engine's context adds: a busy host, a second stream running
 * the expert group, WDDM submission under load. The difference decides the
 * lever. If an isolated 8 KiB H2D costs 5 us, the engine is paying 25 us of
 * something and that something is the target. If it costs 28, the floor is the
 * floor and the only lever left is the NUMBER of transfers.
 *
 * So this runs the same shapes, alone on the device, with nothing else in
 * flight, across the transports that are actually available:
 *
 *   pageable   what the engine does today
 *   pinned     the same two synchronous copies out of page-locked memory
 *   async      pinned, both copies async on one stream, ONE synchronize
 *   graph      the whole H2D + kernel + D2H captured and replayed as one
 *              launch -- three driver submissions become one
 *
 * and, separately, the transfers with NO kernel at all, which is the only way
 * to see the transport cost without a kernel's completion mixed into it.
 *
 * It is a bench, not a gate: it prints numbers and asserts nothing. `make
 * dense-roundtrip-bench`.
 *
 * READING IT. The engine's numbers above come from a PROFILED run with four
 * cudaEventRecord per call and the expert group live on another stream. These
 * come from an idle device. They are a FLOOR, not a prediction of what the
 * engine would see -- the gap between the two is the finding, not a defect.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>

#include "../backend_gpu_compat.h"

#if !COLI_GPU_HAS_GRAPH
#  define BENCH_NO_GRAPH 1
#endif

static double now_ms(void){
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* A stand-in for quant_matmul's int8 per-row branch: one block per output row,
 * 256 threads, a byte per weight. The point is not the kernel's quality -- the
 * engine's own is measured elsewhere -- but that a realistic amount of device
 * work sits between the two copies. */
__global__ void gemv_i8(float *y, const float *x, const int8_t *w,
                        const float *sc, int I, int O){
    int o = blockIdx.x; if (o >= O) return;
    const int8_t *row = w + (size_t)o * I;
    __shared__ float part[256];
    float a = 0.f;
    for (int i = threadIdx.x; i < I; i += 256) a += x[i] * (float)row[i];
    part[threadIdx.x] = a; __syncthreads();
    for (int s = 128; s; s >>= 1){ if (threadIdx.x < (unsigned)s) part[threadIdx.x] += part[threadIdx.x+s]; __syncthreads(); }
    if (!threadIdx.x) y[o] = part[0] * sc[o];
}

enum { REPS = 400, WARM = 50 };

struct Ctx {
    int I, O;
    int8_t *w; float *sc;          /* device, resident like the engine's */
    float *dx, *dy;                /* device staging */
    float *hx_page, *hy_page;      /* pageable host */
    float *hx_pin,  *hy_pin;       /* pinned host */
    size_t xb, yb;
};

static int setup(Ctx *c, int I, int O){
    c->I = I; c->O = O;
    c->xb = (size_t)I * sizeof(float); c->yb = (size_t)O * sizeof(float);
    int8_t *hw = (int8_t*)malloc((size_t)I*O); float *hs = (float*)malloc((size_t)O*4);
    if (!hw || !hs) return 0;
    for (size_t i = 0; i < (size_t)I*O; i++) hw[i] = (int8_t)(i & 63);
    for (int o = 0; o < O; o++) hs[o] = 0.01f;
    if (cudaMalloc(&c->w, (size_t)I*O) != cudaSuccess) return 0;
    if (cudaMalloc(&c->sc, (size_t)O*4) != cudaSuccess) return 0;
    if (cudaMalloc(&c->dx, c->xb) != cudaSuccess) return 0;
    if (cudaMalloc(&c->dy, c->yb) != cudaSuccess) return 0;
    cudaMemcpy(c->w, hw, (size_t)I*O, cudaMemcpyHostToDevice);
    cudaMemcpy(c->sc, hs, (size_t)O*4, cudaMemcpyHostToDevice);
    free(hw); free(hs);
    c->hx_page = (float*)malloc(c->xb); c->hy_page = (float*)malloc(c->yb);
    if (!c->hx_page || !c->hy_page) return 0;
    for (int i = 0; i < I; i++) c->hx_page[i] = 0.001f * (i & 255);
    if (cudaMallocHost(&c->hx_pin, c->xb) != cudaSuccess) return 0;
    if (cudaMallocHost(&c->hy_pin, c->yb) != cudaSuccess) return 0;
    memcpy(c->hx_pin, c->hx_page, c->xb);
    return 1;
}

static void teardown(Ctx *c){
    cudaFree(c->w); cudaFree(c->sc); cudaFree(c->dx); cudaFree(c->dy);
    free(c->hx_page); free(c->hy_page);
    cudaFreeHost(c->hx_pin); cudaFreeHost(c->hy_pin);
}

/* Each arm returns microseconds per round trip. Timed on the host over REPS
 * iterations, which is what the caller actually waits for -- the engine's
 * cudaEvent breakdown is what this file exists to put a floor under, so
 * measuring it the same way would beg the question. */
/* The pageable and pinned arms use the DEFAULT stream, because that is what
 * coli_cuda_matmul does; the async and graph arms need a stream of their own.
 * Every arm is warm loop, sync, timed loop, sync -- the same shape, so the
 * numbers are comparable to each other and not only to themselves. */
static double arm_pageable(Ctx *c, int with_kernel){
    for (int r = 0; r < WARM; r++){
        cudaMemcpy(c->dx, c->hx_page, c->xb, cudaMemcpyHostToDevice);
        if (with_kernel) gemv_i8<<<c->O, 256>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpy(c->hy_page, c->dy, c->yb, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    double t = now_ms();
    for (int r = 0; r < REPS; r++){
        cudaMemcpy(c->dx, c->hx_page, c->xb, cudaMemcpyHostToDevice);
        if (with_kernel) gemv_i8<<<c->O, 256>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpy(c->hy_page, c->dy, c->yb, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    return (now_ms() - t) * 1000.0 / REPS;
}

static double arm_pinned(Ctx *c, int with_kernel){
    for (int r = 0; r < WARM; r++){
        cudaMemcpy(c->dx, c->hx_pin, c->xb, cudaMemcpyHostToDevice);
        if (with_kernel) gemv_i8<<<c->O, 256>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpy(c->hy_pin, c->dy, c->yb, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    double t = now_ms();
    for (int r = 0; r < REPS; r++){
        cudaMemcpy(c->dx, c->hx_pin, c->xb, cudaMemcpyHostToDevice);
        if (with_kernel) gemv_i8<<<c->O, 256>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpy(c->hy_pin, c->dy, c->yb, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    return (now_ms() - t) * 1000.0 / REPS;
}

static double arm_async(Ctx *c, cudaStream_t st, int with_kernel){
    for (int r = 0; r < WARM; r++){
        cudaMemcpyAsync(c->dx, c->hx_pin, c->xb, cudaMemcpyHostToDevice, st);
        if (with_kernel) gemv_i8<<<c->O, 256, 0, st>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpyAsync(c->hy_pin, c->dy, c->yb, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
    double t = now_ms();
    for (int r = 0; r < REPS; r++){
        cudaMemcpyAsync(c->dx, c->hx_pin, c->xb, cudaMemcpyHostToDevice, st);
        if (with_kernel) gemv_i8<<<c->O, 256, 0, st>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
        cudaMemcpyAsync(c->hy_pin, c->dy, c->yb, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
    return (now_ms() - t) * 1000.0 / REPS;
}

#ifndef BENCH_NO_GRAPH
static double arm_graph(Ctx *c, cudaStream_t st, int with_kernel, int *ok){
    *ok = 0;
    if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return 0;
    cudaMemcpyAsync(c->dx, c->hx_pin, c->xb, cudaMemcpyHostToDevice, st);
    if (with_kernel) gemv_i8<<<c->O, 256, 0, st>>>(c->dy, c->dx, c->w, c->sc, c->I, c->O);
    cudaMemcpyAsync(c->hy_pin, c->dy, c->yb, cudaMemcpyDeviceToHost, st);
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(st, &g) != cudaSuccess || !g) { cudaGetLastError(); return 0; }
    cudaGraphExec_t ex = nullptr;
    if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess || !ex) { cudaGraphDestroy(g); cudaGetLastError(); return 0; }
    cudaGraphDestroy(g);
    for (int r = 0; r < WARM; r++){ cudaGraphLaunch(ex, st); cudaStreamSynchronize(st); }
    double t = now_ms();
    for (int r = 0; r < REPS; r++){ cudaGraphLaunch(ex, st); cudaStreamSynchronize(st); }
    double us = (now_ms() - t) * 1000.0 / REPS;
    cudaGraphExecDestroy(ex);
    *ok = 1;
    return us;
}
#endif

static void run(int I, int O, const char *what){
    Ctx c; memset(&c, 0, sizeof c);
    if (!setup(&c, I, O)) { printf("  [%s] setup failed\n", what); teardown(&c); return; }
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

    double pg_k = arm_pageable(&c, 1), pg_0 = arm_pageable(&c, 0);
    double pn_k = arm_pinned(&c, 1),   pn_0 = arm_pinned(&c, 0);
    double as_k = arm_async(&c, st, 1), as_0 = arm_async(&c, st, 0);
    double gr_k = 0, gr_0 = 0; int gk = 0, g0 = 0;
#ifndef BENCH_NO_GRAPH
    gr_k = arm_graph(&c, st, 1, &gk);
    gr_0 = arm_graph(&c, st, 0, &g0);
#endif

    printf("\n  %s  I=%d O=%d   x %zu B -> y %zu B\n", what, I, O, c.xb, c.yb);
    printf("    %-10s %12s %12s %12s\n", "transport", "with kernel", "no kernel", "kernel share");
    printf("    %-10s %9.1f us %9.1f us %9.1f us\n", "pageable", pg_k, pg_0, pg_k - pg_0);
    printf("    %-10s %9.1f us %9.1f us %9.1f us\n", "pinned",   pn_k, pn_0, pn_k - pn_0);
    printf("    %-10s %9.1f us %9.1f us %9.1f us\n", "async+1sync", as_k, as_0, as_k - as_0);
    if (gk && g0)
        printf("    %-10s %9.1f us %9.1f us %9.1f us\n", "graph", gr_k, gr_0, gr_k - gr_0);
    else
        printf("    %-10s %s\n", "graph", "not available on this target");

    cudaStreamDestroy(st);
    teardown(&c);
}

int main(void){
    int dev = 0; cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess){ printf("no CUDA device\n"); return 1; }
    printf("dense round-trip bench: %s, %d reps per arm, idle device\n", p.name, REPS);
    printf("the engine's per-site profile measured ~30 us per transfer and ~11 us of\n"
           "launch/sync residue, under a profiler and with the expert group live on\n"
           "another stream. These are the same shapes with nothing else in flight:\n"
           "a FLOOR to compare that against, not a prediction of it.\n");

    /* The two shapes DeltaNet actually uses on qwen36-35B-A3B:
     *   dnproj  hidden 2048 -> conv_dim 8192 ++ value_dim 4096 = 12288
     *   dnout   value_dim 4096 -> hidden 2048                          */
    run(2048, 12288, "dnproj");
    run(4096,  2048, "dnout");
    /* And the degenerate case: no weights to read, so whatever is left is
     * purely what a round trip costs to ask for. */
    run(64, 64, "tiny (transport only)");

    printf("\n");
    return 0;
}
