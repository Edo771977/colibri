/* The resident dense GEMV alone: how fast is the kernel when nothing else runs?
 *
 * Inside qwen36, COLI_CUDA_PROFILE prices the placed dense GEMVs (dnproj,
 * dnout, attnproj, attnout, lm_head) at about 133 us per call, 82 of them
 * kernel, and the kernel at ~255 GB/s on an RTX 4070 Ti SUPER. But that
 * kernel column is two cudaEvents recorded around one launch on an otherwise
 * idle stream: if the CPU is slow to issue the launch after the first event,
 * the delay is counted as kernel. docs/qwen36-cuda-tier.md records the same
 * calls reading 301 ms one day and 640 ms the next, and suspects exactly that.
 *
 * This program runs the same kernels with no engine around them -- no OpenMP
 * team, no expert tier -- on the same shapes and in the same number of
 * distinct matrices as one qwen36 token (so the working set does not sit in
 * L2), and times each shape three ways:
 *
 *   batch   N launches back to back, two events around the lot: the GPU is
 *           never waiting for the host, so this is the kernel's throughput.
 *   single  one launch between two events, then a synchronize, per call: the
 *           way the engine's profile measures it, minus the engine.
 *   call    coli_cuda_matmul itself, host wall clock per call: H2D of x,
 *           kernel, D2H of y, both copies pageable -- the whole round trip
 *           the engine pays, minus the engine.
 *
 * for each width of COLI_CUDA_I8_ROWS (0, 2, 4, 8), three rounds each, and
 * sums the per-call medians into ms per token over the 81 calls of a token.
 *
 * GAPS. Run that way, the card is never idle, and inside the engine it is:
 * between two dense calls the CPU runs the convolution, the recurrence, the
 * norms and the MoE, so a token is ~81 short GPU bursts with gaps of tens to
 * hundreds of microseconds (and nvidia-smi saw the memory clock drop from
 * 10251 to 5001 MHz after ~1.5 s of that). The gap section repeats single and
 * call at R=2 with the host spinning GAP microseconds before every call, the
 * GPU idle meanwhile; the gap itself is not counted in the time per call. If
 * the per-call cost grows with the gap toward the engine's ~133 us, the
 * cost lives in the card going idle between calls, not in the engine.
 *
 * FRESH OUTPUT. qwen36's step() mallocs the logit buffer (vocab floats,
 * ~1 MB) for every token and its caller frees it, so lm_head's D2H lands in
 * memory the process has never touched; a block that size typically comes
 * straight from the OS, one page fault per 4 KB page on first write. The
 * bench reuses one buffer. The fresh section times call at R=2 three ways:
 * into the reused buffer (control), into a buffer malloc'd for this call (the
 * engine's pattern; malloc and free outside the timed window, as in step()),
 * and into the reused buffer plus a memcpy into a fresh one.
 *
 * Before timing, it checks that every width gives the same bytes as R=0 and
 * that R=0 matches a double-precision CPU reference, so the numbers belong to
 * kernels that compute the right thing.
 *
 * Build and run: make dense-gemv-bench   (from a shell with nvcc; on Windows
 * the x64 Native Tools prompt, as for make cuda-dll). Takes about two minutes
 * and at most ~0.8 GB of VRAM (one shape at a time): close the engine first.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

/* No direct <cuda_runtime.h>: the backend include supplies the runtime. */
#include "../backend_cuda.cu"

#ifdef _WIN32
static int bench_setenv(const char *name, const char *value) { return _putenv_s(name, value); }
#else
static int bench_setenv(const char *name, const char *value) { return setenv(name, value, 1); }
#endif

struct Shape { const char *name; int I, O, count; };
/* One qwen36 (35B-A3B) token: 30 DeltaNet layers, 10 attention layers.
 * dnproj is qkv ++ z fused (2048 -> 8192 + 4096), attnproj q ++ k ++ v fused
 * (2048 -> 8192 + 512 + 512). The sizes reproduce the VRAM the engine's
 * [place]/[dnp] lines report: 0.70 GB of dnproj, 0.31 GB of attnout + dnout,
 * 0.18 GB of attnproj. */
static const Shape SHAPES[] = {
    { "dnproj",   2048,  12288, 30 },
    { "dnout",    4096,   2048, 30 },
    { "attnproj", 2048,   9216, 10 },
    { "attnout",  4096,   2048, 10 },
    { "lm_head",  2048, 248320,  1 },
};
static const int NSHAPES = (int)(sizeof SHAPES / sizeof SHAPES[0]);
static const int WIDTHS[] = { 0, 2, 4, 8 };
static const int NWIDTHS = 4;
static const int ROUNDS = 3;

static int fails;

static void die(const char *what) {
    printf("FATAL %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    exit(1);
}
static void cuda_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { printf("FATAL %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

static void set_width(int r) {
    char v[8]; snprintf(v, sizeof v, "%d", r);
    bench_setenv("COLI_CUDA_I8_ROWS", v);
    if (i8_rows_mode() != r) { printf("FATAL COLI_CUDA_I8_ROWS=%d read back as %d\n", r, i8_rows_mode()); exit(1); }
}

static uint32_t rng_state = 0x2545f491u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
                            rng_state ^= rng_state << 5; return rng_state; }

static double now_us(void) {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median3(double a, double b, double c) {
    double v[3] = { a, b, c }; std::sort(v, v + 3); return v[1];
}

/* Per shape, per width, per round: microseconds per call, three ways. */
static double t_batch[8][NWIDTHS][ROUNDS], t_single[8][NWIDTHS][ROUNDS];
static double t_wall1[8][NWIDTHS][ROUNDS], t_call[8][NWIDTHS][ROUNDS];
/* Gap section, R=2: per shape, per gap, per round. */
static const int GAPS[] = { 0, 50, 200, 500 };
static const int NGAPS = 4;
static double g_kern[8][NGAPS][ROUNDS], g_call[8][NGAPS][ROUNDS];
/* Fresh section, R=2: reused / fresh / reused + memcpy to fresh. */
static double f_time[8][3][ROUNDS];

static void spin_us(double us) {
    if (us <= 0) return;
    double t0 = now_us();
    while (now_us() - t0 < us) { }
}

int main(void) {
    int devs[1] = { 0 };
    /* Unprofiled, pageable: the engine's default arm. Both are read by the
     * backend; the profile flag is cached on first use, so clear it first. */
    bench_setenv("COLI_CUDA_PROFILE", "0");
    bench_setenv("COLI_CUDA_DENSE_PINNED", "0");
    if (!coli_cuda_init(devs, 1)) { printf("FATAL cuda init\n"); return 1; }
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "device properties");
    printf("device 0: %s, sm_%d%d, %.1f GB, L2 %.0f MB\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9, prop.l2CacheSize / 1048576.0);

    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "event");
    cuda_check(cudaEventCreate(&e1), "event");

    for (int si = 0; si < NSHAPES; si++) {
        const Shape &sh = SHAPES[si];
        const int I = sh.I, O = sh.O, n = sh.count;
        const size_t wn = (size_t)I * O;
        const double bytes = (double)wn + (double)O * sizeof(float);   /* weights + scales, as dense_stats */

        std::vector<int8_t> w(wn);
        std::vector<float> sc(O), x(I), y((size_t)O), yref((size_t)O), y0((size_t)O);
        for (size_t i = 0; i < wn; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
        for (int o = 0; o < O; o++) sc[o] = 0.0005f + (float)(rnd() % 1000) / 1e6f;
        for (int i = 0; i < I; i++) x[i] = (float)((int)(rnd() % 2001) - 1000) / 1000.0f;

        /* n distinct matrices, as in the engine: the same bytes at n device
         * addresses, so repeated passes cannot be served from L2. */
        std::vector<ColiCudaTensor *> t(n, nullptr);
        for (int m = 0; m < n; m++)
            if (!coli_cuda_tensor_upload(&t[m], w.data(), sc.data(), 1, I, O, 0)) {
                printf("FATAL upload %s %d/%d (VRAM?)\n", sh.name, m + 1, n); return 1; }
        float *dx = nullptr, *dy = nullptr;
        cuda_check(cudaMalloc(&dx, (size_t)I * sizeof(float)), "dx");
        cuda_check(cudaMalloc(&dy, (size_t)O * sizeof(float)), "dy");
        cuda_check(cudaMemcpy(dx, x.data(), (size_t)I * sizeof(float), cudaMemcpyHostToDevice), "x upload");
        const size_t rb = row_bytes(1, I);

        /* Correctness first: R=0 against a double CPU reference, every width
         * against R=0 byte for byte. */
        for (int o = 0; o < O; o++) {
            double a = 0; const int8_t *r = w.data() + (size_t)o * I;
            for (int i = 0; i < I; i++) a += (double)x[i] * r[i];
            yref[o] = (float)(a * sc[o]);
        }
        double se = 0, sr = 0;
        for (int wi = 0; wi < NWIDTHS; wi++) {
            set_width(WIDTHS[wi]);
            if (!coli_cuda_matmul(&t[0], y.data(), x.data(), w.data(), sc.data(), 1, 1, I, O, 0, 0)) die("matmul");
            if (wi == 0) {
                y0 = y;
                for (int o = 0; o < O; o++) { double d = (double)y[o] - yref[o]; se += d * d; sr += (double)yref[o] * yref[o]; }
            } else if (memcmp(y.data(), y0.data(), (size_t)O * sizeof(float))) {
                printf("FAIL %s: COLI_CUDA_I8_ROWS=%d differs from R=0\n", sh.name, WIDTHS[wi]); fails++;
            }
        }
        double rel = sr > 0 ? sqrt(se / sr) : sqrt(se);
        if (!(rel < 1e-5)) { printf("FAIL %s: R=0 vs CPU reference rel rms %.2e\n", sh.name, rel); fails++; }

        /* Enough calls per measurement to cover ~0.5 s of lm_head and a few
         * hundred calls of the rest. */
        const int passes = n >= 30 ? 20 : n >= 10 ? 60 : 300;
        const int calls = passes * n;

        for (int round = 0; round < ROUNDS; round++)
            for (int wi = 0; wi < NWIDTHS; wi++) {
                set_width(WIDTHS[wi]);
                /* warm-up: one pass */
                for (int m = 0; m < n; m++)
                    quant_matmul_launch(dy, dx, t[m]->weights, t[m]->scales, 1, 1, I, O, rb, t[m]->gs, t[m]->ng);
                cuda_check(cudaDeviceSynchronize(), "warm-up");

                /* batch */
                cudaEventRecord(e0, 0);
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++)
                        quant_matmul_launch(dy, dx, t[m]->weights, t[m]->scales, 1, 1, I, O, rb, t[m]->gs, t[m]->ng);
                cudaEventRecord(e1, 0);
                cuda_check(cudaEventSynchronize(e1), "batch sync");
                cuda_check(cudaGetLastError(), "batch launch");
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                t_batch[si][wi][round] = 1000.0 * ms / calls;

                /* single: event, launch, event, synchronize -- per call */
                double ksum = 0, wsum = 0;
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++) {
                        double h0 = now_us();
                        cudaEventRecord(e0, 0);
                        quant_matmul_launch(dy, dx, t[m]->weights, t[m]->scales, 1, 1, I, O, rb, t[m]->gs, t[m]->ng);
                        cudaEventRecord(e1, 0);
                        cuda_check(cudaEventSynchronize(e1), "single sync");
                        wsum += now_us() - h0;
                        float k = 0; cudaEventElapsedTime(&k, e0, e1); ksum += 1000.0 * k;
                    }
                cuda_check(cudaGetLastError(), "single launch");
                t_single[si][wi][round] = ksum / calls;
                t_wall1[si][wi][round] = wsum / calls;

                /* call: the engine's entry point, pageable copies included */
                double c0 = now_us();
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++)
                        if (!coli_cuda_matmul(&t[m], y.data(), x.data(), w.data(), sc.data(), 1, 1, I, O, 0, 0)) die("matmul");
                t_call[si][wi][round] = (now_us() - c0) / calls;
            }

        /* gaps: R=2, the host idles the card for GAPS[gi] us before each call */
        set_width(2);
        for (int round = 0; round < ROUNDS; round++)
            for (int gi = 0; gi < NGAPS; gi++) {
                double ksum = 0, csum = 0;
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++) {
                        spin_us(GAPS[gi]);
                        cudaEventRecord(e0, 0);
                        quant_matmul_launch(dy, dx, t[m]->weights, t[m]->scales, 1, 1, I, O, rb, t[m]->gs, t[m]->ng);
                        cudaEventRecord(e1, 0);
                        cuda_check(cudaEventSynchronize(e1), "gap single sync");
                        float k = 0; cudaEventElapsedTime(&k, e0, e1); ksum += 1000.0 * k;
                    }
                cuda_check(cudaGetLastError(), "gap single launch");
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++) {
                        spin_us(GAPS[gi]);
                        double c0 = now_us();
                        if (!coli_cuda_matmul(&t[m], y.data(), x.data(), w.data(), sc.data(), 1, 1, I, O, 0, 0)) die("matmul");
                        csum += now_us() - c0;
                    }
                g_kern[si][gi][round] = ksum / calls;
                g_call[si][gi][round] = csum / calls;
            }

        /* fresh output buffer: the engine's malloc-per-token pattern */
        for (int round = 0; round < ROUNDS; round++)
            for (int fm = 0; fm < 3; fm++) {
                double tsum = 0;
                for (int p = 0; p < passes; p++)
                    for (int m = 0; m < n; m++) {
                        float *yf = fm ? (float *)malloc((size_t)O * sizeof(float)) : nullptr;
                        if (fm && !yf) { printf("FATAL malloc\n"); return 1; }
                        float *dst = fm == 1 ? yf : y.data();
                        double c0 = now_us();
                        if (!coli_cuda_matmul(&t[m], dst, x.data(), w.data(), sc.data(), 1, 1, I, O, 0, 0)) die("matmul");
                        if (fm == 2) memcpy(yf, y.data(), (size_t)O * sizeof(float));
                        tsum += now_us() - c0;
                        if (fm == 1 && memcmp(yf, y0.data(), (size_t)O * sizeof(float))) {
                            printf("FAIL %s: fresh-buffer result differs\n", sh.name); fails++; }
                        free(yf);
                    }
                f_time[si][fm][round] = tsum / calls;
            }

        printf("\n%-8s  I=%d O=%d  x%d matrices, %.1f MB each, %d calls per measurement, ref rel rms %.1e\n",
               sh.name, I, O, n, bytes / 1e6, calls, rel);
        printf("  R  round | batch us  GB/s | single kernel us  GB/s  host us | call us\n");
        for (int wi = 0; wi < NWIDTHS; wi++)
            for (int round = 0; round < ROUNDS; round++)
                printf("  %d  %d     | %8.1f %5.0f | %16.1f %5.0f %8.1f | %7.1f\n", WIDTHS[wi], round + 1,
                       t_batch[si][wi][round], bytes / (t_batch[si][wi][round] * 1e3),
                       t_single[si][wi][round], bytes / (t_single[si][wi][round] * 1e3),
                       t_wall1[si][wi][round], t_call[si][wi][round]);
        printf("  gaps at R=2 (gap not counted) | single kernel us  GB/s | call us\n");
        for (int gi = 0; gi < NGAPS; gi++)
            for (int round = 0; round < ROUNDS; round++)
                printf("  gap %3d us  round %d          | %16.1f %5.0f | %7.1f\n", GAPS[gi], round + 1,
                       g_kern[si][gi][round], bytes / (g_kern[si][gi][round] * 1e3), g_call[si][gi][round]);
        printf("  output buffer at R=2          | reused us | fresh us | reused+memcpy us\n");
        for (int round = 0; round < ROUNDS; round++)
            printf("  round %d                       | %9.1f | %8.1f | %16.1f\n", round + 1,
                   f_time[si][0][round], f_time[si][1][round], f_time[si][2][round]);

        cudaFree(dx); cudaFree(dy);
        for (int m = 0; m < n; m++) coli_cuda_tensor_free(t[m]);
    }

    /* Per token: 81 calls, the medians of the three rounds summed per shape. */
    printf("\n--- per qwen36 token (%d calls), median of %d rounds, ms ---\n", 30 + 30 + 10 + 10 + 1, ROUNDS);
    printf("  R | batch | single kernel | single host | call\n");
    for (int wi = 0; wi < NWIDTHS; wi++) {
        double b = 0, s = 0, h = 0, c = 0;
        for (int si = 0; si < NSHAPES; si++) {
            double k = SHAPES[si].count / 1000.0;
            b += k * median3(t_batch[si][wi][0], t_batch[si][wi][1], t_batch[si][wi][2]);
            s += k * median3(t_single[si][wi][0], t_single[si][wi][1], t_single[si][wi][2]);
            h += k * median3(t_wall1[si][wi][0], t_wall1[si][wi][1], t_wall1[si][wi][2]);
            c += k * median3(t_call[si][wi][0], t_call[si][wi][1], t_call[si][wi][2]);
        }
        printf("  %d | %5.2f | %13.2f | %11.2f | %4.2f\n", WIDTHS[wi], b, s, h, c);
    }
    printf("\n--- per qwen36 token at R=2, host gap before every call (gap not counted), median of %d rounds, ms ---\n", ROUNDS);
    printf("  gap us | single kernel | call\n");
    for (int gi = 0; gi < NGAPS; gi++) {
        double s = 0, c = 0;
        for (int si = 0; si < NSHAPES; si++) {
            double k = SHAPES[si].count / 1000.0;
            s += k * median3(g_kern[si][gi][0], g_kern[si][gi][1], g_kern[si][gi][2]);
            c += k * median3(g_call[si][gi][0], g_call[si][gi][1], g_call[si][gi][2]);
        }
        printf("  %6d | %13.2f | %4.2f\n", GAPS[gi], s, c);
    }
    printf("\n--- per qwen36 token at R=2, output buffer, median of %d rounds, ms ---\n", ROUNDS);
    printf("  reused | fresh | reused+memcpy\n");
    {
        double f[3] = { 0, 0, 0 };
        for (int fm = 0; fm < 3; fm++)
            for (int si = 0; si < NSHAPES; si++)
                f[fm] += SHAPES[si].count / 1000.0 * median3(f_time[si][fm][0], f_time[si][fm][1], f_time[si][fm][2]);
        printf("  %6.2f | %5.2f | %13.2f\n", f[0], f[1], f[2]);
    }
    double total_bytes = 0;
    for (int si = 0; si < NSHAPES; si++)
        total_bytes += SHAPES[si].count * ((double)SHAPES[si].I * SHAPES[si].O + SHAPES[si].O * 4.0);
    printf("  weight bytes per token: %.2f GB\n", total_bytes / 1e9);

    bench_setenv("COLI_CUDA_I8_ROWS", "");
    if (fails) { printf("\n%d check(s) FAILED: the timings above are not of a correct kernel\n", fails); return 1; }
    printf("\nOK: every width equals R=0 byte for byte, R=0 equals the CPU reference\n");
    return 0;
}
