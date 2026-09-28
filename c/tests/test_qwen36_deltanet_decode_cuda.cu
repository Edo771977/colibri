/* coli_cuda_deltanet_* (backend_cuda.cu): one DeltaNet decode layer in one
 * backend call, on real silicon. docs/qwen36-deltanet-gpu-plan.md, stage 2.
 *
 * The projections are placed the way qwen36 places them: per-row int8 (fmt 1,
 * one scale per row), uploaded with coli_cuda_tensor_upload. The CPU
 * reference is the same chain the engine runs today: the int8 dnproj GEMV,
 * deltanet()'s middle (dn_ref_step, qwen36_deltanet_ref.h), the int8 dnout
 * GEMV. For the 35B shapes and the CI tiny fixture's:
 *   1. one call from a random state: the output and the state within
 *      tolerance of the CPU; the ring's older columns byte for byte (they
 *      are the old ring shifted), its newest column -- this token's qkv,
 *      the dnproj GEMV's output, summed on the device in another order --
 *      within tolerance;
 *   2. 64 calls in a row, both sides carrying their own state: every output
 *      and the final state within a bound;
 *   3. the same call twice from the same state: the same bytes;
 *   4. state_upload then state_download returns the bytes it was given;
 *      state_zero leaves zeros;
 *   5. COLI_GPU_FAIL_AFTER=0 (the backend's fault hook): decode returns 0,
 *      leaves the output buffer and the state as they were;
 *   6. create refuses projections of the wrong shape, and a missing weight.
 * The tolerances cover two GEMV reductions in another order on top of the
 * kernels' (tests/test_qwen36_deltanet_cuda.cu); the errors are printed.
 *
 * --bench: the 35B shape, 30 handles with their own projections (about 1 GB
 * of VRAM), COLI_CUDA_KEEPALIVE=1 as in the measured engine runs: the wall
 * time per layer of one decode call, against the two coli_cuda_matmul calls
 * the engine makes today for the same layer (without the CPU work between
 * them, which the new call replaces).
 *
 * Build: nvcc -O3 -std=c++17 -arch=native tests/test_qwen36_deltanet_decode_cuda.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "../backend_cuda.cu"
#include "qwen36_deltanet_ref.h"

/* An empty value removes the variable: fault_injected() treats a variable
 * that is set, even to "", as armed. _putenv_s with "" removes it already. */
#ifdef _WIN32
static int set_env(const char *name, const char *value) { return _putenv_s(name, value); }
#else
static int set_env(const char *name, const char *value) { return *value ? setenv(name, value, 1) : unsetenv(name); }
#endif

static int g_fail = 0;
static void check(bool ok, const char *what, double v, double tol) {
    printf("  %-50s %.3e  (tol %.0e)  %s\n", what, v, tol, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}
static void check_flag(bool ok, const char *what) {
    printf("  %-50s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

/* Per-row int8: w [O][I], one scale per row, max|row|/127 (qwen36's dense copies). */
struct I8 { std::vector<int8_t> q; std::vector<float> sc; int I, O; };
static I8 make_i8(int I, int O, float amp) {
    I8 m; m.I = I; m.O = O; m.q.resize((size_t)I * O); m.sc.resize(O);
    std::vector<float> row(I);
    for (int o = 0; o < O; o++) {
        float mx = 0.f;
        for (int i = 0; i < I; i++) { row[i] = urand(-amp, amp); mx = std::max(mx, fabsf(row[i])); }
        float s = mx > 0.f ? mx / 127.f : 1.f; m.sc[o] = s;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(row[i] / s); m.q[(size_t)o * I + i] = (int8_t)std::max(-127, std::min(127, v)); }
    }
    return m;
}
static void gemv_i8(const I8 &m, const float *x, float *y) {
    for (int o = 0; o < m.O; o++) {
        const int8_t *w = &m.q[(size_t)o * m.I]; float acc = 0.f;
        for (int i = 0; i < m.I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * m.sc[o];
    }
}

struct Case {
    DnShape s; Layer L; I8 proj, out;
    ColiCudaTensor *tp = nullptr, *to = nullptr;
    ColiCudaDeltaNet *h = nullptr;
};
static bool make_case(Case &c, DnShape s) {
    c.s = s; c.L = make_layer(&s);
    const int conv_dim = dn_conv_dim(&s), value_dim = dn_value_dim(&s);
    c.proj = make_i8(s.hidden, conv_dim + value_dim, 1.f / sqrtf((float)s.hidden) * 3.f);
    c.out = make_i8(value_dim, s.hidden, 1.f / sqrtf((float)value_dim));
    if (!coli_cuda_tensor_upload(&c.tp, c.proj.q.data(), c.proj.sc.data(), 1, c.proj.I, c.proj.O, 0) ||
        !coli_cuda_tensor_upload(&c.to, c.out.q.data(), c.out.sc.data(), 1, c.out.I, c.out.O, 0)) return false;
    return coli_cuda_deltanet_create(&c.h, c.tp, c.to, s.hidden, s.vh, s.vk, s.kdim, s.vdim, s.convk, s.eps,
                                     c.L.wab.data(), c.L.alog.data(), c.L.dtbias.data(),
                                     c.L.wconv.data(), c.L.normw.data()) != 0;
}
static void free_case(Case &c) {
    coli_cuda_deltanet_free(c.h); c.h = nullptr;
    coli_cuda_tensor_free(c.tp); coli_cuda_tensor_free(c.to); c.tp = c.to = nullptr;
}
/* The engine's chain on the CPU: dnproj, deltanet()'s middle, dnout. */
static void ref_layer(Case &c, const std::vector<float> &x, std::vector<float> &ring, std::vector<float> &rec,
                      std::vector<float> &out) {
    const DnShape &s = c.s; const int conv_dim = dn_conv_dim(&s), value_dim = dn_value_dim(&s);
    std::vector<float> qkvz(conv_dim + value_dim), beta(s.vh), g(s.vh), conv_out(conv_dim), outr(value_dim);
    gemv_i8(c.proj, x.data(), qkvz.data());
    dn_ref_step(&s, x.data(), qkvz.data(), c.L.wab.data(), c.L.alog.data(), c.L.dtbias.data(), c.L.wconv.data(),
                c.L.normw.data(), ring.data(), rec.data(), beta.data(), g.data(), conv_out.data(), outr.data());
    out.resize(s.hidden);
    gemv_i8(c.out, outr.data(), out.data());
}

static void run_shape(const char *name, DnShape s) {
    printf("%s: hidden %d, heads %d/%d, dims %d/%d, conv %d x %d\n",
           name, s.hidden, s.vh, s.vk, s.kdim, s.vdim, dn_conv_dim(&s), s.convk);
    Case c;
    if (!make_case(c, s)) { check_flag(false, "upload the projections and create the handle"); free_case(c); return; }
    const size_t nring = (size_t)dn_conv_dim(&s) * (s.convk - 1), nrec = (size_t)s.vh * s.kdim * s.vdim;
    std::vector<float> x(s.hidden), ring(nring), rec(nrec), out, gout(s.hidden), gring(nring), grec(nrec);

    /* 1. one call from a random state */
    fill(x, -1.7f, 1.7f); fill(ring, -2.f, 2.f); fill(rec, -0.5f, 0.5f);
    std::vector<float> ring0 = ring, rec0 = rec;
    check_flag(coli_cuda_deltanet_state_upload(c.h, rec0.data(), ring0.data()) != 0, "state_upload");
    ref_layer(c, x, ring, rec, out);
    check_flag(coli_cuda_deltanet_decode(c.h, x.data(), gout.data()) != 0, "decode returns 1");
    check_flag(coli_cuda_deltanet_state_download(c.h, grec.data(), gring.data()) != 0, "state_download");
    check(rel_err(gout, out) < 2e-5, "one call: output", rel_err(gout, out), 2e-5);
    check(rel_err(grec, rec) < 1e-5, "one call: state", rel_err(grec, rec), 1e-5);
    {
        const int cols = s.convk - 1; bool old_same = true;
        std::vector<float> gnew, cnew;
        for (size_t cc = 0; cc < nring / cols; cc++) {
            for (int k = 0; k < cols - 1; k++) old_same &= gring[cc * cols + k] == ring[cc * cols + k];
            gnew.push_back(gring[cc * cols + cols - 1]); cnew.push_back(ring[cc * cols + cols - 1]);
        }
        check_flag(old_same, "one call: ring, older columns byte for byte");
        check(rel_err(gnew, cnew) < 1e-5, "one call: ring, newest column (this token's qkv)", rel_err(gnew, cnew), 1e-5);
    }

    /* 3. the same call again from the same state: the same bytes */
    std::vector<float> gout2(s.hidden, NAN);
    coli_cuda_deltanet_state_upload(c.h, rec0.data(), ring0.data());
    check_flag(coli_cuda_deltanet_decode(c.h, x.data(), gout2.data()) != 0 && gout2 == gout,
               "repeat: the same output bytes");

    /* 4. state round trip and zero */
    std::vector<float> rr(nrec), rg(nring);
    coli_cuda_deltanet_state_upload(c.h, rec0.data(), ring0.data());
    coli_cuda_deltanet_state_download(c.h, rr.data(), rg.data());
    check_flag(rr == rec0 && rg == ring0, "state_upload then state_download: the same bytes");
    check_flag(coli_cuda_deltanet_state_zero(c.h) != 0, "state_zero returns 1");
    coli_cuda_deltanet_state_download(c.h, rr.data(), rg.data());
    bool zero = std::all_of(rr.begin(), rr.end(), [](float v) { return v == 0.f; }) &&
                std::all_of(rg.begin(), rg.end(), [](float v) { return v == 0.f; });
    check_flag(zero, "state_zero: zeros");

    /* 5. the fault hook: nothing moves */
    coli_cuda_deltanet_state_upload(c.h, rec0.data(), ring0.data());
    std::vector<float> sentinel(s.hidden, 12345.f), gs = sentinel;
    set_env("COLI_GPU_FAIL_AFTER", "0");
    int r = coli_cuda_deltanet_decode(c.h, x.data(), gs.data());
    set_env("COLI_GPU_FAIL_AFTER", "");
    coli_cuda_deltanet_state_download(c.h, rr.data(), rg.data());
    check_flag(r == 0, "COLI_GPU_FAIL_AFTER=0: decode returns 0");
    check_flag(gs == sentinel && rr == rec0 && rg == ring0, "COLI_GPU_FAIL_AFTER=0: output and state untouched");

    /* 2. 64 calls from a zero state */
    coli_cuda_deltanet_state_zero(c.h);
    std::fill(ring.begin(), ring.end(), 0.f); std::fill(rec.begin(), rec.end(), 0.f);
    double worst = 0; bool all_ok = true;
    for (int t = 0; t < 64; t++) {
        fill(x, -1.7f, 1.7f);
        ref_layer(c, x, ring, rec, out);
        all_ok &= coli_cuda_deltanet_decode(c.h, x.data(), gout.data()) != 0;
        worst = std::max(worst, rel_err(gout, out));
    }
    coli_cuda_deltanet_state_download(c.h, grec.data(), gring.data());
    check_flag(all_ok, "64 calls: every decode returns 1");
    check(worst < 5e-5, "64 calls: worst output", worst, 5e-5);
    check(rel_err(grec, rec) < 5e-5, "64 calls: final state", rel_err(grec, rec), 5e-5);

    /* 6. refusals */
    ColiCudaDeltaNet *bad = (ColiCudaDeltaNet *)1;
    int rb = coli_cuda_deltanet_create(&bad, c.to, c.tp, s.hidden, s.vh, s.vk, s.kdim, s.vdim, s.convk, s.eps,
                                       c.L.wab.data(), c.L.alog.data(), c.L.dtbias.data(), c.L.wconv.data(), c.L.normw.data());
    check_flag(rb == 0 && bad == nullptr, "create refuses swapped projections (and clears the handle)");
    bad = (ColiCudaDeltaNet *)1;
    rb = coli_cuda_deltanet_create(&bad, c.tp, c.to, s.hidden, s.vh, s.vk, s.kdim, s.vdim, s.convk, s.eps,
                                   c.L.wab.data(), nullptr, c.L.dtbias.data(), c.L.wconv.data(), c.L.normw.data());
    check_flag(rb == 0 && bad == nullptr, "create refuses a missing weight");
    free_case(c);
}

static void bench(void) {
    DnShape s = {2048, 32, 16, 128, 128, 4, 1e-6f};
    const int NL = 30, TOK = 200;
    printf("bench: 35B shape, %d layers, %d tokens, keep-alive on\n", NL, TOK);
    std::vector<Case> cs(NL);
    for (int l = 0; l < NL; l++) if (!make_case(cs[l], s)) { printf("  setup failed at layer %d  FAIL\n", l); g_fail++; return; }
    std::vector<float> x(s.hidden), out(s.hidden), qkvz(dn_conv_dim(&s) + dn_value_dim(&s)), outr(dn_value_dim(&s));
    fill(x, -1.7f, 1.7f); fill(outr, -1.f, 1.f);
    typedef std::chrono::steady_clock clk;
    for (int pass = 0; pass < 2; pass++) {       /* pass 0 warms up */
        auto t0 = clk::now();
        for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++)
            if (!coli_cuda_deltanet_decode(cs[l].h, x.data(), out.data())) { printf("  decode failed  FAIL\n"); g_fail++; return; }
        double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / (TOK * NL);
        if (pass) printf("  one decode call:           %7.2f us per layer, %.3f ms per token (30 layers)\n", us, us * NL / 1000.0);
    }
    for (int pass = 0; pass < 2; pass++) {
        auto t0 = clk::now();
        for (int t = 0; t < TOK; t++) for (int l = 0; l < NL; l++) {
            ColiCudaTensor *tp = cs[l].tp, *to = cs[l].to;
            if (!coli_cuda_matmul(&tp, qkvz.data(), x.data(), NULL, NULL, 1, 1, s.hidden, (int)qkvz.size(), 0, 0) ||
                !coli_cuda_matmul(&to, out.data(), outr.data(), NULL, NULL, 1, 1, (int)outr.size(), s.hidden, 0, 0)) {
                printf("  matmul failed  FAIL\n"); g_fail++; return;
            }
        }
        double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / (TOK * NL);
        if (pass) printf("  two coli_cuda_matmul calls: %7.2f us per layer (today's GPU part only, without the CPU work between)\n", us);
    }
    for (auto &c : cs) free_case(c);
}

int main(int argc, char **argv) {
    const bool b = argc > 1 && !strcmp(argv[1], "--bench");
    if (b) set_env("COLI_CUDA_KEEPALIVE", "1");
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("FATAL cuda init\n"); return 2; }
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) printf("device: %s, sm_%d%d\n", p.name, p.major, p.minor);
    if (b) bench();
    else {
        run_shape("35B", DnShape{2048, 32, 16, 128, 128, 4, 1e-6f});
        run_shape("tiny", DnShape{64, 8, 4, 8, 8, 4, 1e-6f});
    }
    coli_cuda_shutdown();
    printf("deltanet decode cuda: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail ? 1 : 0;
}
