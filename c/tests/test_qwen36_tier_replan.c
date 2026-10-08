/* qt_replan: the resident set follows the prompt (port of upstream 0fc8834b,
 * e4a2e3c3 and the test of b842448a, without the in-place overwrite and the
 * evicted-expert ring this port leaves out).
 *
 * The warmstart fills VRAM from a heat file, i.e. from what earlier prompts
 * routed to. A new prompt routes elsewhere, and the LFRU tick corrects that
 * one expert per sixteen tokens. qt_replan takes this prompt's own routing
 * counts (the engine's prefill counts) and swaps residents the prompt never
 * touched for its most-routed non-residents, budget-neutral, through the same
 * victim-first swap the tick uses, in strict count order and only while the
 * newcomer's count beats the victim's.
 *
 * Part 1: the plan (order, cap, per layer, whole table, budget, counters).
 * Part 2: what the upload queue cannot take stays pending and starts on the
 * layer-0 ticks, QT_REPLAN_PER_TICK at a time.
 * Part 3: under QT_UPLOAD_SYNC=1 a re-plan swap has landed by the next group,
 * as an LFRU swap has (CACHE_ROUTE determinism).
 * Fake CUDA backend, no GPU, no toolkit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

static int issue_ok(int device, int count, const float *x) { (void)device; (void)count; (void)x; return 1; }

static volatile int g_block, g_parked;   /* part 2: the uploader parks here while g_block is set */
static void block_upload(int fmt) {
    (void)fmt;
    struct timespec ts = {0, 1000000};
    if (g_block) g_parked = 1;
    while (g_block) nanosleep(&ts, NULL);
}
static void slow_upload(int fmt) {
    (void)fmt;
    struct timespec ts = {0, 50000000};   /* 50 ms a tensor: the uploader is behind */
    nanosleep(&ts, NULL);
}

static int resident_set(int layer, int ne, int *out) {   /* returns count, fills sorted eids */
    int n = 0;
    pthread_mutex_lock(&G.mx);
    for (int e = 0; e < ne; e++) if (qs(layer, e)->resident) out[n++] = e;
    pthread_mutex_unlock(&G.mx);
    return n;
}
static int same(const int *a, int na, const int *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static int pending(void) { pthread_mutex_lock(&G.mx); int p = G.rp_n - G.rp_i; pthread_mutex_unlock(&G.mx); return p; }

enum { D = 64, IH = 32 };
static size_t exp_bytes(void) {
    return 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
}
static void set_budget(double experts) {
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", experts * (double)exp_bytes() / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
}

static void part1(void) {
    enum { NL = 2, NE = 8, TOPK = 2 };
    setenv("QT_UPLOAD_SYNC", "", 1);
    set_budget(8.5);   /* exactly eight experts, four per layer once noted in interleaved order */
    fake_uploads = 0; fake_upload_hook = NULL;
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "part 1: tier starts (int4, per-row scales)");

    static unsigned char g4[NL][NE][D * IH / 2], u4[NL][NE][D * IH / 2], d4[NL][NE][D * IH / 2];
    static float sc[NL][NE][2 * IH + D];
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) {
        memset(g4[l][e], (unsigned char)(e + 1), sizeof g4[l][e]); memset(u4[l][e], (unsigned char)(e + 2), sizeof u4[l][e]);
        memset(d4[l][e], (unsigned char)(e + 3), sizeof d4[l][e]);
        for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.0f;
        /* experts 0..3 of both layers fit the budget and go up; 4..7 only leave
         * their RAM pointers with the tier, as a real warmstart does for the
         * experts it cannot place */
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    }
    qt_fill_wait();
    int res[NE]; int n = resident_set(0, NE, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "warmstart: layer 0 experts 0..3 resident, 4..7 not"); }
    n = resident_set(1, NE, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "warmstart: layer 1 experts 0..3 resident, 4..7 not"); }
    check(fake_uploads == 3 * 8, "eight experts uploaded (three tensors each)");
    size_t used0 = G.used[0];
    uint64_t lfru0 = G.swaps;

    /* the prompt's counts: never 0 and 1, a little 2 and 3, mostly 4 and 5,
     * once 6, never 7 */
    uint32_t freq[NE] = {0, 0, 5, 7, 10, 9, 1, 0};

    /* layer 0, cap 1: only the best pair, 4 (10) for 0 (count 0, the coldest) */
    check(qt_replan(0, freq, 1) == 1, "cap 1 plans exactly one swap");
    qt_fill_wait();
    pthread_mutex_lock(&G.mx);
    check(qs(0, 4)->tg != NULL && qs(0, 0)->tg == NULL, "the newcomer holds tensors, the victim none");
    pthread_mutex_unlock(&G.mx);
    check(fake_uploads == 3 * 9, "the swap is a fresh upload of the newcomer (no overwrite in this backend)");
    n = resident_set(0, NE, res);
    { int want[] = {1, 2, 3, 4}; check(same(res, n, want, 4), "after the first swap: 4 in, 0 out"); }
    n = resident_set(1, NE, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "a layer-0 re-plan leaves layer 1 alone"); }
    check(G.used[0] == used0, "a swap is budget-neutral");
    check(G.swaps == lfru0 && G.rp_done == 1, "the swap counts as a re-plan swap, not an LFRU one");

    /* no cap: 5 (9) for 1 (count 0); 6 (1) against the next victim 2 (5) is
     * not worth a swap, so the plan stops there */
    check(qt_replan(0, freq, 100) == 1, "the second plan holds one pair: a newcomer must beat its victim's count");
    qt_fill_wait();
    n = resident_set(0, NE, res);
    { int want[] = {2, 3, 4, 5}; check(same(res, n, want, 4), "after the second swap: 5 in, 1 out; 2 and 3 stay"); }
    check(pending() == 0, "nothing left pending: the queue was idle, the plan started at once");

    /* the same counts again: the set already matches the prompt */
    check(qt_replan(0, freq, 100) == 0, "a re-plan on the same counts plans nothing");
    check(qt_replan(0, NULL, 100) == 0 && qt_replan(0, freq, 0) == 0 && qt_replan(NL, freq, 100) == 0,
          "NULL counts, a zero cap or a layer out of range plan nothing");

    /* ties: two never-routed residents of equal heat; the lower slot index
     * goes first whatever qsort does (upstream e4a2e3c3) */
    uint32_t freq1[NE] = {9, 0, 0, 0, 0, 0, 0, 10};
    pthread_mutex_lock(&G.mx); for (int e = 0; e < NE; e++) qs(1, e)->heat = 0; pthread_mutex_unlock(&G.mx);
    check(qt_replan(1, freq1, 100) == 1, "layer 1: one swap, 7 for one of the count-0 residents");
    qt_fill_wait();
    n = resident_set(1, NE, res);
    { int want[] = {0, 2, 3, 7}; check(same(res, n, want, 4), "layer 1 keeps 0, and the tie went to the lowest index (1 out)"); }
    n = resident_set(0, NE, res);
    { int want[] = {2, 3, 4, 5}; check(same(res, n, want, 4), "layer 0 untouched by the layer-1 plan"); }

    /* the whole-table form (layer < 0) plans across layers, per device: layer
     * 0's expert 6 (routed once) is worth more than layer 1's never-routed
     * residents, which the per-layer form cannot trade against each other */
    uint32_t all[NL * NE]; memcpy(all, freq, sizeof freq); memcpy(all + NE, freq1, sizeof freq1);
    check(qt_replan(-1, all, 100) == 1, "the whole-table form plans one cross-layer swap");
    qt_fill_wait();
    check(resident_set(0, NE, res) == 5 && resident_set(1, NE, res) == 3, "layer 0 gained a resident, layer 1 lost one");
    check(G.used[0] == used0, "still budget-neutral");
    check(qt_replan(-1, all, 100) == 0, "and then nothing is left worth a swap");
    check(G.rp_planned == 4 && G.rp_done == 4 && G.swaps == lfru0, "four re-plan swaps planned and started, no LFRU swap");

    /* the decode marker: stats after the mark count from here */
    qt_stats_mark();
    check(G.mk_on && G.mk_hits == 0 && G.mk_miss == 0, "the marker snapshots the counters (none yet)");
    qt_stats();
    qt_shutdown();
}

static void part2(void) {
    /* one layer, 128 experts, budget for 60: residents 0..59 never routed,
     * non-residents 60..127 routed 1..68 times. With the uploader parked the
     * queue fills (QT_QCAP) and the rest of the 60 pairs stays pending. */
    enum { NE = 128, RES = 60 };
    setenv("QT_UPLOAD_SYNC", "", 1);
    set_budget(RES + 0.5);
    fake_uploads = 0; fake_upload_hook = NULL; fake_issue_hook = issue_ok;
    check(qt_init(1, NE, D, IH, NE, 1, 0, 1), "part 2: tier starts");
    static unsigned char g4[NE][D * IH / 2], u4[NE][D * IH / 2], d4[NE][D * IH / 2];
    static float sc[NE][2 * IH + D];
    for (int e = 0; e < NE; e++) {
        memset(g4[e], (unsigned char)e, sizeof g4[e]); memset(u4[e], (unsigned char)e, sizeof u4[e]);
        memset(d4[e], (unsigned char)e, sizeof d4[e]); for (int i = 0; i < 2 * IH + D; i++) sc[e][i] = 1.0f;
        qt_note_block(0, e, g4[e], u4[e], d4[e], sc[e], sc[e] + IH, sc[e] + 2 * IH);
    }
    qt_fill_wait();
    int res[NE];
    check(resident_set(0, NE, res) == RES, "part 2: sixty residents");
    uint32_t cnt[NE];
    for (int e = 0; e < NE; e++) cnt[e] = e < RES ? 0 : (uint32_t)(e - RES + 1);

    g_block = 1; g_parked = 0; fake_upload_hook = block_upload;
    check(qt_replan(0, cnt, RES) == RES, "sixty pairs planned");
    { struct timespec ts = {0, 1000000}; int w = 0; while (!g_parked && w++ < 10000) nanosleep(&ts, NULL); }
    check(g_parked, "the uploader took the first swap and parked");
    /* the uploader freed one queue slot when it took that swap: one tick
     * fills it, after which the queue stays full while the uploader is parked */
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    float out[D]; float val[1] = {1};
    int eid = RES - 1;
    uint32_t mask = qt_issue(0, &eid, 1, x); qt_take(mask, val, 1, out);
    int p0 = pending();
    check(p0 > 0 && p0 < RES, "the queue took part of the plan, the rest is pending");
    pthread_mutex_lock(&G.mx); int qn = G.qn; uint64_t lfru0 = G.swaps; pthread_mutex_unlock(&G.mx);
    check(qn == QT_QCAP, "the parked uploader leaves the queue full");

    /* a tick with the queue full starts nothing */
    mask = qt_issue(0, &eid, 1, x); qt_take(mask, val, 1, out);
    check(pending() == p0, "a tick with the queue full starts nothing");

    g_block = 0;
    qt_fill_wait();
    check(pending() == p0, "the queue drains; the pending pairs wait for a tick or a call");

    /* each layer-0 tick starts at most QT_REPLAN_PER_TICK */
    int ticks = 0, ok_rate = 1;
    while (pending() > 0 && ticks < 100) {
        int before = pending();
        mask = qt_issue(0, &eid, 1, x); qt_take(mask, val, 1, out);
        int started = before - pending();
        if (started < 1 || started > QT_REPLAN_PER_TICK) ok_rate = 0;
        ticks++;
        qt_fill_wait();
    }
    check(ok_rate, "every tick starts between one and QT_REPLAN_PER_TICK pending pairs");
    check(ticks == (p0 + QT_REPLAN_PER_TICK - 1) / QT_REPLAN_PER_TICK, "the pending pairs drain in ceil(pending / per-tick) ticks");
    qt_fill_wait();
    int n = resident_set(0, NE, res);
    int top = n == RES; for (int i = 0; i < n && top; i++) if (res[i] != NE - RES + i) top = 0;
    check(top, "the sixty residents are now the sixty most-routed experts (68..127)");
    pthread_mutex_lock(&G.mx); int lfru_same = G.swaps == lfru0, done = (int)G.rp_done; pthread_mutex_unlock(&G.mx);
    check(lfru_same && done == RES, "sixty re-plan swaps started, no LFRU swap");
    check(fake_uploads == 3 * (RES + RES), "sixty newcomers uploaded after the sixty residents");
    fake_upload_hook = NULL; fake_issue_hook = NULL;
    qt_shutdown();
}

static void part3(void) {
    /* QT_UPLOAD_SYNC=1, slow uploads: a re-plan swap planned between two
     * groups has landed when the next group is formed, on any layer */
    setenv("QT_UPLOAD_SYNC", "1", 1);
    set_budget(1.5);
    fake_uploads = 0; fake_upload_hook = NULL; fake_issue_hook = issue_ok;
    check(qt_init(2, 2, D, IH, 2, 1, 0, 1), "part 3: tier starts under QT_UPLOAD_SYNC=1");
    static unsigned char g4[2][2][D * IH / 2], u4[2][2][D * IH / 2], d4[2][2][D * IH / 2];
    static float sc[2][2][2 * IH + D];
    for (int l = 0; l < 2; l++) for (int e = 0; e < 2; e++) {
        memset(g4[l][e], (unsigned char)(e + 1), sizeof g4[l][e]); memset(u4[l][e], (unsigned char)(e + 1), sizeof u4[l][e]);
        memset(d4[l][e], (unsigned char)(e + 1), sizeof d4[l][e]); for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.0f;
    }
    /* layer 1 expert 0 takes the one-expert budget; expert 1 is refused */
    qt_note(1, 0, g4[1][0], u4[1][0], d4[1][0], sc[1][0], sc[1][0] + IH, sc[1][0] + 2 * IH);
    qt_note(1, 1, g4[1][1], u4[1][1], d4[1][1], sc[1][1], sc[1][1] + IH, sc[1][1] + 2 * IH);
    qt_fill_wait();
    check(qt_is_resident(1, 0) && !qt_is_resident(1, 1), "part 3: layer 1 expert 0 resident, expert 1 not");
    fake_upload_hook = slow_upload;
    uint32_t cnt[2] = {0, 5};
    check(qt_replan(1, cnt, 24) == 1, "part 3: one swap planned for layer 1");
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    float out[D]; float val[1] = {1};
    int eid = 1;
    uint32_t mask = qt_issue(1, &eid, 1, x);
    check(mask == 1u, "under QT_UPLOAD_SYNC the next group (layer 1, not a tick) finds the newcomer resident");
    qt_take(mask, val, 1, out);
    fake_upload_hook = NULL; fake_issue_hook = NULL;
    qt_shutdown();
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1);
    fake_ndev = 1;
    part1();
    part2();
    part3();
    if (fails) { printf("test_qwen36_tier_replan: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_replan: the resident set follows the prompt's counts, budget-neutral; pending pairs start on the ticks; QT_UPLOAD_SYNC lands them\n");
    return 0;
}
