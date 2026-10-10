/* QT_ASYNC_ISSUE=1: the expert group launch on a helper thread.
 *
 * What has to hold, on the fake backend (tests/qwen36_fake_cuda.h):
 *   1. off unless exactly "1"; refused, with a message, when the engine
 *      did not call qt_async_allow() or on more than one card;
 *   2. on: qt_issue returns the resident mask, the launch happens on ANOTHER
 *      thread, and qt_take accumulates exactly what the synchronous path
 *      accumulates;
 *   3. a refused launch comes back from qt_take_redo() as the k it covered,
 *      nothing of it reaches `out`, and the miss/hit counters say so -- the
 *      engine computes those k on the CPU;
 *   4. many issues in a row, with the helper sleeping between every one
 *      (QT_ASYNC_SPIN_US=0) and with it spinning, never lose a wake-up (a
 *      lost one hangs: a watchdog turns that into a failure);
 *   5. qt_shutdown joins the helper.
 * The fake take returns, for device d and c rows, row j filled with
 * 100*d + j + 1, so the expected sum is known exactly. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_expert_group_take

enum { NL = 1, NE = 16, D = 8, IH = 4, K = 8 };

static pthread_t issue_thread;
static int issue_calls, issue_on_other_thread, issue_answer = 1, last_count;
static int record_issue(int device, int count, const float *x) {
    (void)device; (void)x;
    issue_calls++;
    if (!pthread_equal(pthread_self(), issue_thread)) issue_on_other_thread++;
    last_count = count;
    return issue_answer;
}
static float take_rows[K * D];
const float *coli_cuda_expert_group_take(int device) {
    for (int j = 0; j < last_count; j++)
        for (int d = 0; d < D; d++) take_rows[j * D + d] = (float)(100 * device + j + 1);
    return take_rows;
}

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

#ifndef _WIN32
#include <unistd.h>
static void on_alarm(int sig) { (void)sig; (void)!write(2, "FAIL: async issue hung (lost wake-up?)\n", 39); _exit(1); }
#endif

static unsigned char g4[NE][D * IH / 2], u4[NE][D * IH / 2], d4[NE][D * IH / 2];
static float sc[NE][2 * IH + D];

static int up(const char *async, const char *spin) {
    setenv("COLI_CUDA", "1", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    if (async) setenv("QT_ASYNC_ISSUE", async, 1); else unsetenv("QT_ASYNC_ISSUE");
    if (spin) setenv("QT_ASYNC_SPIN_US", spin, 1); else unsetenv("QT_ASYNC_SPIN_US");
    if (!qt_init(NL, NE, D, IH, NE, K, 0 /* per-row */, 1 /* int4 */)) return 0;
    for (int eid = 0; eid < NE; eid++) {
        memset(g4[eid], eid + 1, sizeof g4[eid]); memset(u4[eid], eid + 2, sizeof u4[eid]);
        memset(d4[eid], eid + 3, sizeof d4[eid]);
        for (int i = 0; i < 2 * IH + D; i++) sc[eid][i] = 1.0f;
        qt_note_block(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + IH, sc[eid] + 2 * IH);
    }
    qt_fill_wait();
    return 1;
}

/* One issue..take with all K experts resident; returns qt_take's status. */
static int one_round(float *out, uint32_t *mask_out) {
    int eids[K]; for (int k = 0; k < K; k++) eids[k] = k;
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    float val[K]; for (int k = 0; k < K; k++) val[k] = 0.5f + k;
    memset(out, 0, D * sizeof *out);
    uint32_t mask = qt_issue(0, eids, K, x);
    if (mask_out) *mask_out = mask;
    return qt_take(mask, val, K, out);
}

int main(void) {
    /* An engine that never calls qt_async_allow() (qwen38) must not get the
     * helper: it would drop every refused expert. */
    check(up("1", NULL), "tier init without qt_async_allow");
    check(!A.on, "QT_ASYNC_ISSUE=1 must be refused unless the engine called qt_async_allow()");
    qt_shutdown();
    qt_async_allow();
#ifndef _WIN32
    signal(SIGALRM, on_alarm); alarm(60);
#endif
    issue_thread = pthread_self();
    fake_issue_hook = record_issue;

    /* expected: row j = j+1 (device 0), weight 0.5+j */
    float want = 0; for (int j = 0; j < K; j++) want += (0.5f + j) * (float)(j + 1);

    /* ---- 1. off unless exactly "1" ---- */
    check(up("2", NULL), "tier init (QT_ASYNC_ISSUE=2)");
    check(!A.on, "QT_ASYNC_ISSUE=2 must leave the helper off");
    float sync_out[D]; uint32_t m0;
    issue_calls = issue_on_other_thread = 0;
    check(one_round(sync_out, &m0) && m0 == 0xFFu, "synchronous round");
    check(issue_calls == 1 && issue_on_other_thread == 0, "synchronous launch must run on the decode thread");
    check(sync_out[0] == want, "synchronous sum");
    check(qt_take_redo() == 0, "no redo without the switch");
    qt_shutdown();

    /* ---- 2. on: launch elsewhere, same sum ---- */
    check(up("1", NULL), "tier init (QT_ASYNC_ISSUE=1)");
    check(A.on && A.spin_us == 2000, "QT_ASYNC_ISSUE=1 must start the helper, default spin 2000 us");
    float out[D]; uint32_t m1;
    issue_calls = issue_on_other_thread = 0;
    check(one_round(out, &m1) && m1 == 0xFFu, "async round");
    check(issue_calls == 1 && issue_on_other_thread == 1, "async launch must run on the helper thread");
    check(memcmp(out, sync_out, sizeof out) == 0, "async sum must equal the synchronous one, bit for bit");
    check(qt_take_redo() == 0, "an accepted launch leaves no redo");

    /* ---- 3. a refused launch ---- */
    uint64_t miss0 = G.miss, hits0 = G.hits[0];
    issue_answer = 0;
    uint32_t m2;
    int ok = one_round(out, &m2);
    check(ok, "a refused async launch is not a collection failure");
    check(m2 == 0xFFu, "qt_issue returns the resident mask before the backend answers");
    check(qt_take_redo() == 0xFFu, "every refused k must come back from qt_take_redo()");
    int untouched = 1; for (int d = 0; d < D; d++) if (out[d] != 0.f) untouched = 0;
    check(untouched, "nothing of a refused launch may reach out");
    check(G.miss == miss0 + K && G.hits[0] == hits0, "a refused launch counts as K misses, not hits");
    check(A.refused == K, "the refusal is counted for qt_stats");
    issue_answer = 1;
    check(one_round(out, NULL) && out[0] == want && qt_take_redo() == 0, "an accepted launch after a refusal");

    /* ---- 4. many rounds, spinning helper ---- */
    int bad = 0;
    for (int r = 0; r < 2000; r++) if (!one_round(out, NULL) || out[0] != want) bad++;
    check(bad == 0, "2000 rounds with a spinning helper");
    qt_stats();
    qt_shutdown();                                   /* 5: joins the helper */
    check(!A.on, "qt_shutdown must stop the helper");

    /* ---- 4b. sleeping helper: every job needs a wake-up ---- */
    check(up("1", "0"), "tier init (QT_ASYNC_SPIN_US=0)");
    check(A.on && A.spin_us == 0, "QT_ASYNC_SPIN_US=0 must be honoured");
    bad = 0;
    for (int r = 0; r < 2000; r++) if (!one_round(out, NULL) || out[0] != want) bad++;
    check(bad == 0, "2000 rounds with a helper that sleeps between jobs");
    qt_shutdown();

    /* ---- 1b. one card that is not device 0: allowed (the DLL's device
     * cache is thread_local); two cards: refused ---- */
    setenv("COLI_GPUS", "1", 1); fake_ndev = 2;
    check(up("1", NULL), "tier init on device 1");
    check(A.on, "QT_ASYNC_ISSUE=1 must start on a single card that is device 1");
    check(one_round(out, NULL) && qt_take_redo() == 0, "a round on device 1");
    qt_shutdown();
    setenv("COLI_GPUS", "0,1", 1);
    check(up("1", NULL), "tier init on two cards");
    check(!A.on, "QT_ASYNC_ISSUE=1 must be refused on two cards");
    qt_shutdown();

    if (fails) { printf("tier async issue: %d failure(s)\n", fails); return 1; }
    puts("tier async issue: ok");
    return 0;
}
