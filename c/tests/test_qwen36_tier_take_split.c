/* The wait/accum split inside qt_take is the one part of qt_take that upstream
 * does not have: the two-phase form came from there, the instrumentation did
 * not, and the merge of the two had to decide where the clock is read. Nothing
 * exercised it -- every other tier test leaves g_qt_time_take at 0, so `_tm` is
 * false and the three timing lines never run.
 *
 * WHAT IS ASSERTED. Each point is worded as the property the assertion
 * establishes, which is not always the property one would like it to mean; two
 * earlier versions of this header overstated exactly that, and each time a
 * plausible mutation walked through the gap.
 *
 *   A. wait >= NDEV*SLEEP_MS: the drain is inside the wait window. The fake
 *      take sleeps, so this is a bound nanosleep guarantees rather than a
 *      timing guess. It is the most productive assertion here: it also catches
 *      re-reading _w0 inside the drain, scaling wait down, swapping the two
 *      accumulators, and CLOCK_PROCESS_CPUTIME_ID (which does not advance
 *      while the fake sleeps).
 *   B. wait + acc <= the wall time of the whole call, measured from outside.
 *      This establishes that the two phases TOGETHER do not exceed the call:
 *      it excludes charging a phase twice, a boundary read outside the call,
 *      `+` where `-` belongs, and timing acc from _w0 so that it swallows the
 *      whole wait. It does NOT establish that the two windows are disjoint,
 *      nor where the boundary between them sits: two intervals inside a window
 *      of length L can overlap entirely and still sum to less than L. See NOT
 *      COVERED.
 *      It holds for a stronger reason than "the bound loosens under load",
 *      which is false -- measured, the slack does not widen on a contended
 *      host and its minimum drops. It holds by monotonicity: wait + acc is
 *      (end of acc) - _w0, both clock reads lie inside the bracketed call, and
 *      both differences are exact in floating point (the operands are within a
 *      factor of two). The slack is therefore 1 ns, for the rounding of the
 *      sum itself, not the 1 us an earlier version used -- that figure was the
 *      ulp of tv_sec*1e3 for CLOCK_REALTIME, which a monotonic clock would
 *      need some fifty years of uptime to reach, and it was wide enough to
 *      hide a constant added to wait.
 *   C. acc >= 5 us: the publish loop is inside the accumulation window, not
 *      merely a gap between two adjacent clock reads. 2 devices x 16 rows x
 *      D=4096 = 131072 multiply-adds measure 15-86 us depending on load and
 *      build, against ~20-31 ns for two back-to-back clock reads, so the
 *      threshold sits about 3x below the real work under contention and 10x
 *      below it at rest, and three orders of magnitude above the noise.
 *   D. Disarmed, NOT ONE CLOCK IS READ -- counted, not inferred from the
 *      counters staying at zero. Charging nothing while still reading two
 *      clocks per layer per token would keep the letter of the production
 *      comment and break its point.
 *   E. Armed, EXACTLY THREE clocks are read. This pins the premise that makes
 *      the counting meaningful: qwen36_tier.c calls clock_gettime in one place
 *      (qt_ms) and qt_ms is called only from qt_take, so a fourth read means
 *      either a new read in qt_take or a new caller of qt_ms elsewhere in the
 *      file -- and in the second case this test's counter would silently be
 *      measuring something else. It fails loudly instead.
 *   F. Disarmed and armed produce a bit-identical `out`, and that `out` is not
 *      all zeroes. Without the second half the comparison is vacuous: a
 *      publish loop that runs but writes nothing satisfies it.
 *   G. mask == 0 reads no clock AND charges nothing, asserted as bit equality
 *      of both counters. The `&& mask` in `_tm` is local to this tree. Without
 *      it an engine with COLI_TIMERS on charges a ~0 wait for every layer with
 *      no experts in VRAM: not zero in aggregate, but a per-layer constant
 *      added to the arm with the FEWEST resident experts, i.e. a bias pointing
 *      the wrong way in exactly the comparison the split exists to make.
 *   H. A device with no rows in flight does not stop the drain of the next one
 *      (`continue`, not `break`), counted through the fake take's counter.
 *   I. A failed device is still charged its wait, and publishes nothing.
 *   J. Only CLOCK_MONOTONIC is read. The fixture sees the clock id, so this is
 *      a one-line check; an earlier version of this header declared the same
 *      property uncoverable "without injecting the clock", while the clock was
 *      already injected. It catches CLOCK_REALTIME (a wall-clock jump would
 *      produce negative or enormous waits), and CLOCK_MONOTONIC_RAW and
 *      CLOCK_BOOTTIME, which that version did not even name.
 *
 * NOT COVERED, stated rather than left to be discovered:
 *
 *   - An UNDERCHARGE of acc that leaves it above the (C) threshold. Charging
 *     acc for only one of the two devices -- a plausible refactor, e.g. to
 *     report acc per card -- halves it and passes. That matters here: it would
 *     report the accumulation as 5 % of take where the record measures 10 %.
 *     The obvious fix, a closing bound wait + acc >= call - delta, was written
 *     and measured, and it is NOT used: the residue outside both windows
 *     (prologue plus the mutex/broadcast tail) is preemptible, and at delta =
 *     10 us it produced 3 false failures in 300 runs on a contended CPU. A
 *     gate that fails under load is a gate that gets deleted, so the gap is
 *     declared instead of papered over with an intermittent assertion.
 *   - Moving the boundary between the two windows in a way that preserves
 *     their sum: (B) cannot see it, by construction.
 *   - A constant added to either counter that is smaller than the residue
 *     lying OUTSIDE both windows -- the prologue plus the mutex/broadcast
 *     tail, measured at 0.3-3 us here. Verified: a 0.9 us constant added to
 *     wait passes. (B)'s slack is not what hides it and shrinking that slack
 *     from 1 us to 1 ns does not change the outcome; the residue does, and
 *     only the closing bound rejected above could reach it.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NDEV     2
#define NROW     16            /* per device */
#define DIM      4096
#define SLEEP_MS 4

/* Counts qt_take's clock reads and refuses any clock but CLOCK_MONOTONIC; see
 * (D), (E), (J). Defined before the #define below, or the forwarding call would
 * expand into itself. */
static long clock_reads;
static int fixture_clock_gettime(clockid_t c, struct timespec *t) {
    if (c != CLOCK_MONOTONIC) {
        fprintf(stderr, "tier take split: FAIL qt_ms read clock id %d, not "
                        "CLOCK_MONOTONIC (%d)\n", (int)c, (int)CLOCK_MONOTONIC);
        exit(1);
    }
    clock_reads++;
    return clock_gettime(c, t);
}

#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_expert_group_take

static int failed = -1, calls[NDEV];
static float *rows[NDEV];

const float *coli_cuda_expert_group_take(int device) {
    calls[device]++;
    /* Restarted on EINTR: a truncated sleep would fail (A) for a reason that
     * has nothing to do with the code under test. */
    struct timespec ts = {0, SLEEP_MS * 1000000L};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) ;
    return device == failed ? NULL : rows[device];
}

#define clock_gettime fixture_clock_gettime
#include "../qwen36_tier.c"
#undef clock_gettime

/* After the include, where QT_MAX_DEV and QT_MAX_ROWS exist. The mask is a
 * uint32_t and its bits index `val`, so the rows in flight across every device
 * must fit in 32; the fixture hardcodes device 1, and G.dev[] and G.is_k[] have
 * their own ceilings. A comment was not enough -- two configurations that
 * satisfied "NDEV * NROW <= 32" alone wrote past the end of an array. */
_Static_assert(NDEV * NROW <= 32,      "mask is a uint32_t");
_Static_assert(NDEV >= 2,              "the idle-device case needs two devices");
_Static_assert(NDEV <= QT_MAX_DEV,     "G.dev[] and G.is_cnt[] are QT_MAX_DEV wide");
_Static_assert(NROW <= QT_MAX_ROWS,    "G.is_k[] is QT_MAX_ROWS wide");

static float val[NDEV * NROW];

/* One layer's worth of in-flight experts: NROW rows on each device, the k
 * indices partitioned so every bit of the mask is claimed exactly once. */
static void arm_issue(void) {
    G.issue_open = 1;
    for (int di = 0; di < NDEV; di++) {
        G.is_cnt[di] = NROW;
        for (int j = 0; j < NROW; j++) G.is_k[di][j] = di * NROW + j;
        calls[di] = 0;
    }
}

static double now_ms(void) {          /* the real clock: the #define is undone */
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int fail(const char *what) {
    fprintf(stderr, "tier take split: FAIL %s (wait=%.6f acc=%.6f reads=%ld)\n",
            what, g_qt_wait, g_qt_acc, clock_reads);
    return 1;
}

int main(void) {
    G.on = 1; G.ndev = NDEV; G.D = DIM;
    for (int di = 0; di < NDEV; di++) {
        G.dev[di] = di;
        rows[di] = malloc((size_t)NROW * DIM * sizeof *rows[di]);
        if (!rows[di]) return fprintf(stderr, "oom\n"), 1;
        for (int j = 0; j < NROW; j++)
            for (int d = 0; d < DIM; d++)
                rows[di][(size_t)j * DIM + d] = (float)((di * NROW + j) % 7) + 1.0f;
    }
    for (int k = 0; k < NDEV * NROW; k++) val[k] = 1.0f / (float)(k + 2);
    pthread_mutex_init(&G.mx, NULL); pthread_cond_init(&G.cv_take, NULL);

    /* Built with a RIGHT shift: `1u << 32` is undefined and the static assert
     * above permits exactly 32. Correct for any width in 1..32. */
    const uint32_t full = 0xFFFFFFFFu >> (32 - NDEV * NROW);
    const uint32_t lo   = 0xFFFFFFFFu >> (32 - NROW);      /* device 0's bits */
    float *ref  = calloc(DIM, sizeof *ref);
    float *seen = calloc(DIM, sizeof *seen);
    if (!ref || !seen) return fprintf(stderr, "oom\n"), 1;
    int rc = 0;

    /* (D)(F) disarmed: the reference arithmetic, and not one clock read. */
    g_qt_time_take = 0; g_qt_wait = g_qt_acc = 0; clock_reads = 0;
    arm_issue();
    if (!qt_take(full, val, NDEV * NROW, ref))       rc = fail("disarmed take reported failure");
    else if (clock_reads != 0)                       rc = fail("disarmed read a clock");
    else if (g_qt_wait != 0.0 || g_qt_acc != 0.0)    rc = fail("disarmed charged a phase");
    else if (ref[0] == 0.0f)                         rc = fail("the reference run published nothing");
    else if (G.issue_open)                           rc = fail("disarmed left the group open");
    if (rc) goto done;

    /* (A)(B)(C)(E)(F) armed, same inputs. The call is bracketed from outside so
     * the two windows can be checked against something that contains them. */
    g_qt_time_take = 1;
    arm_issue();
    double c0 = now_ms();
    int ok = qt_take(full, val, NDEV * NROW, seen);
    double call_ms = now_ms() - c0;
    if (!ok)                                        { rc = fail("armed take reported failure"); goto done; }
    if (memcmp(ref, seen, DIM * sizeof *ref) != 0)  { rc = fail("arming the split changed the output"); goto done; }
    if (clock_reads != 3)                           { rc = fail("armed did not read exactly three clocks"); goto done; }
    if (!(g_qt_wait >= (double)(NDEV * SLEEP_MS)))  { rc = fail("wait did not capture the drain"); goto done; }
    if (!(g_qt_wait + g_qt_acc <= call_ms + 1e-6))  { rc = fail("wait+acc exceeds the call"); goto done; }
    if (!(g_qt_acc >= 0.005))                       { rc = fail("acc is a clock-read gap, not the publish loop"); goto done; }

    /* (G) mask == 0: no device has experts in flight, so no clock may be read
     * and nothing charged. Bit equality, not a tolerance: the bug this guards
     * is a tiny per-layer constant, which any tolerance would swallow. */
    double w = g_qt_wait, a = g_qt_acc;
    long reads = clock_reads;
    G.issue_open = 1;
    for (int di = 0; di < NDEV; di++) G.is_cnt[di] = 0;
    if (!qt_take(0, val, NDEV * NROW, seen))        { rc = fail("mask==0 take reported failure"); goto done; }
    if (clock_reads != reads)                       { rc = fail("mask==0 read a clock"); goto done; }
    if (g_qt_wait != w || g_qt_acc != a)            { rc = fail("mask==0 charged a phantom wait"); goto done; }
    if (G.issue_open)                               { rc = fail("mask==0 left the group open"); goto done; }

    /* (H) an idle device must not end the drain: device 0 has nothing in
     * flight, device 1 does, and device 1 must still be drained. */
    memcpy(seen, ref, DIM * sizeof *ref);
    arm_issue();
    G.is_cnt[0] = 0;
    if (!qt_take(full & ~lo, val, NDEV * NROW, seen)) { rc = fail("idle-device take reported failure"); goto done; }
    if (calls[0] != 0)                             { rc = fail("drained a device with no rows"); goto done; }
    if (calls[1] != 1)                             { rc = fail("drain stopped at the idle device"); goto done; }

    /* (I) a failed device still waited: wait grows, nothing is published. */
    memcpy(seen, ref, DIM * sizeof *ref);
    for (failed = 0; failed < NDEV; failed++) {
        w = g_qt_wait;
        arm_issue();
        if (qt_take(full, val, NDEV * NROW, seen))      { rc = fail("failed device reported success"); goto done; }
        if (memcmp(ref, seen, DIM * sizeof *ref) != 0)  { rc = fail("failed device published a partial sum"); goto done; }
        if (!(g_qt_wait - w >= (double)(NDEV * SLEEP_MS))) { rc = fail("failed drain not charged to wait"); goto done; }
        for (int di = 0; di < NDEV; di++)
            if (G.is_cnt[di])                           { rc = fail("failed device left rows in flight"); goto done; }
        if (G.issue_open)                               { rc = fail("failed device left the group open"); goto done; }
    }

done:
    free(ref); free(seen);
    for (int di = 0; di < NDEV; di++) free(rows[di]);
    pthread_cond_destroy(&G.cv_take); pthread_mutex_destroy(&G.mx);
    if (rc) return rc;
    puts("tier take split: ok (3 clock reads, CLOCK_MONOTONIC only, wait+acc inside the call)");
    return 0;
}
