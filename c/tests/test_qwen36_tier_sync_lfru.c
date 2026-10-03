/* QT_UPLOAD_SYNC=1 also lands the LFRU swap the layer-0 tick queues, before
 * the group is formed.
 *
 * qt_issue used to drain the upload queue and only then run the tick, so a
 * swap the tick queued landed whenever the uploader got to it (and a swap
 * parks while a group is open). CACHE_ROUTE asks qt_is_resident before the
 * next group, so under QT_UPLOAD_SYNC its routing still depended on the
 * uploader's timing: two runs of the same command could differ. Now, under
 * QT_UPLOAD_SYNC, the tick runs before the drain.
 *
 * Fake CUDA backend, no GPU: one layer, two experts, budget for one. Expert
 * 0 resident and cold, expert 1 refused by the budget and hot; the tick
 * count is set so the next layer-0 issue runs the LFRU pass. Issuing expert 1
 * must find it resident (mask bit 0 set), expert 0 evicted, and nothing in
 * flight. With the old order the issue misses and the swap is still queued. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

static int issue_ok(int device, int count, const float *x) {
    (void)device; (void)count; (void)x; return 1;
}

int main(void) {
    enum { D = 64 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    /* budget for one expert and a half (as test_qwen36_tier_shutdown) */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * 32 / 2) + 3 * dev_alloc_footprint((2 * 32 + D) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1;

    if (!qt_init(1, 2, D, 32, 2, 1, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL: the tier does not start\n");
        return 1;
    }
    check(G_upload_sync == 1, "QT_UPLOAD_SYNC=1 is read at init");

    static unsigned char g4[2][D * 32 / 2], u4[2][D * 32 / 2], d4[2][D * 32 / 2];
    static float sc[2][2 * 32 + D];
    for (int eid = 0; eid < 2; eid++) {
        memset(g4[eid], (unsigned char)(eid + 1), sizeof g4[eid]);
        memset(u4[eid], (unsigned char)(eid + 2), sizeof u4[eid]);
        memset(d4[eid], (unsigned char)(eid + 3), sizeof d4[eid]);
        for (int i = 0; i < 2 * 32 + D; i++) sc[eid][i] = 1.0f;
    }
    qt_note(0, 0, g4[0], u4[0], d4[0], sc[0], sc[0] + 32, sc[0] + 2 * 32);
    qt_note(0, 1, g4[1], u4[1], d4[1], sc[1], sc[1] + 32, sc[1] + 2 * 32);
    qt_fill_wait();
    check(qt_is_resident(0, 0), "expert 0 fits the one-expert budget");
    check(!qt_is_resident(0, 1), "expert 1 is refused by the exhausted budget");

    /* expert 1 hot, expert 0 cold; the next tick is a multiple of 16 */
    pthread_mutex_lock(&G.mx);
    qs(0, 0)->heat = 0;
    qs(0, 1)->heat = 100;
    G.tick = 15;
    uint64_t swaps0 = G.swaps;
    pthread_mutex_unlock(&G.mx);

    fake_issue_hook = issue_ok;
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    int eid = 1;
    uint32_t mask = qt_issue(0, &eid, 1, x);
    pthread_mutex_lock(&G.mx);
    int inflight = G.inflight, r0 = qs(0, 0)->resident, r1 = qs(0, 1)->resident;
    uint64_t swaps = G.swaps - swaps0;
    pthread_mutex_unlock(&G.mx);
    check(swaps == 1, "the layer-0 tick queued one LFRU swap");
    check(r1 && !r0, "the swap landed before the group: expert 1 resident, expert 0 evicted");
    check(inflight == 0, "nothing in flight when the group is formed");
    check(mask == 1u, "the issue finds the swapped-in expert resident");
    float out[D]; memset(out, 0, sizeof out); float val[1] = {1};
    qt_take(mask, val, 1, out);
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_sync_lfru: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_sync_lfru: ok\n");
    return 0;
}
