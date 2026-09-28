/* COLI_DN_GPU=1 in the engine, on the fake CUDA backend
 * (docs/qwen36-deltanet-gpu-plan.md, stage 3).
 *
 * The fake coli_cuda_deltanet_decode runs the engine's own CPU deltanet()
 * on the handle's copy of the state (the fake "device"), so a GPU-path
 * decode is the CPU path moved to other memory: with the state carried
 * correctly, every output and the final state must equal the CPU arm's byte
 * for byte. What that proves is the plumbing the real device cannot be asked
 * about in CI:
 *   - which calls take the path (S == 1 on an eligible layer) and which do
 *     not (S > 1, a layer whose three components are not on one device);
 *   - the ownership hand-offs: the first decode uploads the host state, a
 *     prompt (S > 1) brings it home first, the pin snapshot brings it home,
 *     reset_recurrent makes the device copy stale, and the next decode
 *     uploads again -- including a prompt without a reset in between, which
 *     is serve's prefix reuse;
 *   - a failed decode zeroes the state and stops (here, the fatal hook
 *     instead of exit), a failed upload sends the layer back to the CPU;
 *   - "dnstate" offered with COLI_DN_GPU=0 as with =1, nothing offered
 *     unset; the handles freed at qt_shutdown.
 * The arithmetic of the real path has its own oracle in the CUDA build
 * (tests/test_qwen36_deltanet_decode_cuda.cu).
 *
 * Include order as in test_qwen36_trunk_place.c: engine, fake backend, tier. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

/* Three layers: DeltaNet, attention, DeltaNet. Small dims, all distinct. */
enum { NL = 3, NE = 4, IH = 8, TOPK = 1, H = 16, VH = 2, VK = 1, KD = 4, VD = 4, CK = 4 };
enum { CONV = 2 * VK * KD + VH * VD, VAL = VH * VD };
static const uint8_t KIND[NL] = { 0, 1, 0 };

static unsigned long long rng = 88172645463325252ull;
static float frand(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return lo + (hi - lo) * (float)((rng >> 11) * (1.0 / 9007199254740992.0));
}
static float *rnd(size_t n, float lo, float hi) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = frand(lo, hi);
    return p;
}

static void build_model(Model *m) {
    memset(m, 0, sizeof *m);
    rng = 88172645463325252ull;            /* both arms: the same weights */
    Cfg *c = &m->c;
    c->n_layers = NL; c->n_experts = NE; c->topk = TOPK; c->hidden = H; c->inter = IH;
    c->eps = 1e-6f; c->vocab = 4;
    c->dn_vheads = VH; c->dn_kheads = VK; c->dn_kdim = KD; c->dn_vdim = VD; c->dn_convk = CK; c->dn_conv_dim = CONV;
    c->is_attn = malloc(NL); memcpy(c->is_attn, KIND, NL);
    m->L = calloc(NL, sizeof(Layer));
    m->DN_rec = calloc(NL, sizeof(float *)); m->DN_conv = calloc(NL, sizeof(float *));
    kv_prefix_alloc(&m->kvp, 64);          /* pin_restore asks it whether the snapshot still holds */
    for (int i = 0; i < NL; i++) {
        if (KIND[i]) continue;
        Layer *l = &m->L[i];
        l->dn_qkv = rnd((size_t)CONV * H, -0.5f, 0.5f); qdw_register(l->dn_qkv, H, CONV);
        l->dn_z = rnd((size_t)VAL * H, -0.5f, 0.5f);    qdw_register(l->dn_z, H, VAL);
        l->dn_out = rnd((size_t)H * VAL, -0.5f, 0.5f);  qdw_register(l->dn_out, VAL, H);
        l->dn_b = rnd((size_t)2 * VH * H, -0.3f, 0.3f);
        l->dn_conv = rnd((size_t)CONV * CK, -0.6f, 0.6f);
        l->dn_alog = rnd(VH, 0.f, 2.f);
        l->dn_dtbias = rnd(VH, -4.f, -1.f);
        l->dn_norm = rnd(VD, 0.5f, 1.5f);
        m->DN_rec[i] = calloc((size_t)VH * KD * VD, sizeof(float));
        m->DN_conv[i] = calloc((size_t)CONV * (CK - 1), sizeof(float));
    }
}
static void free_model(Model *m) {
    for (int i = 0; i < NL; i++) {
        if (KIND[i]) continue;
        Layer *l = &m->L[i];
        free(l->dn_qkv); free(l->dn_z); free(l->dn_out); free(l->dn_b); free(l->dn_conv);
        free(l->dn_alog); free(l->dn_dtbias); free(l->dn_norm);
        free(m->DN_rec[i]); free(m->DN_conv[i]);
    }
    free(m->DN_rec); free(m->DN_conv); free(m->L); free(m->c.is_attn);
    free(m->dn_gpu_on); free(m->dn_on_dev);
    free(m->dn_scratch.p);
    kv_prefix_free(&m->kvp);
    pin_drop();
    for (int j = 0; j < g_qdw_n; j++) { free(g_qdw[j].q); free(g_qdw[j].sc); }
    g_qdw_n = 0;
}

/* The fake device: the engine's CPU deltanet() on the handle's buffers. */
static Model *g_m;
static int dn_hook(ColiCudaDeltaNet *h, const float *x, float *out) {
    int layer = -1;
    for (int i = 0; i < NL; i++) if (!KIND[i] && g_m->L[i].dn_b == h->wab) layer = i;
    if (layer < 0) return 0;
    float *r = g_m->DN_rec[layer], *c = g_m->DN_conv[layer];
    g_m->DN_rec[layer] = h->rec; g_m->DN_conv[layer] = h->ring;
    g_dn_force_cpu = 1;
    deltanet(g_m, &g_m->L[layer], layer, (float *)x, 1, 0, out);
    g_dn_force_cpu = 0;
    g_m->DN_rec[layer] = r; g_m->DN_conv[layer] = c;
    return 1;
}
static int fatal_layer = -1;
static void fatal_hook(int layer, const char *what) { (void)what; fatal_layer = layer; }

/* Bring the tier up as main() does: offers (dnproj the way main() fuses
 * it, dnout, dnstate), qt_init, the dnproj upload, dnout placement, setup. */
static void arm_up(Model *m, const char *dn_gpu, const char *place) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", place, 1); setenv("CUDA_EXPERT_GB", "1", 1);
    if (dn_gpu) setenv("COLI_DN_GPU", dn_gpu, 1); else unsetenv("COLI_DN_GPU");
    fake_ndev = 1; fake_dense_compute = 1;
    G_offer_n = 0; G_place_n = 0; G_place_done = 0; G_auto_on = 0;
    build_model(m);
    g_m = m;
    for (int i = 0; i < NL; i++)
        if (!KIND[i]) qt_trunk_offer("dnproj", i, (size_t)(CONV + VAL) * H + (size_t)(CONV + VAL) * sizeof(float));
    trunk_offer_out(m);
    dn_offer_state(m);
    ck(qt_init(NL, NE, H, IH, NE, TOPK, 0, 1), "qt_init");
    for (int i = 0; i < NL; i++) {
        if (KIND[i]) continue;
        int dev = qt_place_of("dnproj", i);
        if (dev == QT_PLACE_CPU) continue;
        const int8_t *q1 = NULL, *q2 = NULL; const float *s1 = NULL, *s2 = NULL;
        for (int j = 0; j < g_qdw_n; j++) {
            if (g_qdw[j].w == m->L[i].dn_qkv) { q1 = g_qdw[j].q; s1 = g_qdw[j].sc; }
            if (g_qdw[j].w == m->L[i].dn_z)   { q2 = g_qdw[j].q; s2 = g_qdw[j].sc; }
        }
        int8_t *qf = malloc((size_t)(CONV + VAL) * H); float *sf = malloc((size_t)(CONV + VAL) * sizeof(float));
        memcpy(qf, q1, (size_t)CONV * H); memcpy(qf + (size_t)CONV * H, q2, (size_t)VAL * H);
        memcpy(sf, s1, CONV * sizeof(float)); memcpy(sf + CONV, s2, VAL * sizeof(float));
        qt_dnproj_init(i, qf, sf, H, CONV + VAL, dev);
        free(qf); free(sf);
    }
    trunk_place_out(m);
    dn_gpu_setup(m);
}
static void arm_down(Model *m) { qt_shutdown(); free_model(m); unsetenv("COLI_DN_GPU"); }

/* The sequence both arms run, every output appended to log (NULL: not kept).
 * Returns the number of floats written. */
enum { LOGMAX = 4096 };
static int pin_ok;
static int run_sequence(Model *m, float *log, float *snap) {
    int n = 0;
    float x[4 * H], out[4 * H];
    unsigned long long save = rng;
    rng = 0x2545F4914F6CDD1Dull;           /* both arms: the same inputs */
#define STEP(S) do { for (int i = 0; i < (S) * H; i++) x[i] = frand(-1.5f, 1.5f); \
        for (int l = 0; l < NL; l++) { if (KIND[l]) continue; \
            deltanet(m, &m->L[l], l, x, (S), 0, out); \
            if (log && n + (S) * H <= LOGMAX) { memcpy(log + n, out, (size_t)(S) * H * sizeof(float)); n += (S) * H; } } } while (0)
    reset_recurrent(m);
    STEP(3);                                /* a prompt: CPU, host state */
    for (int t = 0; t < 4; t++) STEP(1);    /* decode: uploads, then on the device */
    Q36PinState *st = q36_pin_state_save(m, NULL);   /* brings the state home */
    if (snap && st) {
        int k = 0;
        for (int l = 0; l < NL; l++) {
            if (KIND[l]) continue;
            memcpy(snap + k, st->rec[l], (size_t)VH * KD * VD * sizeof(float)); k += VH * KD * VD;
            memcpy(snap + k, st->conv[l], (size_t)CONV * (CK - 1) * sizeof(float)); k += CONV * (CK - 1);
        }
    }
    q36_pin_state_free(st);
    /* serve's pin: save at 7 tokens, two more decodes on the device, then a
     * request that extends the pinned 7 restores the host snapshot -- the
     * device copy is two tokens ahead and must not be used */
    static const int ids[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    static const float logit[4] = { 0 };
    kv_prefix_clear(&m->kvp); kv_prefix_record(&m->kvp, ids, 0, 7);
    pin_save(m, ids, 7, logit);
    for (int t = 0; t < 2; t++) STEP(1);
    pin_ok = pin_restore(m, ids, 9) == 7;
    for (int t = 0; t < 2; t++) STEP(1);
    STEP(2);                                /* a prompt without a reset: serve's prefix reuse */
    for (int t = 0; t < 3; t++) STEP(1);
    reset_recurrent(m);
    for (int t = 0; t < 2; t++) STEP(1);
    dn_all_to_host(m);
#undef STEP
    rng = save;
    return n;
}
static void final_state(Model *m, float *dst) {
    int k = 0;
    for (int l = 0; l < NL; l++) {
        if (KIND[l]) continue;
        memcpy(dst + k, m->DN_rec[l], (size_t)VH * KD * VD * sizeof(float)); k += VH * KD * VD;
        memcpy(dst + k, m->DN_conv[l], (size_t)CONV * (CK - 1) * sizeof(float)); k += CONV * (CK - 1);
    }
}

int main(void) {
    enum { NSTATE = 2 * (VH * KD * VD + CONV * (CK - 1)) };
    static float log_cpu[LOGMAX], log_gpu[LOGMAX], snap_cpu[NSTATE], snap_gpu[NSTATE], fin_cpu[NSTATE], fin_gpu[NSTATE];
    Model m;
    fake_dn_decode_hook = dn_hook;
    g_dn_fatal_hook = fatal_hook;

    printf("CPU arm: COLI_DN_GPU unset\n");
    arm_up(&m, NULL, "dnproj=0,dnout=0,dnstate=0");
    int no_offer = 1;
    for (int o = 0; o < G_offer_n; o++) if (!strcmp(G_offer[o].name, "dnstate")) no_offer = 0;
    ck(no_offer, "unset: no dnstate offer");
    ck(m.dn_gpu_on == NULL, "unset: no layer on the GPU path");
    int n_cpu = run_sequence(&m, log_cpu, snap_cpu);
    final_state(&m, fin_cpu);
    arm_down(&m);

    printf("OFF arm: COLI_DN_GPU=0\n");
    arm_up(&m, "0", "dnproj=0,dnout=0,dnstate=0");
    int offers = 0;
    for (int o = 0; o < G_offer_n; o++)
        if (!strcmp(G_offer[o].name, "dnstate")) {
            offers++;
            ck(G_offer[o].bytes == qt_dn_state_bytes(H, VH, VK, KD, VD, CK), "=0: dnstate offer charges qt_dn_state_bytes");
        }
    ck(offers == 2, "=0: one dnstate offer per DeltaNet layer");
    ck(m.dn_gpu_on == NULL, "=0: reserved, but no layer on the GPU path");
    arm_down(&m);

    printf("ON arm: COLI_DN_GPU=1\n");
    fake_dn_creates = fake_dn_decodes = fake_dn_uploads = fake_dn_downloads = fake_dn_frees = 0;
    arm_up(&m, "1", "dnproj=0,dnout=0,dnstate=0");
    ck(m.dn_gpu_on && m.dn_gpu_on[0] && m.dn_gpu_on[2] && !m.dn_gpu_on[1], "=1: both DeltaNet layers on the path, not the attention one");
    ck(fake_dn_creates == 2, "=1: two handles");
    int n_gpu = run_sequence(&m, log_gpu, snap_gpu);
    final_state(&m, fin_gpu);
    ck(n_gpu == n_cpu && !memcmp(log_gpu, log_cpu, (size_t)n_cpu * sizeof(float)), "every output byte for byte against the CPU arm");
    ck(!memcmp(snap_gpu, snap_cpu, sizeof snap_cpu), "the pin snapshot byte for byte (the state came home first)");
    ck(!memcmp(fin_gpu, fin_cpu, sizeof fin_cpu), "the final state byte for byte");
    ck(pin_ok, "pin_restore took the 7-token snapshot");
    /* 13 decode tokens x 2 layers go through the handle; the three prompt
     * rows and the two prompt rows never do */
    ck(fake_dn_decodes == 26, "26 decode calls through the handles, none for the prompts");
    /* uploads: after the first prompt, after the snapshot, after the
     * restore, after the second prompt, after the reset -- per layer. The
     * restore downloads nothing: it overwrites the host copy. */
    ck(fake_dn_uploads == 10, "10 state uploads (5 hand-offs x 2 layers)");
    ck(fake_dn_downloads == 6, "6 downloads: snapshot, second prompt, final (x 2 layers)");

    printf("failures\n");
    float x[H], out[H];
    for (int i = 0; i < H; i++) x[i] = 0.1f * (float)(i % 5);
    deltanet(&m, &m.L[0], 0, x, 1, 0, out);          /* layer 0 now on the device */
    fake_dn_fail_decode = 1; fatal_layer = -1;
    deltanet(&m, &m.L[0], 0, x, 1, 0, out);
    fake_dn_fail_decode = 0;
    int zero = 1;
    for (int i = 0; i < VH * KD * VD; i++) zero &= m.DN_rec[0][i] == 0.f;
    ck(fatal_layer == 0, "a failed decode reaches the fatal path (exit 3 in the engine)");
    ck(zero && !m.dn_on_dev[0] && !m.dn_on_dev[2], "... after zeroing the state and forgetting the device copies");
    fake_dn_fail_upload = 1; fatal_layer = -1;
    deltanet(&m, &m.L[2], 2, x, 1, 0, out);          /* host owns after the reset: this uploads */
    fake_dn_fail_upload = 0;
    ck(fatal_layer == -1 && !m.dn_gpu_on[2], "a failed upload sends that layer back to the CPU, no fatal");
    arm_down(&m);
    ck(fake_dn_frees == 2, "qt_shutdown freed both handles");

    printf("ON arm, dnstate not placed\n");
    arm_up(&m, "1", "dnproj=0,dnout=0");
    ck(m.dn_gpu_on == NULL, "without dnstate on the device, no layer takes the path");
    arm_down(&m);

    printf("ON arm, a DLL without the entry points\n");
    fake_dn_absent = 1;
    arm_up(&m, "1", "dnproj=0,dnout=0,dnstate=0");
    ck(m.dn_gpu_on == NULL, "coli_cuda_has_deltanet() = 0: the two-call path stays");
    arm_down(&m);
    fake_dn_absent = 0;

    printf("ON arm under SERVE=1\n");
    setenv("SERVE", "1", 1);
    arm_up(&m, "1", "dnproj=0,dnout=0,dnstate=0");
    int served_offers = 0;
    for (int o = 0; o < G_offer_n; o++) if (!strcmp(G_offer[o].name, "dnstate")) served_offers++;
    ck(served_offers == 0 && m.dn_gpu_on == NULL, "SERVE=1: no dnstate reservation, no layer on the path (stage 3b)");
    arm_down(&m);
    unsetenv("SERVE");

    printf("handle bytes\n");
    arm_up(&m, "1", "dnproj=0,dnout=0,dnstate=0");
    ck(coli_cuda_deltanet_bytes(G_dn[0]) == qt_dn_state_bytes(H, VH, VK, KD, VD, CK),
       "the fake handle reports qt_dn_state_bytes (qt_dn_init compares them on the card)");
    arm_down(&m);

    printf("dn gpu: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
