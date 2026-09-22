/* The two OUTPUT projections on the placement table, driven through the
 * engine's own trunk_place_out() on the fake CUDA backend.
 *
 * qwen36_tier.h has advertised `dnout` in its COLI_PLACE example since the
 * table was written, and qt_place_of() has always answered for any name --
 * but the engine asked only about "lmhead" and "dnproj", so COLI_PLACE=dnout=0
 * parsed, stored a device, and moved nothing. A test that only exercised the
 * tier could not see that: the tier was right the whole time (that is what
 * test_qwen36_tier_dense pins). The gap was the engine never asking.
 *
 * So this drives the engine side: that attnout lands on attention layers and
 * dnout on DeltaNet ones and never the other way round, that a placed matrix
 * answers the same numbers as the CPU int8 path it replaced, that the
 * calloc'd zero in a fresh Layer reads as "on the CPU" rather than as handle
 * zero, and -- the one that protects a default run -- that with no COLI_PLACE
 * the automatic placer moves neither of them, because neither is offered to
 * it until someone has measured whether it pays.
 *
 * Include order as in test_qwen36_tier_int8_engine.c: engine first (it never
 * references coli_cuda_*), then the fake backend, then the tier source, so
 * the tier's statics live in this TU. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv: MinGW has none */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

/* Four layers, alternating kinds, so "placed on the right layers" cannot pass
 * by placing everything. o_in and value_dim differ from each other and from
 * hidden: a swapped dimension is then a wrong answer, not a lucky one. */
enum { NL = 4, NE = 4, D = 32, IH = 16, TOPK = 2, VH = 2, VD = 6, O_IN = 24 };
enum { VALUE_DIM = VH * VD };
/* attention input projections: q_out and kv_out differ, and differ from
 * each other, so a fused block assembled at the wrong offset is a wrong
 * answer rather than a lucky one. */
enum { QH = 2, QHD = 8, KVH = 1, KHD = 4 };
enum { Q_OUT = QH * QHD, KV_OUT = KVH * KHD, QKV_OUT = Q_OUT + 2 * KV_OUT };
/* widest input either projection takes, for the one scratch row below */
#define XMAX ((int)O_IN > (int)VALUE_DIM ? (int)O_IN : (int)VALUE_DIM)
static const uint8_t KIND[NL] = { 1, 0, 1, 0 };   /* 1 = attention, 0 = DeltaNet */

/* Small integers on both sides: every product and every partial sum is exact
 * in f32, so summation order cannot separate the CPU path from the fake
 * device's. That is what lets the comparison below be an equality. */
static float wval(int layer, int64_t i) { return (float)((int)((i + layer * 7) % 15) - 7); }
static float xval(int64_t i)            { return (float)((int)(i % 9) - 4); }

static void build_model(Model *m) {
    memset(m, 0, sizeof *m);
    m->c.n_layers = NL; m->c.n_experts = NE; m->c.topk = TOPK;
    m->c.hidden = D; m->c.inter = IH; m->c.expert_gs = 0;
    m->c.o_in = O_IN; m->c.dn_vheads = VH; m->c.dn_vdim = VD;
    m->c.q_heads = QH; m->c.q_head_dim = QHD;
    m->c.kv_heads = KVH; m->c.k_head_dim = KHD;
    m->c.is_attn = malloc(NL);
    memcpy(m->c.is_attn, KIND, NL);
    m->L = calloc(NL, sizeof(Layer));     /* the engine's own zeroing */
    for (int i = 0; i < NL; i++) {
        int I = KIND[i] ? O_IN : VALUE_DIM;
        float *W = malloc((size_t)I * D * sizeof(float));
        for (int64_t e = 0; e < (int64_t)I * D; e++) W[e] = wval(i, e);
        if (KIND[i]) m->L[i].o = W; else m->L[i].dn_out = W;
        qdw_register(W, I, D);            /* the int8 copy the CPU path uses */
        if (!KIND[i]) continue;
        /* q, k and v for the attention layers. Distinct generators so a fused
         * block that concatenated them in the wrong order, or carried the
         * wrong scale row, cannot still match. */
        int outs[3] = { Q_OUT, KV_OUT, KV_OUT };
        float **dst[3] = { &m->L[i].q, &m->L[i].k, &m->L[i].v };
        for (int t = 0; t < 3; t++) {
            float *M = malloc((size_t)D * outs[t] * sizeof(float));
            /* The amplitude differs per component, and that is not decoration.
             * wval has period 15 over rows of D = 32, so every row already
             * spans -7..7 and qdw_register's per-row max-abs would be exactly
             * 7 for q, k and v alike -- identical scales, and swapping the k
             * and v SCALE rows inside the fused block would change no number
             * and pass this test. A reviewer demonstrated exactly that. With
             * x7, x14, x21 the three scales differ and the swap is fatal.
             * Still small integers, so every product and partial sum stays
             * exact in f32 and the comparison below stays an equality. */
            for (int64_t e = 0; e < (int64_t)D * outs[t]; e++)
                M[e] = wval(i * 3 + t + 1, e) * (float)(t + 1);
            *dst[t] = M;
            qdw_register(M, D, outs[t]);
        }
    }
}

static void free_model(Model *m) {
    for (int i = 0; i < NL; i++) {
        free(KIND[i] ? m->L[i].o : m->L[i].dn_out);
        free(m->L[i].q); free(m->L[i].k); free(m->L[i].v);
    }
    free(m->L); free(m->c.is_attn);
    /* qdw_register malloc'd an int8 copy and a scale row per matrix; the
     * engine keeps them for the process lifetime, a test between two arms
     * must not. */
    for (int j = 0; j < g_qdw_n; j++) { free(g_qdw[j].q); free(g_qdw[j].sc); }
    g_qdw_n = 0;
}

static void tier_up(const char *place) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", place, 1);
    setenv("CUDA_EXPERT_GB", "1", 1);
    fake_ndev = 1; fake_dense_compute = 1;
    /* Three process-lifetime statics the engine sets once and never revisits:
     * the offer table, and COLI_PLACE's cached parse. qt_shutdown clears
     * neither, correctly -- one process, one placement. An arm here that
     * inherited them would read the PREVIOUS arm's COLI_PLACE: that is what
     * arm 4 did on the first run of this file, watching the automatic placer
     * place components that arm 1 had named and it had not. */
    G_offer_n = 0;
    G_place_n = 0; G_place_done = 0; G_auto_on = 0;
}

static size_t attn_bytes(void) { return (size_t)D * O_IN      + (size_t)D * sizeof(float); }
static size_t dn_bytes(void)   { return (size_t)D * VALUE_DIM + (size_t)D * sizeof(float); }
static size_t qkv_bytes(void)  { return (size_t)D * QKV_OUT   + (size_t)QKV_OUT * sizeof(float); }

/* One arm: bring the tier up under `place`, offer, init, place. */
static void run_arm(Model *m, const char *place, const char *what) {
    tier_up(place);
    build_model(m);
    trunk_offer_out(m);               /* as main() does, before qt_init */
    trunk_offer_attnproj(m);
    ck(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), what);
    trunk_place_out(m);
    trunk_place_attnproj(m);
}

int main(void) {
    Model m;

    /* The two names must be asked SEPARATELY, so give them different answers.
     * With both on device 0 the assertions below pass even when the names are
     * swapped inside trunk_place_out -- they would only be testing that a
     * handle lands in the field matching is_attn[], which is the one thing
     * that cannot go wrong. Asymmetric targets make a swap fatal. */
    printf(" 1. attnout=0, dnout=cpu -- only the attention layers move\n");
    run_arm(&m, "experts=0,attnout=0,dnout=cpu", "tier starts");
    ck(m.L[0].h_attnout > 0 && m.L[2].h_attnout > 0, "attnout placed on both attention layers");
    ck(m.L[1].h_dnout == 0 && m.L[3].h_dnout == 0, "dnout=cpu leaves the DeltaNet layers alone");
    ck(m.L[0].h_dnout == 0 && m.L[2].h_dnout == 0, "no dnout handle on an attention layer");
    ck(m.L[1].h_attnout == 0 && m.L[3].h_attnout == 0, "no attnout handle on a DeltaNet layer");
    ck(qt_dense_count() == 2, "two resident matrices, not four");
    ck(G_trunk_bytes[0] == 2 * attn_bytes(), "only the attnout bytes charged to the trunk");
    qt_shutdown(); free_model(&m);

    printf(" 2. attnout=cpu, dnout=0 -- and the mirror image\n");
    run_arm(&m, "experts=0,attnout=cpu,dnout=0", "tier starts");
    ck(m.L[1].h_dnout > 0 && m.L[3].h_dnout > 0, "dnout placed on both DeltaNet layers");
    ck(m.L[0].h_attnout == 0 && m.L[2].h_attnout == 0, "attnout=cpu leaves the attention layers alone");
    ck(qt_dense_count() == 2, "two resident matrices, not four");
    ck(G_trunk_bytes[0] == 2 * dn_bytes(), "only the dnout bytes charged to the trunk");
    qt_shutdown(); free_model(&m);

    printf(" 3. both placed: the bytes are charged, and the numbers survive\n");
    run_arm(&m, "experts=0,attnout=0,dnout=0", "tier starts");
    /* The expert budget is the allowance minus the trunk that landed on the
     * device, summed over the OFFERS. A matrix placed without one is uploaded
     * and charged to nobody, and the warmstart then fills experts into VRAM
     * that is already spoken for -- the R4 regression, under a new name. */
    ck(G_trunk_bytes[0] == 2 * attn_bytes() + 2 * dn_bytes(),
       "the placed bytes are charged to the device's trunk, not to nobody");
    {
        float x[XMAX], cpu[D], gpu[D];
        for (int i = 0; i < XMAX; i++) x[i] = xval(i);
        for (int i = 0; i < NL; i++) {
            int I = KIND[i] ? O_IN : VALUE_DIM;
            const float *W = KIND[i] ? m.L[i].o : m.L[i].dn_out;
            int h = KIND[i] ? m.L[i].h_attnout : m.L[i].h_dnout;
            memset(cpu, 0, sizeof cpu); memset(gpu, 0, sizeof gpu);
            matmul_d(cpu, x, W, 1, I, D);
            ck(trunk_out_matmul(h, gpu, x, I, D), "the device answers");
            ck(memcmp(cpu, gpu, sizeof cpu) == 0, "identical to the CPU int8 result");
        }
    }
    {
        float x[O_IN], y[D];
        for (int i = 0; i < O_IN; i++) x[i] = xval(i);
        for (int i = 0; i < D; i++) y[i] = 1234.f;
        ck(!trunk_out_matmul(0, y, x, O_IN, D), "handle zero does not reach the tier");
        ck(y[0] == 1234.f, "and writes nothing");
    }
    qt_shutdown(); free_model(&m);

    printf(" 4. the automatic placer sees them and, with room, takes them\n");
    /* This assertion used to read the other way: the first version of this
     * patch withheld the offers so that a default run could not change while
     * nobody had measured whether a placed matrix pays for the driver
     * round-trip it adds. It has been measured (-7.77 ms/token, four runs a
     * side, spread 2.1 -- docs/qwen36-cuda-tier.md), so the offers go in and
     * auto decides. `off` below is what withholding them now means. */
    run_arm(&m, "", "tier starts in auto mode");     /* "" == unset == auto */
    ck(m.L[0].h_attnout > 0 && m.L[2].h_attnout > 0, "auto takes attnout on the attention layers");
    ck(m.L[1].h_dnout   > 0 && m.L[3].h_dnout   > 0, "auto takes dnout on the DeltaNet layers");
    /* Since 22 September attnproj is offered too, so auto holds one output
     * projection per layer PLUS one fused input projection per attention
     * layer. Counting NL here would pass again the day someone withdraws the
     * attnproj offer, which is exactly the regression this should catch. */
    ck(m.L[0].h_attnproj > 0 && m.L[2].h_attnproj > 0, "auto takes attnproj on the attention layers");
    ck(qt_dense_count() == NL + 2, "one output projection per layer, plus attnproj on the two attention layers");
    ck(G_trunk_bytes[0] == 2 * attn_bytes() + 2 * dn_bytes() + 2 * qkv_bytes(),
       "auto charges the same bytes an explicit placement charges");
    qt_shutdown(); free_model(&m);

    printf(" 5. COLI_PLACE=off is the escape hatch, and still empties the trunk\n");
    /* With the offers unconditional, `off` is the only way back to experts
     * only -- so it has to keep working, or the documented escape hatch is a
     * word in a table. */
    run_arm(&m, "off", "tier starts with COLI_PLACE=off");
    for (int i = 0; i < NL; i++)
        ck(m.L[i].h_attnout == 0 && m.L[i].h_dnout == 0 && m.L[i].h_attnproj == 0,
           "off places nothing, attnproj included");
    ck(qt_dense_count() == 0, "no resident matrix taken");
    ck(G_trunk_bytes[0] == 0, "and no trunk bytes charged");
    qt_shutdown(); free_model(&m);

    printf(" 6. attnproj: q ++ k ++ v fused into one GEMV, same numbers\n");
    run_arm(&m, "experts=0,attnproj=0", "tier starts");
    ck(m.L[0].h_attnproj > 0 && m.L[2].h_attnproj > 0, "attnproj placed on both attention layers");
    ck(m.L[1].h_attnproj == 0 && m.L[3].h_attnproj == 0, "and on neither DeltaNet layer");
    ck(m.L[0].h_attnout == 0 && m.L[1].h_dnout == 0,
       "naming attnproj alone places nothing else");
    ck(G_trunk_bytes[0] == 2 * qkv_bytes(), "the fused bytes are charged to the trunk");
    {
        /* The fused GEMV must answer exactly what the three separate CPU
         * matmuls answer, in that order. This is the assertion the fusion
         * exists for: a wrong offset or a mis-assembled scale row survives
         * every other check in this file. */
        float x[D], want[QKV_OUT], got[QKV_OUT];
        for (int i = 0; i < D; i++) x[i] = xval(i);
        for (int i = 0; i < NL; i++) {
            if (!KIND[i]) continue;
            memset(want, 0, sizeof want); memset(got, 0, sizeof got);
            matmul_d(want,                  x, m.L[i].q, 1, D, Q_OUT);
            matmul_d(want + Q_OUT,          x, m.L[i].k, 1, D, KV_OUT);
            matmul_d(want + Q_OUT + KV_OUT, x, m.L[i].v, 1, D, KV_OUT);
            ck(trunk_out_matmul(m.L[i].h_attnproj, got, x, D, QKV_OUT), "the device answers");
            ck(memcmp(want, got, sizeof want) == 0, "fused == q, then k, then v");
        }
    }
    qt_shutdown(); free_model(&m);

    printf(" 7. attnproj is offered to auto, on attention layers only\n");
    /* Explicit-only until 22 September 2026, on the argument that kept
     * dnout/attnout out until 19 September: a default does not move on a
     * prediction. The A/B has now been run on the target hardware -- median
     * -4.80 ms/token, 6/6 negative, 33.13 -> 39.70 tok/s, counterbalanced
     * order, frozen heat table, cpu-miss 0.00 in both arms
     * (docs/experiments/qwen36-attnproj-place-2026-09-22-raw.txt) -- so the
     * offer goes in unconditionally and the placer weighs it like the others.
     *
     * What must NOT change is the shape of the offer: attnproj_bytes returns 0
     * for a DeltaNet layer, so those are still never offered and still never
     * placed. That half of the old assertion is the half worth keeping. */
    run_arm(&m, "", "tier starts in auto mode");
    for (int i = 0; i < NL; i++) {
        if (KIND[i])
            ck(m.L[i].h_attnproj != 0, "auto takes attnproj on an attention layer");
        else
            ck(m.L[i].h_attnproj == 0, "auto leaves DeltaNet layers alone");
    }
    qt_shutdown(); free_model(&m);

    printf(fails ? "FAILED\n" : "OK\n");
    return fails != 0;
}
