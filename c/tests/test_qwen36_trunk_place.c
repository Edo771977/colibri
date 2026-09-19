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
    m->c.is_attn = malloc(NL);
    memcpy(m->c.is_attn, KIND, NL);
    m->L = calloc(NL, sizeof(Layer));     /* the engine's own zeroing */
    for (int i = 0; i < NL; i++) {
        int I = KIND[i] ? O_IN : VALUE_DIM;
        float *W = malloc((size_t)I * D * sizeof(float));
        for (int64_t e = 0; e < (int64_t)I * D; e++) W[e] = wval(i, e);
        if (KIND[i]) m->L[i].o = W; else m->L[i].dn_out = W;
        qdw_register(W, I, D);            /* the int8 copy the CPU path uses */
    }
}

static void free_model(Model *m) {
    for (int i = 0; i < NL; i++) free(KIND[i] ? m->L[i].o : m->L[i].dn_out);
    free(m->L); free(m->c.is_attn);
}

static void tier_up(const char *place) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", place, 1);
    setenv("CUDA_EXPERT_GB", "1", 1);
    fake_ndev = 1; fake_dense_compute = 1;
}

int main(void) {
    Model m;

    printf(" 1. explicit COLI_PLACE moves both output projections\n");
    /* experts=0, not experts=cpu: the engine only reaches trunk_place_out()
     * inside the `if (qt_init(...))` that brings the expert tier up, so a
     * test that placed the trunk with the tier off would be exercising a
     * path the engine never takes. */
    tier_up("experts=0,attnout=0,dnout=0");
    build_model(&m);
    ck(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts");
    trunk_place_out(&m);
    ck(m.L[0].h_attnout > 0 && m.L[2].h_attnout > 0, "attnout placed on both attention layers");
    ck(m.L[1].h_dnout   > 0 && m.L[3].h_dnout   > 0, "dnout placed on both DeltaNet layers");
    ck(m.L[0].h_dnout == 0 && m.L[2].h_dnout == 0, "no dnout handle on an attention layer");
    ck(m.L[1].h_attnout == 0 && m.L[3].h_attnout == 0, "no attnout handle on a DeltaNet layer");
    ck(qt_dense_count() == NL, "one resident matrix per layer, no more");

    printf(" 2. a placed matrix answers what the CPU int8 path answered\n");
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

    printf(" 3. handle zero is the CPU, not the first matrix\n");
    {
        float x[O_IN], y[D];
        for (int i = 0; i < O_IN; i++) x[i] = xval(i);
        for (int i = 0; i < D; i++) y[i] = 1234.f;
        ck(!trunk_out_matmul(0, y, x, O_IN, D), "a calloc'd Layer does not reach the tier");
        ck(y[0] == 1234.f, "and writes nothing");
    }
    qt_shutdown();
    free_model(&m);
    g_qdw_n = 0;                          /* the next arm registers its own */

    printf(" 4. without COLI_PLACE the automatic placer moves neither\n");
    tier_up("");                          /* "" == unset == auto */
    build_model(&m);
    ck(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts in auto mode");
    trunk_place_out(&m);
    for (int i = 0; i < NL; i++)
        ck(m.L[i].h_attnout == 0 && m.L[i].h_dnout == 0,
           "auto places no output projection (they are never offered to it)");
    ck(qt_dense_count() == 0, "no resident matrix taken");
    qt_shutdown();
    free_model(&m);

    printf(fails ? "FAILED\n" : "OK\n");
    return fails != 0;
}
