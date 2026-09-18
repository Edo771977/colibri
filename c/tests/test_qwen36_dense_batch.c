/* Model-free exactness gate for Qwen's dense-int8 prefill kernel.  The old
 * path calls matmul_q once per prompt row; the new path shares every int8
 * weight decode between two rows.  Compare raw float bytes across even/odd
 * row counts, vector tails, and both sides of the OpenMP threshold. */
#define COLI_QWEN_BATCH_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static float input_value(int64_t i, int salt) {
    int v = (int)((i * 37 + salt * 19) % 251);
    return (float)(v - 125) / (float)(97 + salt);
}

static void one_shape(int S, int I, int O) {
    float *x = falloc((int64_t)S*I);
    int8_t *q = malloc((size_t)O*I);
    float *sc = falloc(O);
    float *ref = falloc((int64_t)S*O), *got = falloc((int64_t)S*O);
    CHECK(q != NULL, "weight allocation failed");
    if (!q) exit(2);
    for (int64_t i=0;i<(int64_t)S*I;i++) x[i]=input_value(i,1);
    for (int64_t i=0;i<(int64_t)O*I;i++) q[i]=(int8_t)((i*29+7)%255-127);
    for (int o=0;o<O;o++) sc[o]=0.001f*(float)(1+(o*11)%31);

    for (int s=0;s<S;s++)
        matmul_q(ref+(int64_t)s*O,x+(int64_t)s*I,q,sc,I,O);
    matmul_q_batch(got,x,q,sc,S,I,O);
    if (memcmp(ref,got,(size_t)S*O*sizeof(float))) {
        int shown=0, different=0; float worst=0.f;
        for (int64_t i=0;i<(int64_t)S*O;i++) if (ref[i]!=got[i]) {
            float d=fabsf(ref[i]-got[i]); if(d>worst)worst=d;
            if(shown++<4)fprintf(stderr,"diff[%lld] ref=%a batch=%a delta=%g\n",
                                 (long long)i,ref[i],got[i],d);
            different++;
        }
        CHECK(0,"S=%d I=%d O=%d differs at %d values, worst=%g",
              S,I,O,different,worst);
    }
    printf("qwen dense batch exact: S=%d I=%d O=%d\n",S,I,O);
    free(x);free(q);free(sc);free(ref);free(got);
}

static void clear_qdw(void) {
    for(int i=0;i<g_qdw_n;i++){free(g_qdw[i].q);free(g_qdw[i].sc);}
    g_qdw_n=0;
}

static void shared_case(const char *format,int quantized) {
    enum { S=12,D=64,I=32 };
    Model m;memset(&m,0,sizeof(m));m.c.hidden=D;m.c.shared_inter=I;
    Layer l;memset(&l,0,sizeof(l));
    l.sh_g=falloc((int64_t)I*D);l.sh_u=falloc((int64_t)I*D);
    l.sh_d=falloc((int64_t)D*I);l.sh_gate=falloc(D);
    for(int64_t i=0;i<(int64_t)I*D;i++){l.sh_g[i]=input_value(i,2);l.sh_u[i]=input_value(i,3);}
    for(int64_t i=0;i<(int64_t)D*I;i++)l.sh_d[i]=input_value(i,4);
    for(int i=0;i<D;i++)l.sh_gate[i]=input_value(i,5);
    if(quantized){qdw_register(l.sh_g,D,I);qdw_register(l.sh_u,D,I);qdw_register(l.sh_d,I,D);}
    float *x=falloc((int64_t)S*D),*seed=falloc((int64_t)S*D);
    float *ref=falloc((int64_t)S*D),*got=falloc((int64_t)S*D);
    float *g=falloc(I),*u=falloc(I),*hh=falloc(D);
    for(int64_t i=0;i<(int64_t)S*D;i++){x[i]=input_value(i,6);seed[i]=input_value(i,7);}

    memcpy(ref,seed,(size_t)S*D*sizeof(float));setenv("QWEN_SHARED_BATCH","0",1);
    setenv("QWEN_DENSE_BATCH","0",1);g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,S,ref,g,u,hh);
    CHECK(g_qwen_matmul_d_calls==(uint64_t)S*3,"%s scalar calls=%llu expected=%d",format,
          (unsigned long long)g_qwen_matmul_d_calls,S*3);

    memcpy(got,seed,(size_t)S*D*sizeof(float));unsetenv("QWEN_SHARED_BATCH");
    unsetenv("QWEN_DENSE_BATCH");g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,S,got,g,u,hh);
    CHECK(!memcmp(ref,got,(size_t)S*D*sizeof(float)),"%s shared batch is not scalar bit-exact",format);
    CHECK(g_qwen_matmul_d_calls==3,"%s batch calls=%llu expected=3",format,
          (unsigned long long)g_qwen_matmul_d_calls);

    memcpy(got,seed,(size_t)D*sizeof(float));g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,1,got,g,u,hh);
    CHECK(!memcmp(ref,got,(size_t)D*sizeof(float)),"%s decode row changed",format);
    CHECK(g_qwen_matmul_d_calls==3,"%s decode calls=%llu expected=3",format,
          (unsigned long long)g_qwen_matmul_d_calls);
    printf("qwen shared batch exact: format=%s S=%d calls=%d -> 3\n",format,S,S*3);

    clear_qdw();free(l.sh_g);free(l.sh_u);free(l.sh_d);free(l.sh_gate);
    free(x);free(seed);free(ref);free(got);free(g);free(u);free(hh);
}

/* The decode shared expert has two shapes on the same binary: three matmul_d
 * calls, which is three OpenMP regions, and qwen_shared_fused_row, which is
 * one. Only the row-to-thread mapping differs, so they must agree to the BIT --
 * the token-exact oracle and KV-prefix reuse both read these floats. Shapes
 * below and above the if(O >= 256) team threshold, a scalar tail, and a shared
 * expert wider than the hidden size. */
static void shared_fused_case(int D, int Ish) {
    Model m; memset(&m,0,sizeof(m)); m.c.hidden=D; m.c.shared_inter=Ish;
    Layer l; memset(&l,0,sizeof(l));
    l.sh_g=falloc((int64_t)Ish*D); l.sh_u=falloc((int64_t)Ish*D);
    l.sh_d=falloc((int64_t)D*Ish); l.sh_gate=falloc(D);
    for(int64_t i=0;i<(int64_t)Ish*D;i++){l.sh_g[i]=input_value(i,2);l.sh_u[i]=input_value(i,3);}
    for(int64_t i=0;i<(int64_t)D*Ish;i++)l.sh_d[i]=input_value(i,4);
    for(int i=0;i<D;i++)l.sh_gate[i]=input_value(i,5);

    float *x=falloc(D),*seed=falloc(D),*ref=falloc(D),*got=falloc(D);
    float *sh=falloc(Ish),*shu=falloc(Ish),*shd=falloc(D);
    for(int i=0;i<D;i++){x[i]=input_value(i,6);seed[i]=input_value(i,7);}

    /* Unregistered weights are still f32: the fused path must decline rather
     * than read an int8 copy that does not exist. */
    memcpy(got,seed,(size_t)D*sizeof(float));
    CHECK(qwen_shared_fused_row(&l,x,got,D,Ish,sh)==0,
          "D=%d Ish=%d fused path ran on f32 weights",D,Ish);
    CHECK(!memcmp(got,seed,(size_t)D*sizeof(float)),
          "D=%d Ish=%d declined fused path still wrote to out",D,Ish);

    qdw_register(l.sh_g,D,Ish); qdw_register(l.sh_u,D,Ish); qdw_register(l.sh_d,Ish,D);
    CHECK(g_qdw_n==3,"D=%d Ish=%d expected three int8 copies, got %d",D,Ish,g_qdw_n);

    /* The three-call form, exactly as moe() spells it. */
    memcpy(ref,seed,(size_t)D*sizeof(float));
    matmul_d(sh,x,l.sh_g,1,D,Ish);
    matmul_d(shu,x,l.sh_u,1,D,Ish);
    for(int i=0;i<Ish;i++){float sv=sh[i];sh[i]=(sv/(1.f+expf(-sv)))*shu[i];}
    matmul_d(shd,sh,l.sh_d,1,Ish,D);
    float sgate=1.f;
    {   float sg=0.f; const float *wg=l.sh_gate;
        for(int i=0;i<D;i++) sg+=x[i]*wg[i];
        sgate=1.f/(1.f+expf(-sg)); }
    for(int d=0;d<D;d++) ref[d]+=sgate*shd[d];

    memcpy(got,seed,(size_t)D*sizeof(float));
    CHECK(qwen_shared_fused_row(&l,x,got,D,Ish,sh)==1,
          "D=%d Ish=%d fused path declined on int8 weights",D,Ish);
    if(memcmp(ref,got,(size_t)D*sizeof(float))){
        int shown=0,different=0;float worst=0.f;
        for(int d=0;d<D;d++) if(ref[d]!=got[d]){
            float e=fabsf(ref[d]-got[d]); if(e>worst)worst=e;
            if(shown++<4)fprintf(stderr,"diff[%d] three-call=%a fused=%a delta=%g\n",
                                 d,ref[d],got[d],e);
            different++;
        }
        CHECK(0,"D=%d Ish=%d fused shared expert differs at %d values, worst=%g",
              D,Ish,different,worst);
    }
    printf("qwen shared fused exact: D=%d Ish=%d (3 regions -> 1)\n",D,Ish);

    clear_qdw();
    free(l.sh_g);free(l.sh_u);free(l.sh_d);free(l.sh_gate);
    free(x);free(seed);free(ref);free(got);free(sh);free(shu);free(shd);
}

int main(void) {
    /* This gate owns the dense-int8 mode regardless of the caller's shell. */
    unsetenv("COLI_DENSE_I8");
    one_shape(1, 17, 13);       /* decode-shaped fallback */
    one_shape(2, 32, 31);       /* one vector block, one row pair */
    one_shape(4, 64, 73);       /* even prompt, serial OpenMP clause */
    one_shape(5, 67, 259);      /* odd prompt + scalar tail + parallel clause */
    shared_case("f32",0);
    shared_case("int8",1);
    shared_fused_case(64,32);        /* both loops below the team threshold */
    shared_fused_case(2048,512);     /* the shipping Qwen3.6 shared expert */
    shared_fused_case(288,320);      /* scalar tail, shared wider than hidden */
    unsetenv("QWEN_SHARED_BATCH");unsetenv("QWEN_DENSE_BATCH");
    if(failures){fprintf(stderr,"qwen dense batch: %d failure(s)\n",failures);return 1;}
    puts("qwen dense batch: ok");return 0;
}
