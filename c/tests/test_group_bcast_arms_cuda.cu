/* The broadcast contract, on EVERY arm of the expert-group dispatcher.
 *
 * WHY THIS FILE EXISTS, precisely. GroupDesc gained `xoff` so one activation
 * row could be shared by every chunk instead of copied K times: `offset` keeps
 * addressing gate/up/y, which are per-expert, and `xoff` addresses x, which
 * may be a single row. Six kernels were converted. TWO READERS WERE MISSED --
 * grouped_hidden_f8w_dual and the host-side generic arm -- and both shipped.
 *
 * The miss was invisible for three reasons worth writing down, because each
 * one is a lesson about how to test this:
 *
 *   1. The engine measured all day was qwen36 with int4 experts, which takes
 *      the g4 arm. The g4 arm was converted. The arms that were not are the
 *      ones that machine never runs.
 *   2. grouped_hidden_f8w_dual is the CUDA DEFAULT (COLI_F8_DEFAULT=1). Its
 *      mode-0 sibling, which nobody runs by default, IS converted. The fix
 *      landed on the arm that is not the default.
 *   3. tests/test_grouped_g4_cuda.cu asserts the broadcast contract -- for
 *      the g4 arm only. A per-arm contract needs a per-arm test.
 *
 * So this walks the arms, and a future arm that forgets xoff fails here.
 *
 * THE ORDER OF THE THREE RUNS IS THE TEST. A naive version -- run the
 * duplicated form and the broadcast form with the same vector and compare --
 * passes on broken code. ctx->x only ever GROWS, so after a duplicated run it
 * still holds `total` rows; a kernel reading x+offset*D then finds the
 * previous call's rows sitting there, and if those rows were the same vector
 * the wrong read returns the right answer.
 *
 * Hence: duplicate V1 first (this grows the buffer and leaves V1 in rows
 * 1..total-1), then broadcast V2 (only row 0 is uploaded; a broken kernel
 * reads V1 from the stale rows), then duplicate V2 as the reference. A
 * correct kernel gives the same bits for the last two; a broken one does not.
 *
 * NOT COVERED: the e8/IQ3 arm (fmt 6). Building an E8 tensor by hand is a
 * different job from this one, and grouped_hidden_e8_dual does read xoff
 * today -- checked by inspection, which is weaker than this file and is said
 * so rather than implied.
 *
 * Build: part of `make cuda-test`.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#include "../backend_cuda.h"

#ifdef _WIN32
static int setenv(const char *name,const char *value,int overwrite){
    (void)overwrite; return _putenv_s(name,value);
}
static int unsetenv(const char *name){ return _putenv_s(name,""); }
#endif

enum { NEXP = 3, D = 256, I = 128, GS = 64 };

static int fails;
static void check(int ok,const char *what){ if(!ok){ printf("  FAIL: %s\n",what); fails++; } }

static uint32_t rs = 0xC0FFEEu;
static uint32_t rnd(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static float rndf(void){ return (float)((int)(rnd()%2001)-1000)/1000.0f; }
/* e4m3 with the two NaN encodings avoided, like tests/test_fp8_warp_cuda.cu. */
static uint8_t rnd_e4m3(void){ uint8_t b; do { b=(uint8_t)(rnd()&0xFF); } while(b==0x7F||b==0xFF); return b; }

struct Group { ColiCudaTensor *g[NEXP], *u[NEXP], *d[NEXP]; };

static void group_free(Group *G){
    /* A partially built group is the normal case when an upload fails, so
     * every slot is checked rather than assumed live. */
    for(int c=0;c<NEXP;c++){
        if(G->g[c]) coli_cuda_tensor_free(G->g[c]);
        if(G->u[c]) coli_cuda_tensor_free(G->u[c]);
        if(G->d[c]) coli_cuda_tensor_free(G->d[c]);
        G->g[c]=G->u[c]=G->d[c]=NULL;
    }
}

/* fmt 1 (int8 per-row) -> the GENERIC host arm, one of the two that was
 * broken. fmt 2 (packed int4 per-row) -> the s4/w4 arm. The backend converts
 * offset-binary nibbles to signed on upload, so raw bytes are fine here. */
static int build_simple(Group *G,int fmt){
    size_t rbD = fmt==1 ? (size_t)D : (size_t)((D+1)/2);
    size_t rbI = fmt==1 ? (size_t)I : (size_t)((I+1)/2);
    for(int c=0;c<NEXP;c++){
        uint8_t *wg=(uint8_t*)malloc(rbD*I), *wu=(uint8_t*)malloc(rbD*I), *wd=(uint8_t*)malloc(rbI*D);
        float *sg=(float*)malloc((size_t)I*4), *su=(float*)malloc((size_t)I*4), *sd=(float*)malloc((size_t)D*4);
        if(!wg||!wu||!wd||!sg||!su||!sd) return 0;
        for(size_t i=0;i<rbD*I;i++){ wg[i]=(uint8_t)(rnd()&0xFF); wu[i]=(uint8_t)(rnd()&0xFF); }
        for(size_t i=0;i<rbI*D;i++) wd[i]=(uint8_t)(rnd()&0xFF);
        for(int o=0;o<I;o++){ sg[o]=0.001f+0.0001f*(o&7); su[o]=0.001f+0.0001f*(o&5); }
        for(int o=0;o<D;o++) sd[o]=0.001f+0.0001f*(o&3);
        int ok = coli_cuda_tensor_upload(&G->g[c],wg,sg,fmt,D,I,0)
              && coli_cuda_tensor_upload(&G->u[c],wu,su,fmt,D,I,0)
              && coli_cuda_tensor_upload(&G->d[c],wd,sd,fmt,I,D,0);
        free(wg);free(wu);free(wd);free(sg);free(su);free(sd);
        if(!ok) return 0;
    }
    return 1;
}

/* fmt 4 with gs>0 -> the g4 arm. The control: this one was converted, and it
 * must keep passing or the fix broke what worked. */
static int build_g4(Group *G){
    int ngD=(D+GS-1)/GS, ngI=(I+GS-1)/GS;
    for(int c=0;c<NEXP;c++){
        uint8_t *wg=(uint8_t*)malloc((size_t)((D+1)/2)*I), *wu=(uint8_t*)malloc((size_t)((D+1)/2)*I);
        uint8_t *wd=(uint8_t*)malloc((size_t)((I+1)/2)*D);
        float *sg=(float*)malloc((size_t)I*ngD*4), *su=(float*)malloc((size_t)I*ngD*4), *sd=(float*)malloc((size_t)D*ngI*4);
        if(!wg||!wu||!wd||!sg||!su||!sd) return 0;
        for(size_t i=0;i<(size_t)((D+1)/2)*I;i++){ wg[i]=(uint8_t)(rnd()&0xFF); wu[i]=(uint8_t)(rnd()&0xFF); }
        for(size_t i=0;i<(size_t)((I+1)/2)*D;i++) wd[i]=(uint8_t)(rnd()&0xFF);
        for(int i=0;i<I*ngD;i++){ sg[i]=0.002f+0.0001f*(i&7); su[i]=0.002f+0.0001f*(i&5); }
        for(int i=0;i<D*ngI;i++) sd[i]=0.002f+0.0001f*(i&3);
        int ok = coli_cuda_tensor_upload_g(&G->g[c],wg,sg,4,D,I,0,GS)
              && coli_cuda_tensor_upload_g(&G->u[c],wu,su,4,D,I,0,GS)
              && coli_cuda_tensor_upload_g(&G->d[c],wd,sd,4,I,D,0,GS);
        free(wg);free(wu);free(wd);free(sg);free(su);free(sd);
        if(!ok) return 0;
    }
    return 1;
}

/* fmt 8 -> the fp8 arm, whose WARP variant is the CUDA default and was the
 * other broken reader. Block scales laid out as tests/test_fp8_warp_cuda.cu
 * builds them. */
static int build_f8(Group *G){
    int nbD=(D+127)/128, nbI=(I+127)/128;
    for(int c=0;c<NEXP;c++){
        uint8_t *wg=(uint8_t*)malloc((size_t)I*D), *wu=(uint8_t*)malloc((size_t)I*D), *wd=(uint8_t*)malloc((size_t)D*I);
        float *sg=(float*)malloc((size_t)nbI*nbD*4), *su=(float*)malloc((size_t)nbI*nbD*4), *sd=(float*)malloc((size_t)nbD*nbI*4);
        if(!wg||!wu||!wd||!sg||!su||!sd) return 0;
        for(size_t i=0;i<(size_t)I*D;i++){ wg[i]=rnd_e4m3(); wu[i]=rnd_e4m3(); }
        for(size_t i=0;i<(size_t)D*I;i++) wd[i]=rnd_e4m3();
        for(int i=0;i<nbI*nbD;i++){ sg[i]=ldexpf(1.f+0.01f*(i&15),-12); su[i]=ldexpf(1.f+0.02f*(i&7),-12); }
        for(int i=0;i<nbD*nbI;i++) sd[i]=ldexpf(1.f+0.03f*(i&3),-12);
        int ok = coli_cuda_tensor_upload(&G->g[c],wg,sg,8,D,I,0)
              && coli_cuda_tensor_upload(&G->u[c],wu,su,8,D,I,0)
              && coli_cuda_tensor_upload(&G->d[c],wd,sd,8,I,D,0);
        free(wg);free(wu);free(wd);free(sg);free(su);free(sd);
        if(!ok) return 0;
    }
    return 1;
}

/* One issue/take, copying the result out. rows[] is all-ones: the broadcast
 * contract only accepts x_rows==1 when max_rows==1, which is the decode shape
 * it exists for. */
static int run(Group *G,const float *x,int x_rows,float *out){
    int rows[NEXP]; for(int c=0;c<NEXP;c++) rows[c]=1;
    if(!coli_cuda_expert_group_issue_x(G->g,G->u,G->d,rows,NEXP,x,x_rows)) return 0;
    const float *y=coli_cuda_expert_group_take(0);
    if(!y) return 0;
    memcpy(out,y,(size_t)NEXP*D*sizeof(float));
    return 1;
}

static void arm(const char *name,Group *G,const float *v1,const float *v2,
                const float *dup1,const float *dup2){
    float *y_bcast=(float*)malloc((size_t)NEXP*D*4), *y_ref=(float*)malloc((size_t)NEXP*D*4);
    float *scratch=(float*)malloc((size_t)NEXP*D*4);
    char what[160];
    if(!y_bcast||!y_ref||!scratch){ printf("  FAIL alloc %s\n",name); fails++; return; }
    (void)v1;

    /* 1. duplicated V1 -- grows ctx->x to NEXP rows and leaves V1 in rows 1..2 */
    snprintf(what,sizeof what,"%s: duplicated V1 ran",name);
    check(run(G,dup1,NEXP,scratch),what);
    /* 2. broadcast V2 -- only row 0 is uploaded; a reader using `offset`
     *    picks up V1 from the stale rows above it */
    snprintf(what,sizeof what,"%s: broadcast V2 ran",name);
    check(run(G,v2,1,y_bcast),what);
    /* 3. duplicated V2 -- the reference the broadcast must equal */
    snprintf(what,sizeof what,"%s: duplicated V2 ran",name);
    check(run(G,dup2,NEXP,y_ref),what);

    int same=!memcmp(y_bcast,y_ref,(size_t)NEXP*D*sizeof(float));
    /* Locate the damage: expert 0 reads row 0 either way, so a broken arm is
     * correct there and wrong for 1..NEXP-1. Saying which makes the failure
     * diagnose itself. */
    int bad_first=-1;
    for(int c=0;c<NEXP && bad_first<0;c++)
        if(memcmp(y_bcast+(size_t)c*D,y_ref+(size_t)c*D,(size_t)D*sizeof(float))) bad_first=c;
    printf("  %-18s %s%s\n",name, same?"bitwise":"DIFFERS",
           same?"":(bad_first==0?"  (expert 0 too: not an xoff miss)":"  (expert 0 ok, 1.. wrong: xoff)"));
    snprintf(what,sizeof what,"%s: broadcast == duplicated, bitwise",name);
    check(same,what);
    free(y_bcast); free(y_ref); free(scratch);
}

int main(void){
    int devs[1]={0};
    if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }
    if(!coli_cuda_has_group_x_broadcast()){
        printf("broadcast export absent: nothing to test\n"); return 0; }

    float lut[256]; for(int i=0;i<256;i++){
        /* the backend's own e4m3 table is what matters for parity here; any
         * monotone decode does, since both runs use the same one */
        int s=(i>>7)&1, e=(i>>3)&15, m=i&7; float v;
        if(e==0) v=ldexpf((float)m/8.f,-6); else v=ldexpf(1.f+(float)m/8.f,e-7);
        lut[i]=s?-v:v;
    }
    if(!coli_cuda_fp8_set_lut(lut)){ printf("FAIL set_lut\n"); return 1; }

    float *v1=(float*)malloc((size_t)D*4), *v2=(float*)malloc((size_t)D*4);
    float *dup1=(float*)malloc((size_t)NEXP*D*4), *dup2=(float*)malloc((size_t)NEXP*D*4);
    if(!v1||!v2||!dup1||!dup2){ printf("FAIL alloc\n"); return 1; }
    for(int i=0;i<D;i++){ v1[i]=rndf(); v2[i]=rndf()*0.5f+0.25f; }   /* must DIFFER */
    for(int c=0;c<NEXP;c++){ memcpy(dup1+(size_t)c*D,v1,(size_t)D*4);
                             memcpy(dup2+(size_t)c*D,v2,(size_t)D*4); }

    printf("broadcast contract, per dispatcher arm (x_rows=1 must equal duplicated rows)\n");
    Group G; memset(&G,0,sizeof G);

    if(build_simple(&G,1)) arm("generic fmt=1",&G,v1,v2,dup1,dup2);
    else { printf("  FAIL build fmt=1\n"); fails++; }
    group_free(&G);

    if(build_simple(&G,2)) arm("s4/w4 fmt=2",&G,v1,v2,dup1,dup2);
    else { printf("  FAIL build fmt=2\n"); fails++; }
    group_free(&G);

    if(build_g4(&G)) arm("g4 fmt=4 gs=64",&G,v1,v2,dup1,dup2);
    else { printf("  FAIL build fmt=4\n"); fails++; }
    group_free(&G);

    /* Both fp8 modes, and the DEFAULT one first -- it is the one that was
     * broken, and a reader skimming the output should meet it before the
     * variant that was already correct. */
    if(build_f8(&G)){
        unsetenv("COLI_CUDA_F8_WARP");
        arm("fp8 warp (default)",&G,v1,v2,dup1,dup2);
        setenv("COLI_CUDA_F8_WARP","0",1);
        arm("fp8 mode 0",&G,v1,v2,dup1,dup2);
        unsetenv("COLI_CUDA_F8_WARP");
    } else { printf("  FAIL build fmt=8\n"); fails++; }
    group_free(&G);

    free(v1);free(v2);free(dup1);free(dup2);
    coli_cuda_shutdown();
    if(fails){ printf("test_group_bcast_arms_cuda: %d failure(s)\n",fails); return 1; }
    printf("test_group_bcast_arms_cuda: ok\n");
    return 0;
}
