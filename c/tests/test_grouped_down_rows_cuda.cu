/* grouped_down_g4r<R>: R output rows per block for the expert down projection
 * (COLI_CUDA_DOWN_ROWS).
 *
 * The kernel changes ONE thing about grouped_down_g4: a block carries R output
 * rows instead of one, so the activation row is read once per R rows and each
 * thread issues R independent packed-byte loads instead of one. The per-thread
 * partition of the input (b = tid, tid+256, ...), the multiplication order
 * (x * nibble * scale), the 256-wide reduction tree and the epilogue are all
 * untouched -- so every output must be BITWISE identical to the original's,
 * and the contract here is memcmp, not a tolerance.
 *
 * The identity also rests on nvcc contracting x*nibble*scale into the same
 * FMA in both kernels. The expressions are character-for-character the same,
 * so it should, but "should" is why this test exists: if a future toolchain
 * contracts them differently the memcmp below fails loudly instead of the
 * logits drifting quietly.
 *
 * A tolerance test would be the wrong gate twice over: it would pass a kernel
 * that quietly reassociated the sum, and it would pass a dispatch that never
 * fired (the same kernel run twice is trivially identical). Hence both the
 * memcmp and the launch counter.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_grouped_down_rows_cuda.cu \
 *            -o tests/test_grouped_down_rows
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

#ifdef _WIN32
static int setenv(const char *name,const char *value,int overwrite){
    (void)overwrite; char buf[128];
    snprintf(buf,sizeof buf,"%s=%s",name,value);
    return _putenv(buf);
}
#endif

static int fails;
static void check(int ok,const char *what){
    if(!ok){ printf("FAIL %s\n",what); fails++; }
}

static unsigned long long rs=88172645463325252ULL;
static unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>11); }
static float rndf(void){ return (float)((int)(rnd()%2000)-1000)/1000.0f; }

/* double-precision mirror of the kernel's semantics: nibble-8 offset binary,
 * per-group partial dot times its scale. Not the identity gate -- the gate
 * that keeps identity from being identity between two wrong kernels. */
static void cpu_down_g4(const uint8_t *q,const float *sc,int K,int O,int gs,
                        const float *x,float *y){
    int rb=(K+1)/2, ng=gs>0?(K+gs-1)/gs:1, egs=gs>0?gs:K;
    for(int o=0;o<O;o++){
        const uint8_t *row=q+(size_t)o*rb; const float *scl=sc+(size_t)o*ng;
        double a=0;
        for(int g=0; g*egs<K; g++){
            int base=g*egs, glen=egs; if(base+glen>K) glen=K-base;
            double p=0;
            for(int i=base;i<base+glen;i++){
                uint8_t v=row[i>>1]; int n=(i&1)?(v>>4):(v&15);
                p+=(double)x[i]*(n-8);
            }
            a+=p*scl[g];
        }
        y[o]=(float)a;
    }
}

static double rel_rms(const float *a,const float *b,size_t n){
    double num=0,den=0;
    for(size_t i=0;i<n;i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return den>0?sqrt(num/den):sqrt(num);
}

#define COUNT 2

/* One shape, both arms, memcmp. rows per chunk are {1, 2} so the s/offset
 * indexing is exercised, not just the o axis. */
static void shape(int D,int I,int gs){
    const int rowsper[COUNT]={1,2};
    int offs[COUNT], total=0, max_rows=0;
    for(int c=0;c<COUNT;c++){ offs[c]=total; total+=rowsper[c];
                              if(rowsper[c]>max_rows) max_rows=rowsper[c]; }
    int rb=(I+1)/2, ng=gs>0?(I+gs-1)/gs:1;
    char name[160];

    uint8_t *hd[COUNT]; float *hds[COUNT];
    uint8_t *qd[COUNT]; float *sd[COUNT];
    GroupDesc host[COUNT];
    for(int c=0;c<COUNT;c++){
        hd[c]=(uint8_t*)malloc((size_t)D*rb);
        hds[c]=(float*)malloc((size_t)D*ng*4);
        for(size_t i=0;i<(size_t)D*rb;i++) hd[c][i]=(uint8_t)(rnd()&255);
        for(size_t i=0;i<(size_t)D*ng;i++) hds[c][i]=0.01f+0.05f*(float)(rnd()%1000)/1000.0f;
        cudaMalloc(&qd[c],(size_t)D*rb); cudaMalloc(&sd[c],(size_t)D*ng*4);
        cudaMemcpy(qd[c],hd[c],(size_t)D*rb,cudaMemcpyHostToDevice);
        offset_to_signed_s4<<<64,256>>>(qd[c],(size_t)D*rb);
        cudaMemcpy(sd[c],hds[c],(size_t)D*ng*4,cudaMemcpyHostToDevice);
        host[c]={NULL,NULL,qd[c],NULL,NULL,sd[c],4,4,4,rowsper[c],offs[c],gs,gs,gs};
    }
    GroupDesc *ddesc; cudaMalloc(&ddesc,sizeof host);
    cudaMemcpy(ddesc,host,sizeof host,cudaMemcpyHostToDevice);

    float *x,*y0,*yr;
    cudaMallocManaged(&x,(size_t)total*I*4);
    cudaMallocManaged(&y0,(size_t)total*D*4);
    cudaMallocManaged(&yr,(size_t)total*D*4);
    for(size_t i=0;i<(size_t)total*I;i++) x[i]=rndf();
    size_t yn=(size_t)total*D;

    setenv("COLI_CUDA_DOWN_ROWS","0",1);
    down_g4_launch(y0,x,ddesc,D,I,max_rows,COUNT,0);
    check(cudaDeviceSynchronize()==cudaSuccess,"original ran");

    /* the reference underneath the identity */
    double worst=0;
    for(int c=0;c<COUNT;c++)
        for(int s=0;s<rowsper[c];s++){
            float *ref=(float*)malloc((size_t)D*4);
            cpu_down_g4(hd[c],hds[c],I,D,gs,x+(size_t)(offs[c]+s)*I,ref);
            double r=rel_rms(y0+(size_t)(offs[c]+s)*D,ref,(size_t)D);
            if(r>worst) worst=r;
            free(ref);
        }
    snprintf(name,sizeof name,"original within reference bound [D=%d I=%d gs=%d]",D,I,gs);
    check(worst<1e-5,name);

    printf("  [D=%5d I=%5d gs=%2d] ref rms %.2e |",D,I,gs,worst);
    for(int R=2;R<=8;R*=2){
        char v[4]; snprintf(v,sizeof v,"%d",R);
        setenv("COLI_CUDA_DOWN_ROWS",v,1);
        memset(yr,0,yn*sizeof(float));
        down_g4_launch(yr,x,ddesc,D,I,max_rows,COUNT,0);
        snprintf(name,sizeof name,"R=%d ran [D=%d I=%d]",R,D,I);
        check(cudaDeviceSynchronize()==cudaSuccess,name);
        int same=!memcmp(y0,yr,yn*sizeof(float));
        printf("  R=%d %s",R,same?"bitwise":"DIFFERS");
        snprintf(name,sizeof name,"R=%d bitwise identical [D=%d I=%d gs=%d]",R,D,I,gs);
        check(same,name);
    }
    printf("\n");

    /* proof of bite: identity between two kernels that read nothing passes
     * every check above. Flip one nibble and the R kernel must move. */
    {
        size_t probe=(size_t)(rb>1?1:0);
        uint8_t before; cudaMemcpy(&before,qd[0]+probe,1,cudaMemcpyDeviceToHost);
        uint8_t after=(uint8_t)(before^0x11);
        cudaMemcpy(qd[0]+probe,&after,1,cudaMemcpyHostToDevice);
        setenv("COLI_CUDA_DOWN_ROWS","8",1);
        memset(yr,0,yn*sizeof(float));
        down_g4_launch(yr,x,ddesc,D,I,max_rows,COUNT,0);
        cudaDeviceSynchronize();
        snprintf(name,sizeof name,"mutated weight changes the R kernel's output [D=%d I=%d]",D,I);
        check(memcmp(y0,yr,yn*sizeof(float))!=0,name);
        cudaMemcpy(qd[0]+probe,&before,1,cudaMemcpyHostToDevice);
    }

    for(int c=0;c<COUNT;c++){ cudaFree(qd[c]); cudaFree(sd[c]); free(hd[c]); free(hds[c]); }
    cudaFree(ddesc); cudaFree(x); cudaFree(y0); cudaFree(yr);
}

int main(void){
    /* dispatch probe first: everything below is worthless if the branch is
     * never taken. */
    {
        const int D=64,I=128,gs=64, rb=(I+1)/2, ng=(I+gs-1)/gs;
        uint8_t *hd=(uint8_t*)malloc((size_t)D*rb);
        float *hds=(float*)malloc((size_t)D*ng*4);
        for(size_t i=0;i<(size_t)D*rb;i++) hd[i]=(uint8_t)(rnd()&255);
        for(size_t i=0;i<(size_t)D*ng;i++) hds[i]=0.02f;
        uint8_t *qd; float *sd;
        cudaMalloc(&qd,(size_t)D*rb); cudaMalloc(&sd,(size_t)D*ng*4);
        cudaMemcpy(qd,hd,(size_t)D*rb,cudaMemcpyHostToDevice);
        offset_to_signed_s4<<<64,256>>>(qd,(size_t)D*rb);
        cudaMemcpy(sd,hds,(size_t)D*ng*4,cudaMemcpyHostToDevice);
        GroupDesc host[1]={{NULL,NULL,qd,NULL,NULL,sd,4,4,4,1,0,gs,gs,gs}};
        GroupDesc *ddesc; cudaMalloc(&ddesc,sizeof host);
        cudaMemcpy(ddesc,host,sizeof host,cudaMemcpyHostToDevice);
        float *x,*y; cudaMallocManaged(&x,(size_t)I*4); cudaMallocManaged(&y,(size_t)D*4);
        for(int i=0;i<I;i++) x[i]=rndf();

        setenv("COLI_CUDA_DOWN_ROWS","0",1);
        check(down_rows_mode()==0,"ROWS=0 parses as off");
        uint64_t before=g_downr_launches;
        down_g4_launch(y,x,ddesc,D,I,1,1,0);
        check(cudaDeviceSynchronize()==cudaSuccess,"dispatch probe, ROWS=0");
        check(g_downr_launches==before,"ROWS=0 must NOT take the row-blocked branch");

        setenv("COLI_CUDA_DOWN_ROWS","8",1);
        check(down_rows_mode()==8,"ROWS=8 parses as 8");
        down_g4_launch(y,x,ddesc,D,I,1,1,0);
        check(cudaDeviceSynchronize()==cudaSuccess,"dispatch probe, ROWS=8");
        check(g_downr_launches==before+1,"ROWS=8 MUST take the row-blocked branch");

        setenv("COLI_CUDA_DOWN_ROWS","3",1);   /* not instantiated */
        check(down_rows_mode()==0,"an uninstantiated R reads as off, not rounded");

        cudaFree(qd); cudaFree(sd); cudaFree(ddesc); cudaFree(x); cudaFree(y);
        free(hd); free(hds);
    }
    printf("dispatch: the row-blocked branch fires on ROWS=8 and not on ROWS=0\n");

    printf("grouped_down_g4: R rows per block must be BITWISE identical\n");
    shape(2048, 512, 64);   /* qwen36 trunk geometry: D divisible by 2/4/8 */
    shape( 200,  96, 64);   /* tail group (96 % 64 = 32) */
    shape(  13,  96, 64);   /* D % R != 0 for every R: short last block */
    shape(   5,  96,  0);   /* D < R at R=8: one short block, per-row scales */
    shape(2050,  17,  0);   /* odd I: the i+1<I tail nibble */
    shape(   3,   1,  0);   /* one nibble, one column */
    shape( 512, 768, 64);   /* I > 2*256: more than one loop iteration/thread */

    /* determinism: the same launch twice must give the same bits */
    {
        setenv("COLI_CUDA_DOWN_ROWS","4",1);
        printf("determinism (R=4, same input twice):\n");
        shape(1024, 256, 64);
    }

    if(fails){ printf("FAIL: %d\n",fails); return 1; }
    printf("OK\n");
    return 0;
}
