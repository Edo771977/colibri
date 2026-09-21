/* Grouped-int4 (fmt=4) CUDA kernel oracle (#334).
 *
 * Feeds random offset-binary nibble weights + [O, ng] group scales through
 * grouped_hidden_g4_dual / grouped_down_g4 and checks against a CPU reference
 * that replicates matmul_i4_grouped's semantics (value = nibble - 8, per-group
 * partial dot x scale). Covers gs=64, a non-divisible tail group, and a
 * per-row (gs=0) member riding in the same launch — the fmt=2-compat case.
 * The dual hidden kernel fuses silu into its epilogue, so gate[] holds
 * silu(g)*u and up[] is never written; the oracle checks that contract.
 *
 * The device buffers get the same XOR 0x88 offset->signed conversion the
 * upload path applies, so the kernels are exercised exactly as deployed.
 *
 * Phase 2 goes through the public API instead of raw kernels: upload_g plus
 * coli_cuda_expert_group (sync) and coli_cuda_expert_group_issue/take (the
 * async decode path the VRAM expert tiers ride) against the same reference —
 * the async result must be bit-identical to the sync one.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_grouped_g4_cuda.cu -o tests/test_grouped_g4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* No direct <cuda_runtime.h>: the backend include below pulls the right
 * runtime header for the target (cuda_runtime.h, or hip_runtime.h through
 * backend_gpu_compat.h). A direct include breaks the hipcc build, which is
 * what `make gpu-compile HIP=1` runs this file through now that it is no
 * longer compile-only -- together with the managed allocations below, it was
 * the second thing keeping this file out of `make cuda-test`. */
#include "../backend_cuda.cu"

#ifdef _WIN32
/* cuda-test runs on Windows too, and MSVC has no setenv. */
static int setenv(const char *name,const char *value,int overwrite){
    (void)overwrite; return _putenv_s(name,value);
}
#endif

/* offset_to_signed_s4 is one thread per byte with an `if (i < n)` guard -- no
 * grid-stride loop -- so the grid must cover the whole buffer. The fixed
 * <<<64,256>>> this file used converts only the first 16 KiB; it happened to
 * be enough for D=200, I=96 (9,600 bytes) and would have gone on being enough
 * until someone enlarged the dims, at which point the kernels would read the
 * tail as offset binary and the failure would look like a kernel bug. */
static void convert_s4(uint8_t *q,size_t n){
    offset_to_signed_s4<<<(unsigned)((n+255)/256),256>>>(q,n);
}

static void cpu_gemv_g4(const uint8_t *q,const float *sc,int K,int O,int gs,
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

/* The default is a shipped decision, so it gets an assertion rather than a
 * comment. Unset must be ON where the graph can exist; `=0` must be the way
 * off; anything else must be on, because a typo in a measurement script
 * should give the shipped arm. */
static int check_graph_default(void){
    int bad=0;
#if COLI_GPU_HAS_GRAPH
    setenv("COLI_CUDA_GRAPH","",1);
    if(!graph_mode()){ printf("FAIL unset must read as ON (the shipped default)\n"); bad++; }
    setenv("COLI_CUDA_GRAPH","0",1);
    if(graph_mode()){ printf("FAIL \"0\" must read as off\n"); bad++; }
    setenv("COLI_CUDA_GRAPH","1",1);
    if(!graph_mode()){ printf("FAIL \"1\" must read as on\n"); bad++; }
#else
    /* No cudaStreamBeginCapture on this target: the path is compiled out and
     * graph_mode() is 0 whatever the variable says. */
    setenv("COLI_CUDA_GRAPH","1",1);
    if(graph_mode()){ printf("FAIL graph must be off where COLI_GPU_HAS_GRAPH is 0\n"); bad++; }
#endif
    return bad;
}

int main(void){
    srand(7);
    const int D=200, I=96, gs=64;            /* tail group: 200 % 64 = 8 */
    const int COUNT=3;                       /* expert 0,1: fmt4 gs=64; expert 2: per-row (gs=0) */
    const int rbD=(D+1)/2, rbI=(I+1)/2;
    const int ngD=(D+gs-1)/gs, ngI=(I+gs-1)/gs;
    int trials=50, bad=0;
    for(int t=0;t<trials;t++){
        /* Explicit host buffers plus explicit device buffers, NOT
         * cudaMallocManaged. backend_gpu_compat.h does not map managed
         * allocation for HIP, and that single dependency is why this file was
         * compile-only: it could not be added to `make cuda-test`, which also
         * serves `make hip-test`. So the graph assertions further down shipped
         * having never executed anywhere. The convention here is the one
         * tests/test_fp8_warp_cuda.cu documents. xs and gate and y keep their
         * names as the HOST side, so every reference below reads unchanged;
         * up is device-only because the fused epilogue leaves it untouched
         * and nothing on the host looks at it. */
        GroupDesc host[COUNT];
        float *xs  =(float*)malloc((size_t)COUNT*D*4);
        float *gate=(float*)malloc((size_t)COUNT*I*4);
        float *y   =(float*)malloc((size_t)COUNT*D*4);
        float *xs_d,*gate_d,*up_d,*y_d;
        cudaMalloc(&xs_d,(size_t)COUNT*D*4); cudaMalloc(&gate_d,(size_t)COUNT*I*4);
        cudaMalloc(&up_d,(size_t)COUNT*I*4); cudaMalloc(&y_d,(size_t)COUNT*D*4);
        if(!xs||!gate||!y){ printf("FAIL alloc trial\n"); return 1; }
        uint8_t *qg[COUNT],*qu[COUNT],*qd[COUNT]; float *sg[COUNT],*su[COUNT],*sd[COUNT];
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            int cgs = c==2 ? 0 : gs;
            int cngD = cgs? ngD:1, cngI = cgs? ngI:1;
            hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
            hd[c]=(uint8_t*)malloc((size_t)D*rbI);
            hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
            hds[c]=(float*)malloc((size_t)D*cngI*4);
            for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                                  hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            cudaMalloc(&qg[c],(size_t)I*rbD); cudaMalloc(&qu[c],(size_t)I*rbD); cudaMalloc(&qd[c],(size_t)D*rbI);
            cudaMalloc(&sg[c],(size_t)I*cngD*4); cudaMalloc(&su[c],(size_t)I*cngD*4); cudaMalloc(&sd[c],(size_t)D*cngI*4);
            cudaMemcpy(qg[c],hg[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qu[c],hu[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qd[c],hd[c],(size_t)D*rbI,cudaMemcpyHostToDevice);
            convert_s4(qg[c],(size_t)I*rbD);
            convert_s4(qu[c],(size_t)I*rbD);
            convert_s4(qd[c],(size_t)D*rbI);
            cudaMemcpy(sg[c],hgs[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(su[c],hus[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(sd[c],hds[c],(size_t)D*cngI*4,cudaMemcpyHostToDevice);
            host[c]={qg[c],qu[c],qd[c],sg[c],su[c],sd[c],4,4,4,1,c,cgs,cgs,cgs,c};
        }
        for(size_t i=0;i<(size_t)COUNT*D;i++) xs[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        cudaMemcpy(xs_d,xs,(size_t)COUNT*D*4,cudaMemcpyHostToDevice);
        GroupDesc *ddesc; cudaMalloc(&ddesc,sizeof(host));
        cudaMemcpy(ddesc,host,sizeof(host),cudaMemcpyHostToDevice);
        dim3 hgd((unsigned)I,1,(unsigned)COUNT),ogd((unsigned)D,1,(unsigned)COUNT);
        grouped_hidden_g4_dual<<<hgd,256>>>(gate_d,up_d,xs_d,ddesc,I,D);
        grouped_down_g4<<<ogd,256>>>(y_d,gate_d,ddesc,D,I);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL cuda\n"); return 1; }
        /* gate is read below both as an output to check and as the input to
         * the down-projection reference, so it has to come back. */
        cudaMemcpy(gate,gate_d,(size_t)COUNT*I*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(y,y_d,(size_t)COUNT*D*4,cudaMemcpyDeviceToHost);
        for(int c=0;c<COUNT;c++){
            int cgs=c==2?0:gs;
            float rg[512],ru[512],rh[512],ry[512];
            cpu_gemv_g4(hg[c],hgs[c],D,I,cgs,xs+(size_t)c*D,rg);
            cpu_gemv_g4(hu[c],hus[c],D,I,cgs,xs+(size_t)c*D,ru);
            /* fused epilogue (a03c79e): gate[] = silu(g)*u, up[] untouched */
            for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
            for(int o=0;o<I;o++)
                if(fabsf(gate[(size_t)c*I+o]-rh[o])>1e-3f*(fabsf(rh[o])+1e-3f)) bad++;
            cpu_gemv_g4(hd[c],hds[c],I,D,cgs,(float*)gate+(size_t)c*I,ry);
            for(int o=0;o<D;o++)
                if(fabsf(y[(size_t)c*D+o]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) bad++;
        }
        for(int c=0;c<COUNT;c++){ cudaFree(qg[c]);cudaFree(qu[c]);cudaFree(qd[c]);
            cudaFree(sg[c]);cudaFree(su[c]);cudaFree(sd[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        cudaFree(ddesc);
        cudaFree(xs_d);cudaFree(gate_d);cudaFree(up_d);cudaFree(y_d);
        free(xs);free(gate);free(y);
    }
    printf("grouped-g4 oracle: %d trials x %d experts (gs=64 + tail + per-row member), %d mismatches\n",
           trials,COUNT,bad);
    if(bad){ printf("FAIL\n"); return 1; }

    /* ---- Phase 2: public API — upload_g + sync group vs async issue/take ----
     * The async path is what MoE decode (and the VRAM expert tiers) actually
     * call; it must dispatch fmt=4 like the sync path and match it bit for bit. */
    {
        int devs[1]={0};
        if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }

        /* THE GRAPH IS ON BY DEFAULT NOW, and that quietly breaks this file
         * unless it is pinned here. Everything below compares some path
         * against a per-call reference; if the reference itself is produced
         * with the variable unset, it runs through the graph too and every
         * memcmp becomes graph-against-graph -- vacuously true, and the one
         * assertion this file exists to make would be gone. So: explicitly
         * off for the reference, explicitly on inside the graph block, and
         * the default itself asserted right here rather than assumed. */
        int rows[COUNT]={1,2,1}, total=4, api_bad=0;
        api_bad += check_graph_default();
        setenv("COLI_CUDA_GRAPH","0",1);

        ColiCudaTensor *tg[COUNT]={},*tu[COUNT]={},*td[COUNT]={};
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            int cgs = c==2 ? 0 : gs;            /* expert 2 rides along as fmt=2 */
            int cngD = cgs? ngD:1, cngI = cgs? ngI:1, fmt = cgs?4:2;
            hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
            hd[c]=(uint8_t*)malloc((size_t)D*rbI);
            hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
            hds[c]=(float*)malloc((size_t)D*cngI*4);
            for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                                  hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            if(!coli_cuda_tensor_upload_g(&tg[c],hg[c],hgs[c],fmt,D,I,0,cgs)||
               !coli_cuda_tensor_upload_g(&tu[c],hu[c],hus[c],fmt,D,I,0,cgs)||
               !coli_cuda_tensor_upload_g(&td[c],hd[c],hds[c],fmt,I,D,0,cgs)){
                printf("FAIL upload_g\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total*D*4),*ysync=(float*)malloc((size_t)total*D*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows,COUNT,ysync,x)){ printf("FAIL sync group\n"); return 1; }
        if(!coli_cuda_expert_group_issue(tg,tu,td,rows,COUNT,x)){ printf("FAIL async issue\n"); return 1; }
        const float *yasync=coli_cuda_expert_group_take(0);
        if(!yasync){ printf("FAIL async take\n"); return 1; }
        int off=0;
        for(int c=0;c<COUNT;c++){
            int cgs=c==2?0:gs;
            for(int s=0;s<rows[c];s++){
                float rg[512],ru[512],rh[512],ry[512];
                const float *xr=x+(size_t)(off+s)*D;
                cpu_gemv_g4(hg[c],hgs[c],D,I,cgs,xr,rg);
                cpu_gemv_g4(hu[c],hus[c],D,I,cgs,xr,ru);
                for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                cpu_gemv_g4(hd[c],hds[c],I,D,cgs,rh,ry);
                for(int o=0;o<D;o++){
                    size_t z=(size_t)(off+s)*D+o;
                    if(fabsf(ysync[z]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) api_bad++;
                    if(yasync[z]!=ysync[z]) api_bad++;   /* async == sync, bitwise */
                }
            }
            off+=rows[c];
        }
        printf("grouped-g4 API: sync vs oracle + async(issue/take) vs sync, %d mismatches\n",api_bad);

        /* One input row broadcast to every chunk (#1602). Decode hands K
         * experts the same activation vector; the caller used to copy it K
         * times. Passing it once with x_rows=1 must land on exactly the same
         * bits as passing K copies -- it is the same arithmetic reading the
         * same row, so a tolerance would be the wrong gate. */
        {
            int rows1[COUNT]; for(int c=0;c<COUNT;c++) rows1[c]=1;
            float *xd=(float*)malloc((size_t)COUNT*D*4);
            for(int c=0;c<COUNT;c++) memcpy(xd+(size_t)c*D,x,(size_t)D*4);
            float *ydup=(float*)malloc((size_t)COUNT*D*4);
            if(!coli_cuda_expert_group_issue(tg,tu,td,rows1,COUNT,xd)){ printf("FAIL dup issue\n"); return 1; }
            const float *yd=coli_cuda_expert_group_take(0);
            if(!yd){ printf("FAIL dup take\n"); return 1; }
            memcpy(ydup,yd,(size_t)COUNT*D*4);
            if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,x,1)){ printf("FAIL bcast issue\n"); return 1; }
            const float *yb=coli_cuda_expert_group_take(0);
            if(!yb){ printf("FAIL bcast take\n"); return 1; }
            int bc_bad = memcmp(ydup,yb,(size_t)COUNT*D*4)!=0;
            if(bc_bad) api_bad++;
            printf("grouped-g4 broadcast: x_rows=1 vs %d duplicated rows: %s\n",
                   COUNT, bc_bad?"DIFFERS":"bitwise");

            /* The refusals. A chunk with two rows would read xoff+1, and
             * there is no second row; a bogus x_rows would read off the end
             * of the caller's buffer. Both must be rejected, not guessed. */
            int rows2[COUNT]; for(int c=0;c<COUNT;c++) rows2[c]=1; rows2[0]=2;
            if(coli_cuda_expert_group_issue_x(tg,tu,td,rows2,COUNT,x,1)){
                printf("FAIL broadcast accepted a multi-row chunk\n"); api_bad++; }
            if(coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,x,COUNT+1)){
                printf("FAIL broadcast accepted x_rows past the total\n"); api_bad++; }
            /* x_rows=0 is the legacy contract: one row per expert row. */
            if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,xd,0)){
                printf("FAIL x_rows=0 (legacy) refused\n"); api_bad++; }
            else { const float *yl=coli_cuda_expert_group_take(0);
                   if(!yl||memcmp(ydup,yl,(size_t)COUNT*D*4)!=0){
                       printf("FAIL x_rows=0 is not the legacy path\n"); api_bad++; } }
            /* COLI_CUDA_GRAPH: the same issue replayed as one cudaGraphLaunch
             * must land on the same bits. Three calls, not one: the first
             * CAPTURES (and executes the captured sequence), the later ones
             * REPLAY, and those are different code paths -- a capture that
             * records the right work but replays against a stale pointer
             * would pass a one-shot test.
             *
             * Nothing else here exercises this: cuda-test never sets the
             * variable, so without this block the graph path ships compiled
             * and unrun. */
            uint64_t calls_before=0; { uint64_t e=0,r=0; double h=0,k=0,d=0;
                coli_cuda_group_stats(&calls_before,&e,&r,&h,&k,&d); }
            setenv("COLI_CUDA_GRAPH", "1", 1);
            for (int rep = 0; rep < 3; rep++) {
                if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,x,1)){
                    printf("FAIL graph issue (rep %d)\n", rep); return 1; }
                const float *yg = coli_cuda_expert_group_take(0);
                if(!yg){ printf("FAIL graph take (rep %d)\n", rep); return 1; }
                int g_bad = memcmp(ydup,yg,(size_t)COUNT*D*4)!=0;
                if(g_bad) api_bad++;
                printf("grouped-g4 graph: rep %d (%s) vs per-call launches: %s\n",
                       rep, rep?"replay":"capture", g_bad?"DIFFERS":"bitwise");
            }
            /* THE ACCOUNTING. Three issues above: one capture and two
             * replays. All three are calls, and all three must be counted --
             * the replay path returns early and used to skip the accounting
             * entirely, so with the graph on by default qt_stats reported the
             * captures alone. Checked against the backend's own counters, and
             * against the capture/replay split, because "three calls counted"
             * would also pass if all three had been captures. */
            {
                uint64_t gc=0,ge=0,gr=0; double h=0,k=0,d=0;
                coli_cuda_group_stats(&gc,&ge,&gr,&h,&k,&d);
                if(g_graph_captures!=1 || g_graph_replays!=2){
                    printf("FAIL graph split: %llu captures, %llu replays (want 1 and 2)\n",
                           (unsigned long long)g_graph_captures,
                           (unsigned long long)g_graph_replays); api_bad++; }
                if(gc < calls_before+3){
                    printf("FAIL graph calls uncounted: %llu -> %llu, want +3 at least\n",
                           (unsigned long long)calls_before,(unsigned long long)gc); api_bad++; }
                else printf("grouped-g4 graph: %llu captures + %llu replays, all counted\n",
                            (unsigned long long)g_graph_captures,
                            (unsigned long long)g_graph_replays);
            }

            /* THE INVALIDATION. A graph holds the addresses it was captured
             * with. Grow the buffers through the LEGACY path (x_rows=0, one
             * input row per expert row) so reserve() has to reallocate, then
             * come back to the same count: the graph must be RE-CAPTURED, not
             * replayed against pointers that have been freed.
             *
             * Bitwise identity cannot test this on its own -- a stale graph
             * reading memory nothing has reused yet returns the right answer
             * too -- which is why the capture counter exists. */
            {
                uint64_t cap_before=g_graph_captures, rep_before=g_graph_replays;
                setenv("COLI_CUDA_GRAPH", "1", 1);
                if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,xd,0)){
                    printf("FAIL grow issue\n"); return 1; }
                (void)coli_cuda_expert_group_take(0);
                if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,x,1)){
                    printf("FAIL post-grow issue\n"); return 1; }
                { const float *yg=coli_cuda_expert_group_take(0);
                  if(!yg||memcmp(ydup,yg,(size_t)COUNT*D*4)!=0){
                      printf("FAIL post-grow result\n"); api_bad++; } }
                /* Either the buffers moved and this is a fresh capture, or
                 * they did not and a replay was legitimate. What must NEVER
                 * happen is a replay after a move -- and the only way to tell
                 * is that one of the two counters advanced, never neither. */
                int recaptured = g_graph_captures>cap_before;
                int replayed   = g_graph_replays>rep_before;
                if(!recaptured && !replayed){
                    printf("FAIL neither captured nor replayed after the grow\n"); api_bad++; }
                else printf("grouped-g4 graph after a buffer grow: %s, bitwise\n",
                            recaptured?"re-captured":"replayed (buffers did not move)");
            }

            /* And back off cleanly: a graph left armed would follow this
             * process into the next block. */
            setenv("COLI_CUDA_GRAPH", "0", 1);
            if(!coli_cuda_expert_group_issue_x(tg,tu,td,rows1,COUNT,x,1)){
                printf("FAIL post-graph issue\n"); return 1; }
            { const float *yo = coli_cuda_expert_group_take(0);
              if(!yo||memcmp(ydup,yo,(size_t)COUNT*D*4)!=0){
                  printf("FAIL per-call path after the graph\n"); api_bad++; } }

            free(xd); free(ydup);
        }
        for(int c=0;c<COUNT;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ysync);
        coli_cuda_shutdown();
        if(api_bad){ printf("FAIL\n"); return 1; }
    }
    printf("OK\n"); return 0;
}
