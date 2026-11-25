 /*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev
Copyright 2019 Srikanth Nagella, Edoardo Pasca

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Adapted from CCPi TGV_GPU_core.cu. Corrected rectangular indexing,
// separate primal/extrapolated buffers, and exact zero-flux adjoints.
// Q packs diagonal components followed by sqrt(2)*off-diagonal components.
// P,Q are stored normalised to unit balls; all other fields are float32.
#include <cuda_fp16.h>

__device__ int tgv_pair(int a,int b,int ndim) {
    if(a==b) return a;
    if(a>b) { int tmp=a; a=b; b=tmp; }
    int index=ndim;
    for(int j=0;j<ndim;++j) for(int k=j+1;k<ndim;++k,++index)
        if(a==j && b==k) return index;
    return 0;
}

template<typename T>
__global__ void tgv_dual(const float *ubar,const float *vbar,T *p,T *q,
    float sigma,float alpha1,float alpha0,int nx,int ny,int nz,int ndim) {
    const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    const long long size=static_cast<long long>(nx)*ny*nz;
    if(i>=size) return;
    const long long coord[3]={i%nx,(i/nx)%ny,i/(static_cast<long long>(nx)*ny)};
    const int shape[3]={nx,ny,nz};
    const long long stride[3]={1,nx,static_cast<long long>(nx)*ny};
    float candidate[3]={0,0,0};
    float norm=0;
    for(int a=0;a<ndim;++a) {
        const float grad=(coord[a]+1<shape[a]) ? ubar[i+stride[a]]-ubar[i] : 0;
        candidate[a]=alpha1*float(p[a*size+i])+sigma*(grad-vbar[a*size+i]);
        norm=hypotf(norm,candidate[a]);
    }
    for(int a=0;a<ndim;++a) p[a*size+i]=candidate[a]/fmaxf(alpha1,norm);
    float tensor[6];
    for(int a=0;a<ndim;++a) {
        const float grad=(coord[a]+1<shape[a]) ? vbar[a*size+i+stride[a]]-vbar[a*size+i] : 0;
        tensor[a]=alpha0*float(q[a*size+i])+sigma*grad;
    }
    int index=ndim;
    for(int a=0;a<ndim;++a) for(int b=a+1;b<ndim;++b,++index) {
        const float ga=(coord[a]+1<shape[a]) ? vbar[b*size+i+stride[a]]-vbar[b*size+i] : 0;
        const float gb=(coord[b]+1<shape[b]) ? vbar[a*size+i+stride[b]]-vbar[a*size+i] : 0;
        tensor[index]=alpha0*float(q[index*size+i])+sigma*0.7071067811865475f*(ga+gb);
    }
    norm=0;
    for(int a=0;a<index;++a) norm=hypotf(norm,tensor[a]);
    for(int a=0;a<index;++a) q[a*size+i]=tensor[a]/fmaxf(alpha0,norm);
}

template<typename T>
__global__ void tgv_primal(const float *f,const float *u,float *nextu,float *ubar,
    const float *v,float *nextv,float *vbar,const T *p,const T *q,
    float weight,float fidelity_weight,float tau,float alpha1,float alpha0,int nonneg,
    int nx,int ny,int nz,int ndim) {
    const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    const long long size=static_cast<long long>(nx)*ny*nz;
    if(i>=size) return;
    const long long coord[3]={i%nx,(i/nx)%ny,i/(static_cast<long long>(nx)*ny)};
    const int shape[3]={nx,ny,nz};
    const long long stride[3]={1,nx,static_cast<long long>(nx)*ny};
    float adjoint=0;
    for(int a=0;a<ndim;++a) {
        if(coord[a]>0) adjoint+=float(p[a*size+i-stride[a]]);
        if(coord[a]+1<shape[a]) adjoint-=float(p[a*size+i]);
    }
    float value=weight*(u[i]-tau*alpha1*adjoint)+fidelity_weight*f[i];
    // Preserve NaNs for the wrapper stability check (fmaxf(NaN,0) hides them).
    if(nonneg && value<0) value=0;
    nextu[i]=value;
    ubar[i]=2*value-u[i];
    for(int a=0;a<ndim;++a) {
        float adjoint_e=0;
        for(int b=0;b<ndim;++b) {
            const int component=tgv_pair(a,b,ndim);
            const float factor=(a==b) ? 1.0f : 0.7071067811865475f;
            if(coord[b]>0) adjoint_e+=factor*float(q[component*size+i-stride[b]]);
            if(coord[b]+1<shape[b]) adjoint_e-=factor*float(q[component*size+i]);
        }
        const long long j=a*size+i;
        const float value_v=v[j]+tau*(alpha1*float(p[j])-alpha0*adjoint_e);
        nextv[j]=value_v;
        vbar[j]=2*value_v-v[j];
    }
}
