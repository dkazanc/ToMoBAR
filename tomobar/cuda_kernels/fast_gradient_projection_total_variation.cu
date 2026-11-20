 /*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazantsev
Copyright 2017 Srikanth Nagella, Edoardo Pasca

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

// CuPy adaptation of CCPi FGP-TV (Beck and Teboulle).
// Zero-flux forward differences and their exact adjoint. No host allocations,
// residual scratch reuse, or device-wide synchronisation from the source.
#include <cuda_fp16.h>

template<typename T>
__global__ void fgp_primal(const float *f, float *u,
    const T *r0, const T *r1, const T *r2,
    float strength, int nonneg, int nx, int ny, int nz, int ndim) {
    const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    const long long size=static_cast<long long>(nx)*ny*nz;
    if(i>=size) return;
    const long long coord[3]={i%nx,(i/nx)%ny,i/(static_cast<long long>(nx)*ny)};
    const int shape[3]={nx,ny,nz};
    const long long stride[3]={1,nx,static_cast<long long>(nx)*ny};
    const T *r[3]={r0,r1,r2};
    float adjoint=0;
    for(int a=0;a<ndim;++a) {
        if(coord[a]>0) adjoint+=float(r[a][i-stride[a]]);
        if(coord[a]+1<shape[a]) adjoint-=float(r[a][i]);
    }
    const float value=f[i]-strength*adjoint;
    u[i]=nonneg ? fmaxf(value,0.0f) : value;
}

template<typename T>
__global__ void fgp_dual(const float *u,
    const T *r0, const T *r1, const T *r2,
    const T *old0, const T *old1, const T *old2,
    T *p0, T *p1, T *p2, T *next0, T *next1, T *next2,
    float strength, float momentum, int anisotropic,
    int nx, int ny, int nz, int ndim) {
    const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<long long>(nx)*ny*nz) return;
    const long long coord[3]={i%nx,(i/nx)%ny,i/(static_cast<long long>(nx)*ny)};
    const int shape[3]={nx,ny,nz};
    const long long stride[3]={1,nx,static_cast<long long>(nx)*ny};
    const T *r[3]={r0,r1,r2}, *old[3]={old0,old1,old2};
    T *p[3]={p0,p1,p2}, *next[3]={next0,next1,next2};
    float v[3]={0,0,0};
    for(int a=0;a<ndim;++a) {
        // Algebraically scaled projected gradient avoids 1/strength overflow.
        if(coord[a]+1<shape[a])
            v[a]=strength*float(r[a][i])+(u[i+stride[a]]-u[i])/(4.0f*ndim);
    }
    const float norm=hypotf(hypotf(v[0],v[1]),v[2]);
    for(int a=0;a<ndim;++a) {
        const float denominator=fmaxf(strength,anisotropic ? fabsf(v[a]) : norm);
        const float value=v[a]/denominator;
        // Momentum must use the actually stored projected dual in half mode.
        p[a][i]=value;
        next[a][i]=float(p[a][i])+momentum*(float(p[a][i])-float(old[a][i]));
    }
}
