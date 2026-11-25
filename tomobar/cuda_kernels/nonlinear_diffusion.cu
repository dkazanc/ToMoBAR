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

// CuPy adaptation: deterministic out-of-place explicit diffusion,
// zero-flux edge stencils, 64-bit flat indexing, and projected image updates.
// Host wrapper enforces a nonnegative-coefficient explicit time step.
__device__ float ndf_flux(float difference,float sigma,int penalty) {
    if(sigma==0) return difference;
    const float magnitude=fabsf(difference);
    if(penalty==1 || penalty==5) {
        if(penalty==5 && magnitude>sigma && magnitude-sigma>sigma) return 0;
        return magnitude>sigma ? copysignf(1.0f,difference) : difference/sigma;
    }
    if(penalty==2) {
        // Reciprocal branch avoids squaring an unbounded difference/sigma.
        if(magnitude>sigma) {
            const float ratio=sigma/magnitude;
            return copysignf(sigma*ratio/(1+ratio*ratio),difference);
        }
        const float ratio=difference/sigma;
        return difference/(1+ratio*ratio);
    }
    if(penalty==3) {
        if(magnitude>sigma) return 0;
        const float ratio=difference/sigma;
        const float factor=1-ratio*ratio;
        return difference*factor*factor;
    }
    return magnitude>sigma ? 0 : difference; // penalty 4
}

extern "C" __global__ void nonlinear_diffusion(
    const float *f,const float *current,float *next,
    float step,float step_strength,float sigma,int penalty,int nonneg,
    int nx,int ny,int nz,int ndim) {
    const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<long long>(nx)*ny*nz) return;
    const long long coord[3]={i%nx,(i/nx)%ny,i/(static_cast<long long>(nx)*ny)};
    const int shape[3]={nx,ny,nz};
    const long long stride[3]={1,nx,static_cast<long long>(nx)*ny};
    float flux=0;
    for(int axis=0;axis<ndim;++axis) {
        if(coord[axis]>0) flux+=ndf_flux(current[i-stride[axis]]-current[i],sigma,penalty);
        if(coord[axis]+1<shape[axis]) flux+=ndf_flux(current[i+stride[axis]]-current[i],sigma,penalty);
    }
    float value=current[i]+step_strength*flux-step*(current[i]-f[i]);
    // Keep NaNs visible so the wrapper reports numerical failures.
    if(nonneg && value<0) value=0;
    next[i]=value;
}
