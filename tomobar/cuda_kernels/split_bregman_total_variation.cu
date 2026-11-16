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

// Adapted from CCPi TV_SB_GPU_core.cu for CuPy. The image solve uses
// ping-pong Jacobi sweeps. Corrected boundary stencils use forward differences
// with zero boundary flux and their exact adjoint, matching ToMoBAR PD-TV.
// Auxiliary storage may be half; all arithmetic and primal images are float.
#include <cuda_fp16.h>

template<typename T>
__global__ void gauss_seidel2D_kernel(float *U, float *A, float *U_prev, T *Dx, T *Dy, T *Bx, T *By, float lambda, float mu, float normConst, int N, int M, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=N || j>=M) return;
    const int index=i+N*j;
    float sum=0; int degree=0;
    if (i>0) {
        sum += U_prev[index-(1)] + float(Dx[index-(1)]) - float(Bx[index-(1)]);
        ++degree;
    }
    if (i+1<N) {
        sum += U_prev[index+(1)] - float(Dx[index]) + float(Bx[index]);
        ++degree;
    }
    if (j>0) {
        sum += U_prev[index-(N)] + float(Dy[index-(N)]) - float(By[index-(N)]);
        ++degree;
    }
    if (j+1<M) {
        sum += U_prev[index+(N)] - float(Dy[index]) + float(By[index]);
        ++degree;
    }
    U[index]=(mu*A[index]+lambda*sum)/(mu+lambda*degree);
}

template<typename T>
__global__ void updDxDy_shrinkIso2D_kernel(float *U, T *Dx, T *Dy, T *Bx, T *By, float threshold, int N, int M, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=N || j>=M) return;
    const int index=i+N*j;
    const float v0=(i+1<N) ? U[index+(1)]-U[index]+float(Bx[index]) : 0.0f;
    const float v1=(j+1<M) ? U[index+(N)]-U[index]+float(By[index]) : 0.0f;
    const float norm=hypotf(v0,v1);
    const float scale=(norm>0) ? fmaxf(1.0f-threshold/norm,0.0f) : 0.0f;
    Dx[index]=scale*v0;
    Dy[index]=scale*v1;
}

template<typename T>
__global__ void updDxDy_shrinkAniso2D_kernel(float *U, T *Dx, T *Dy, T *Bx, T *By, float threshold, int N, int M, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=N || j>=M) return;
    const int index=i+N*j;
    const float v0=(i+1<N) ? U[index+(1)]-U[index]+float(Bx[index]) : 0.0f;
    const float v1=(j+1<M) ? U[index+(N)]-U[index]+float(By[index]) : 0.0f;
    Dx[index]=copysignf(fmaxf(fabsf(v0)-threshold,0.0f),v0);
    Dy[index]=copysignf(fmaxf(fabsf(v1)-threshold,0.0f),v1);
}

template<typename T>
__global__ void updBxBy2D_kernel(float *U, T *Dx, T *Dy, T *Bx, T *By, int N, int M, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=N || j>=M) return;
    const int index=i+N*j;
    Bx[index]=(i+1<N) ? float(Bx[index])+U[index+(1)]-U[index]-float(Dx[index]) : 0.0f;
    By[index]=(j+1<M) ? float(By[index])+U[index+(N)]-U[index]-float(Dy[index]) : 0.0f;
}

template<typename T>
__global__ void gauss_seidel3D_kernel(float *U, float *A, float *U_prev, T *Dx, T *Dy, T *Dz, T *Bx, T *By, T *Bz, float lambda, float mu, float normConst, int N, int M, int Z, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    const int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i>=N || j>=M || k>=Z) return;
    const int index=i+N*j+N*M*k;
    float sum=0; int degree=0;
    if (i>0) {
        sum += U_prev[index-(1)] + float(Dx[index-(1)]) - float(Bx[index-(1)]);
        ++degree;
    }
    if (i+1<N) {
        sum += U_prev[index+(1)] - float(Dx[index]) + float(Bx[index]);
        ++degree;
    }
    if (j>0) {
        sum += U_prev[index-(N)] + float(Dy[index-(N)]) - float(By[index-(N)]);
        ++degree;
    }
    if (j+1<M) {
        sum += U_prev[index+(N)] - float(Dy[index]) + float(By[index]);
        ++degree;
    }
    if (k>0) {
        sum += U_prev[index-(N*M)] + float(Dz[index-(N*M)]) - float(Bz[index-(N*M)]);
        ++degree;
    }
    if (k+1<Z) {
        sum += U_prev[index+(N*M)] - float(Dz[index]) + float(Bz[index]);
        ++degree;
    }
    U[index]=(mu*A[index]+lambda*sum)/(mu+lambda*degree);
}

template<typename T>
__global__ void updDxDy_shrinkIso3D_kernel(float *U, T *Dx, T *Dy, T *Dz, T *Bx, T *By, T *Bz, float threshold, int N, int M, int Z, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    const int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i>=N || j>=M || k>=Z) return;
    const int index=i+N*j+N*M*k;
    const float v0=(i+1<N) ? U[index+(1)]-U[index]+float(Bx[index]) : 0.0f;
    const float v1=(j+1<M) ? U[index+(N)]-U[index]+float(By[index]) : 0.0f;
    const float v2=(k+1<Z) ? U[index+(N*M)]-U[index]+float(Bz[index]) : 0.0f;
    const float norm=hypotf(hypotf(v0,v1),v2);
    const float scale=(norm>0) ? fmaxf(1.0f-threshold/norm,0.0f) : 0.0f;
    Dx[index]=scale*v0;
    Dy[index]=scale*v1;
    Dz[index]=scale*v2;
}

template<typename T>
__global__ void updDxDy_shrinkAniso3D_kernel(float *U, T *Dx, T *Dy, T *Dz, T *Bx, T *By, T *Bz, float threshold, int N, int M, int Z, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    const int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i>=N || j>=M || k>=Z) return;
    const int index=i+N*j+N*M*k;
    const float v0=(i+1<N) ? U[index+(1)]-U[index]+float(Bx[index]) : 0.0f;
    const float v1=(j+1<M) ? U[index+(N)]-U[index]+float(By[index]) : 0.0f;
    const float v2=(k+1<Z) ? U[index+(N*M)]-U[index]+float(Bz[index]) : 0.0f;
    Dx[index]=copysignf(fmaxf(fabsf(v0)-threshold,0.0f),v0);
    Dy[index]=copysignf(fmaxf(fabsf(v1)-threshold,0.0f),v1);
    Dz[index]=copysignf(fmaxf(fabsf(v2)-threshold,0.0f),v2);
}

template<typename T>
__global__ void updBxBy3D_kernel(float *U, T *Dx, T *Dy, T *Dz, T *Bx, T *By, T *Bz, int N, int M, int Z, int ImSize)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    const int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i>=N || j>=M || k>=Z) return;
    const int index=i+N*j+N*M*k;
    Bx[index]=(i+1<N) ? float(Bx[index])+U[index+(1)]-U[index]-float(Dx[index]) : 0.0f;
    By[index]=(j+1<M) ? float(By[index])+U[index+(N)]-U[index]-float(Dy[index]) : 0.0f;
    Bz[index]=(k+1<Z) ? float(Bz[index])+U[index+(N*M)]-U[index]-float(Dz[index]) : 0.0f;
}
