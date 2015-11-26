#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "helper_image.h"
// row pass using texture lookups

#include "helper_device_thread.h"


__device__ __constant__ float Kenrnel[128];

__global__ void GaussianBlur(  const float* src, float* dst, int width, const int ksize)
{
    //int idx = GetGlobalIdx();
    int mid = ksize/2;
    float sum = 0.0f;

    int pix_i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pix_j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int idx = pix_j*width + pix_i;

    #pragma unroll
    for (int j = 0; j < ksize; ++j)
    {
        int y_off = (j - mid);

        int bounded_y =  min( max( y_off + pix_j, 0), width-1 ) - pix_j;

        int pix_y_offset = bounded_y*width;
#pragma unroll
        for(int i = 0; i < ksize ; ++i)
        {
            int x_off = i - mid;

            int bounded_x =  min( max( x_off + pix_i, 0), width-1 ) - pix_i;

            int final_idx = idx + pix_y_offset + bounded_x;

            float src_val = src[final_idx];
            float filter = Kenrnel[i + j*ksize];
            sum += src_val*filter;
        }
    }

    dst[idx] = sum;
}


__global__ void GaussianHPass(  const float* src, float* dst, int width, const int ksize)
{
//int idx = GetGlobalIdx();
    int mid = ksize/2;
    float sum = 0.0f;

    int pix_i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pix_j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int idx = pix_j*width + pix_i;

#pragma unroll
    for(int i = 0; i < ksize ; ++i)
    {
        int x_off = i - mid;

        int bounded_x =  min( max( x_off + pix_i, 0), width-1 ) - pix_i;

        int final_idx = idx  + bounded_x;

        float src_val = src[final_idx];
        float filter = Kenrnel[i];
        sum += src_val*filter;
    }

    dst[idx] = sum;
}



__global__ void GaussianVPass(  const float* src, float* dst, int width, const int ksize)
{
//int idx = GetGlobalIdx();
    int mid = ksize/2;
    float sum = 0.0f;

    int pix_i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pix_j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int idx = pix_j*width + pix_i;

    #pragma unroll
    for (int j = 0; j < ksize; ++j)
    {
        int y_off = (j - mid);

        int bounded_y =  min( max( y_off + pix_j, 0), width-1 ) - pix_j;

        int pix_y_offset = bounded_y*width;

        //for(int i = 0; i < ksize ; ++i)
        int final_idx = idx + pix_y_offset ;

        float src_val = src[final_idx];
        float filter = Kenrnel[j];
        sum += src_val*filter;
    }


    dst[idx] = sum;
}


static float HostK[128];
void GaussianCUDA5x5( const float* src, float* dst, float sigma, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    GetGaussianKernel( 5, sigma, HostK);
    cudaMemcpyToSymbolAsync(Kenrnel, HostK, 25*sizeof(float), 0 , cudaMemcpyHostToDevice);

    GaussianBlur<<<numBlocks, threadsPerBlock>>>(src, dst, width, 5 );

}


void GaussianCUDA9x9( const float* src, float* dst, float sigma, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    GetGaussianKernel( 9, sigma, HostK);
    cudaMemcpyToSymbolAsync(Kenrnel, HostK, 81*sizeof(float), 0 , cudaMemcpyHostToDevice);

    GaussianBlur<<<numBlocks, threadsPerBlock>>>(src, dst, width, 9 );
}

void GaussianCUDA5x5TwoPass( const float* src, float* interm, float* dst, float sigma, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    GetGaussianLinear( 5, sigma, HostK);
    cudaMemcpyToSymbolAsync(Kenrnel, HostK, 81*sizeof(float), 0 , cudaMemcpyHostToDevice);

    GaussianHPass<<<numBlocks, threadsPerBlock>>>(src, interm, width, 5 );
    GaussianVPass<<<numBlocks, threadsPerBlock>>>(interm, dst, width, 5 );
}


void GaussianCUDA9x9TwoPass( const float* src, float* interm, float* dst, float sigma, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    GetGaussianLinear( 9, sigma, HostK);
    cudaMemcpyToSymbolAsync(Kenrnel, HostK, 81*sizeof(float), 0 , cudaMemcpyHostToDevice);

    GaussianHPass<<<numBlocks, threadsPerBlock>>>(src, interm, width, 9 );
    cudaDeviceSynchronize();
    GaussianVPass<<<numBlocks, threadsPerBlock>>>(interm, dst, width, 9 );
}

