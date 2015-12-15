#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_functions.h"


// row pass using texture lookups

#include "helper_device_thread.h"

__global__ void
sharrFilterX(const float* src, float* dst, int width,int height )
{

    int idx = GetGlobalIdx();

    const float src_x_u_l  = src[idx - width - 1]*-3.0f;
    const float src_x_l    = src[idx - 1]*-10.0f;
    const float src_x_d_l  = src[idx + width - 1]*-3.0f;

    const float src_x_u_r  = src[idx - width + 1]* 3.0f;
    const float src_x_r    = src[idx + 1]* 10.0f;
    const float src_x_d_r  = src[idx + width + 1]*3.0f;

    dst[idx] = src_x_u_l+src_x_u_r+src_x_l+src_x_r+src_x_d_l+src_x_d_r;

}

__global__ void
sharrFilterY( const float* src, float* dst, int width,int height)
{
    int idx = GetGlobalIdx();

    const float src_y_u_l  = src[idx - width - 1]*-3.0f;
    const float src_y_l    = src[idx - width    ]*-10.0f;
    const float src_y_d_l  = src[idx - width + 1]*-3.0f;

    const float src_y_u_r  = src[idx + width - 1]* 3.0f;
    const float src_y_r    = src[idx + width    ]* 10.0f;
    const float src_y_d_r  = src[idx + width + 1]* 3.0f;

    dst[idx] = src_y_u_l+src_y_l+src_y_d_l+src_y_u_r+src_y_r+src_y_d_r;
}

__global__ void
sharrDerivativeFilterX(const float* src, float* dst, int width,int height, float mkernel, float ekernel, int step )
{

    int idx = GetGlobalIdx();

    const float src_x_u_l  = src[idx - width*step - step]*-ekernel;
    const float src_x_l    = src[idx - step]*-mkernel;
    const float src_x_d_l  = src[idx + width*step - step]*-ekernel;

    const float src_x_u_r  = src[idx - width*step + step]* ekernel;
    const float src_x_r    = src[idx + step]* mkernel;
    const float src_x_d_r  = src[idx + width*step + step]*ekernel;

    dst[idx] = src_x_u_l+src_x_u_r+src_x_l+src_x_r + src_x_d_l+src_x_d_r;

}

__global__ void
sharrDerivativeFilterY( const float* src, float* dst, int width,int height, float mkernel, float ekernel, int step )
{
    int idx = GetGlobalIdx();
    const float src_y_u_l  = src[idx - width*step - step]*-ekernel;
    const float src_y_l    = src[idx - width*step       ]*-mkernel;
    const float src_y_d_l  = src[idx - width*step + step]*-ekernel;

    const float src_y_u_r  = src[idx + width*step - step]* ekernel;
    const float src_y_r    = src[idx + width*step       ]* mkernel;
    const float src_y_d_r  = src[idx + width*step + step]* ekernel;

    dst[idx] = src_y_u_l+src_y_l+src_y_d_l+src_y_u_r+src_y_r+src_y_d_r;

}



__global__ void DownSize2x2Cuda(const float* src, float* dst, int width , int s_width )
{

    int idx = GetGlobalIdx();
    int row = idx/width;
    int row_idx = idx%width;

    int src_height = row*2*s_width;
    int src_width  = row_idx*2;
    int src_idx = src_height + src_width;

    float s0 = src[src_idx];
    float sr = src[src_idx+1];
    float sd = src[src_idx+s_width];
    float srd = src[src_idx+s_width+1];


    dst[idx] = (s0 + sr + sd + srd) * 0.25f;
}

__global__ void
DeterminantHessianCuda(const float* lxx, const float* lyy, const float* lxy, float* det, float sigma_quad)
{
    int idx = GetGlobalIdx();

    // since the numbers are really small multiply sigma_quad might be better?
    det[idx] = (lxx[idx]*lyy[idx]*sigma_quad - lxy[idx]*lxy[idx]*sigma_quad);

}



// TODO  sharr filter almost 10x slower, investigate!
void ScharrFilterCuda( const float* src, float* dst, int width, int height , int direction )
{

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    if( direction == 0)
    {
        sharrFilterX<<<numBlocks, threadsPerBlock>>>(src, dst, width, height);
    }
    else
    {
        sharrFilterY<<<numBlocks, threadsPerBlock>>>(src, dst, width, height);
    }

}

void SharrDerivativeCUDA(const float* src, float* dst, float mkernel,
                         float ekernel, int direction,int step, int width, int height)
{
    //TODO increase thread size to 4 improved,, memory fetching bottleneck?
    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    if( direction == 0)
    {
        sharrDerivativeFilterX<<<numBlocks, threadsPerBlock>>>(src, dst, width, height,mkernel,ekernel,step/2);
    }
    else
    {
        sharrDerivativeFilterY<<<numBlocks, threadsPerBlock>>>(src, dst, width, height,mkernel,ekernel,step/2);
    }
}


void DownSize2x2AreaFastCUDA( const float* src, float* dst, int width, int height  )
{

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    DownSize2x2Cuda<<<numBlocks, threadsPerBlock>>>(src, dst, width, width*2);
}


void DeterminantHessianCUDA(const float* lxx, const float* lyy, const float* lxy, float* det, float sigma_quad, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    DeterminantHessianCuda<<<numBlocks, threadsPerBlock>>>(lxx, lyy, lxy, det, sigma_quad);
}