#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "helper_device_thread.h"

__global__ void
PM_G2Cuda(  const float* lx, const float* ly, float* dst, float inv_k )
{
    int idx = GetGlobalIdx();

    const float lx_val  = lx[idx];
    const float ly_val  = ly[idx];
    dst[idx] = 1.0f/( 1.0f + inv_k*(lx_val*lx_val + ly_val*ly_val) );
/*
    cv::Size sz = Lx.size();
    float inv_k = 1.0 / (k*k);
    for (int y = 0; y < sz.height; y++) {
        const float* Lx_row = Lx.ptr<float>(y);
        const float* Ly_row = Ly.ptr<float>(y);
        float* dst_row = dst.ptr<float>(y);
        for (int x = 0; x < sz.width; x++)
            dst_row[x] = 1.0 / (1.0+inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]));
    }
    */
}
//////////////////////////////////////////////////////////////////////////////////
__global__ void
NLDStepScalarCuda( const float* csrc,const float* stLd, float* dlt, float hstepSize, int width)
{
    int idx = GetGlobalIdx();

    const float src_u = csrc[idx - width];
    const float src_m = csrc[idx];
    const float src_d = csrc[idx + width];
    const float src_l = csrc[idx-1];
    const float src_r = csrc[idx+1];

    const float dstLd_u = stLd[idx - width];
    const float dstLd_m = stLd[idx];
    const float dstLd_d = stLd[idx + width];
    const float dstLd_l = stLd[idx-1];
    const float dstLd_r = stLd[idx+1];

    float xpos =  (src_m+src_r)*(dstLd_r-dstLd_m);
    float xneg =  (src_l+src_m)*(dstLd_m-dstLd_l);
    float ypos =  (src_m+src_d)*(dstLd_d-dstLd_m);
    float yneg =  (src_u+src_m)*(dstLd_m-dstLd_u);

    float step_val =  hstepSize*(xpos-xneg + ypos-yneg);
    dlt[idx] = dstLd_m + step_val;
    //lStep[idx] = hstepSize*(xpos-xneg + ypos-yneg);
/*
    // First row
    const float* c_row = c.ptr<float>(0);
    const float* c_row_p = c.ptr<float>(1);
    float* Ld_row = Ld.ptr<float>(0);
    float* Ld_row_p = Ld.ptr<float>(1);
    float* Lstep_row = Lstep.ptr<float>(0);

    for (int x = 1; x < Lstep.cols-1; x++) {
        float xpos = (c_row[x]+c_row[x+1])*(Ld_row[x+1]-Ld_row[x]);
        float xneg = (c_row[x-1]+c_row[x])*(Ld_row[x]-Ld_row[x-1]);
        float ypos = (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
        Lstep_row[x] = 0.5*stepsize*(xpos-xneg + ypos);
    }

    float xpos = (c_row[0]+c_row[1])*(Ld_row[1]-Ld_row[0]);
    float ypos = (c_row[0]+c_row_p[0])*(Ld_row_p[0]-Ld_row[0]);
    Lstep_row[0] = 0.5*stepsize*(xpos + ypos);

    int x = Lstep.cols-1;
    float xneg = (c_row[x-1]+c_row[x])*(Ld_row[x]-Ld_row[x-1]);
    ypos = (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
    Lstep_row[x] = 0.5*stepsize*(-xneg + ypos);

    // Last row
    c_row = c.ptr<float>(Lstep.rows-1);
    c_row_p = c.ptr<float>(Lstep.rows-2);
    Ld_row = Ld.ptr<float>(Lstep.rows-1);
    Ld_row_p = Ld.ptr<float>(Lstep.rows-2);
    Lstep_row = Lstep.ptr<float>(Lstep.rows-1);

    for (int x = 1; x < Lstep.cols-1; x++) {
        float xpos = (c_row[x]+c_row[x+1])*(Ld_row[x+1]-Ld_row[x]);
        float xneg = (c_row[x-1]+c_row[x])*(Ld_row[x]-Ld_row[x-1]);
        float ypos   = (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
        Lstep_row[x] = 0.5*stepsize*(xpos-xneg + ypos);
    }

    xpos = (c_row[0]+c_row[1])*(Ld_row[1]-Ld_row[0]);
    ypos = (c_row[0]+c_row_p[0])*(Ld_row_p[0]-Ld_row[0]);
    Lstep_row[0] = 0.5*stepsize*(xpos + ypos);

    x = Lstep.cols-1;
    xneg = (c_row[x-1]+c_row[x])*(Ld_row[x]-Ld_row[x-1]);
    ypos = (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
    Lstep_row[x] = 0.5*stepsize*(-xneg + ypos);

    // First and last columns
    for (int i = 1; i < Lstep.rows-1; i++) {

        const float* c_row = c.ptr<float>(i);
        const float* c_row_m = c.ptr<float>(i-1);
        const float* c_row_p = c.ptr<float>(i+1);
        float* Ld_row = Ld.ptr<float>(i);
        float* Ld_row_p = Ld.ptr<float>(i+1);
        float* Ld_row_m = Ld.ptr<float>(i-1);
        Lstep_row = Lstep.ptr<float>(i);

        float xpos = (c_row[0]+c_row[1])*(Ld_row[1]-Ld_row[0]);
        float ypos = (c_row[0]+c_row_p[0])*(Ld_row_p[0]-Ld_row[0]);
        float yneg = (c_row_m[0]+c_row[0])*(Ld_row[0]-Ld_row_m[0]);
        Lstep_row[0] = 0.5*stepsize*(xpos+ypos-yneg);

        float xneg = (c_row[Lstep.cols-2]+c_row[Lstep.cols-1])*(Ld_row[Lstep.cols-1]-Ld_row[Lstep.cols-2]);
        ypos = (c_row[Lstep.cols-1]+c_row_p[Lstep.cols-1])*(Ld_row_p[Lstep.cols-1]-Ld_row[Lstep.cols-1]);
        yneg = (c_row_m[Lstep.cols-1]+c_row[Lstep.cols-1])*(Ld_row[Lstep.cols-1]-Ld_row_m[Lstep.cols-1]);
        Lstep_row[Lstep.cols-1] = 0.5*stepsize*(-xneg+ypos-yneg);
    }

    // Ld = Ld + Lstep
    for (int y = 0; y < Lstep.rows; y++) {
        float* Ld_row = Ld.ptr<float>(y);
        float* Lstep_row = Lstep.ptr<float>(y);
        for (int x = 0; x < Lstep.cols; x++) {
            Ld_row[x] = Ld_row[x] + Lstep_row[x];
        }
    }

    */
}



void PM_G2Cuda( const float* lx, const float* ly, float* dst, int width, int height, float k_percent  )
{

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    float inv_k = 1.0 / (k_percent*k_percent);
    PM_G2Cuda<<<numBlocks, threadsPerBlock>>>(lx, ly , dst, inv_k);

}

void NLDStepScalarCUDA(const float* src, const float* stLd, float* dlt, float stepSize, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    NLDStepScalarCuda<<<numBlocks, threadsPerBlock>>>(src, stLd , dlt, 0.5f*stepSize,width );
}