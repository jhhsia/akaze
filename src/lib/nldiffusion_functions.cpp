//=============================================================================
//
// nldiffusion_functions.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file nldiffusion_functions.cpp
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "nldiffusion_functions.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../gpu/helper_cuda.h"
#include "../gpu/ImageOperation.h"
#include "../gpu/helper_image.h"
using namespace std;

inline int fRound(float flt) {
    return (int)(flt+0.5f);
}
#define NO_CUDA 1

#if NO_CUDA
    #define SHARR_CUDA 0
    #define SHARR_DER_CUDA 0
    #define HASSIAN_CUDA 0
    #define NLD_CUDA 0
    #define HALF_SIZE_CUDA 0
    #define GAUSSIAN_CUDA 0

#else
    #define SHARR_CUDA 1
    #define SHARR_DER_CUDA 1
    #define HASSIAN_CUDA 1
    #define NLD_CUDA 1
    #define HALF_SIZE_CUDA 1
    #define GAUSSIAN_CUDA 1
#endif
/* ************************************************************************* */
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, size_t ksize_x,
                             size_t ksize_y, float sigma) {

  // Compute an appropriate kernel size according to the specified sigma
  if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
    ksize_x = ceil(2.0*(1.0 + (sigma-0.8)/(0.3)));
    ksize_y = ksize_x;
  }

  // The kernel size must be and odd number
  if ((ksize_x % 2) == 0)
    ksize_x += 1;

  if ((ksize_y % 2) == 0)
    ksize_y += 1;

#if 0
    cv::Mat temp = cv::Mat( src.rows, src.cols, CV_32F, 1.0f );

        float* a_row = temp.ptr<float>(0);
    for(int i = 0; i < src.cols; ++i)
    {
        //a_row[i] = 0;
    }

#endif

#if (!GAUSSIAN_CUDA)
    double t1 = 0.0, t2 = 0.0;
  // Perform the Gaussian Smoothing with border replication
  t1 = cv::getTickCount();
    cv::GaussianBlur(src, dst, cv::Size(ksize_x, ksize_y), sigma, sigma, cv::BORDER_REPLICATE);
     t2 = cv::getTickCount();
    float time = 1000.0*(t2-t1) / cv::getTickFrequency();
    printf("-- gaussian_2D_convolution %.4f ksize  %d\n",time,ksize_x );
#else

    //cv::Mat temp = cv::Mat( src.rows, src.cols, CV_32F, 1.0f );
    //cv::GaussianBlur(src, temp, cv::Size(ksize_x, ksize_y), sigma, sigma, cv::BORDER_REPLICATE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /// cudaaaaa
    {
        float* d_src;
        float* d_dst;
        float* d_intem;

        size_t img_size = src.cols*src.rows*sizeof(float);
        cudaMalloc(&d_src, img_size);
        cudaMalloc(&d_dst, img_size);
        cudaMalloc(&d_intem, img_size);
        //float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(d_src, src.data, img_size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        if(ksize_y == 9)
        {
            GaussianCUDA9x9TwoPass(d_src, d_intem, d_dst, sigma, src.cols, src.rows );
        }
        else
        {
           // GaussianCUDA5x5(d_src, d_dst, sigma, src.cols, src.rows );
            GaussianCUDA5x5TwoPass(d_src, d_intem, d_dst, sigma, src.cols, src.rows );
        }
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        if(dst.data == NULL)
        {
            dst = cv::Mat( src.rows, src.cols, CV_32F);
        }

        cudaMemcpy(dst.data, d_dst, img_size, cudaMemcpyDeviceToHost);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
#if 0

        cudaMemcpy(client_dst, d_dst, img_size, cudaMemcpyDeviceToHost);
        //temp.copyTo(dst);
        //

        for(int i = 0; i < src.rows; ++i)
        {
            const float* src_x_u  = temp.ptr<float>(i);
            int start_offset = i*src.cols;
            for(int j = 6; j < 7; ++j)
            {
                const float* cuda_s   = client_dst+start_offset;
                float diff = src_x_u[j]-cuda_s[j];
                if(diff > 0.000001f)
                {
                    printf("-- %d , %d \n", i, j);
                    printf("-- %.5f/%.5f diff %.6f\n",src_x_u[j], cuda_s[j], src_x_u[j]-cuda_s[j] );
                }
            }
        }

        printf(" ksize = %d/%d\n" , ksize_y, ksize_x);
#endif
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_intem);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
        printf("-- gaussian_2D_convolution %.4f ksize  %d\n",milliseconds,ksize_x );

    }
#endif



}

/* ************************************************************************* */
void image_derivatives_scharr_jay(const cv::Mat& src, cv::Mat& dst,
                                  const size_t xorder, const size_t yorder);

void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst,
                              const size_t xorder, const size_t yorder) {

#if (!SHARR_CUDA)
    image_derivatives_scharr_jay(src, dst, xorder,yorder  );

    // CUDAAAAA
#else

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float* d_src;
        float* d_dst;

        size_t img_size = dst.cols*dst.rows*sizeof(float);
        cudaMalloc(&d_src, img_size);
        cudaMalloc(&d_dst, img_size);
       // float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(d_src, src.data, img_size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);

        ScharrFilterCuda(d_src, d_dst, src.cols, src.rows, yorder);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);

        cudaMemcpy(dst.data, d_dst, img_size, cudaMemcpyDeviceToHost);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);



        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
        printf("-- derivatives_scharr %.6f \n",milliseconds );

#if 0
        cudaMemcpy(client_dst, d_dst, img_size, cudaMemcpyDeviceToHost);
    //
      int start_offset = 64*src.cols;
        int diffcout = 0;
    for(int i = 0; i < src.cols; ++i)
    {
        const float* src_x_u  = dst.ptr<float>(64);
        const float* cuda_s   = client_dst+start_offset;
        float diff = src_x_u[i]-cuda_s[i];
        if(diff != 0)
        {
            ++diffcout;
            printf("-- %.2f/%.2f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
        }

    }

       // printf("-- %.2f/%.2f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );


      //
#endif
        cudaFree(d_src);
        cudaFree(d_dst);
  }
#endif

}


void image_derivatives_scharr_jay(const cv::Mat& src, cv::Mat& dst,
                              const size_t xorder, const size_t yorder) {


    int width = src.cols;
    int height = src.rows;
    if( xorder == 1)
    {
        for(int j = 1; j < height -1; ++j)
        {
            const float* src_x_u  = src.ptr<float>(j-1);
            const float* src_x    = src.ptr<float>(j);
            const float* src_x_d  = src.ptr<float>(j+1);

            float* dst_x = dst.ptr<float>(j);
            for(int i = 1 ; i < width -1; ++i )
            {
                float left = (src_x_u[i-1] + src_x_d[i-1]) * -3.0f + -10.0f * src_x[i-1];
                float right = (src_x_u[i+1] + src_x_d[i+1]) * 3.0f + 10.0f * src_x[i+1];
                dst_x[i] = left+right;
            }
        }
    }
    else
    {
        for(int j = 1; j < height -1; ++j)
        {
            const float* src_x_u  = src.ptr<float>(j-1);
            //const float* src_x    = src.ptr<float>(j);
            const float* src_x_d  = src.ptr<float>(j+1);

            float* dst_x = dst.ptr<float>(j);
            for(int i = 1 ; i < width -1; ++i )
            {
                float up = (src_x_u[i-1] + src_x_u[i+1]) * -3.0f + -10.0f * src_x_u[i];
                float down = (src_x_d[i-1] + src_x_d[i+1]) * 3.0f + 10.0f * src_x_d[i];
                dst_x[i] = up+down;
            }
        }
    }
}

/* ************************************************************************* */
void image_derivatives_sobel(const cv::Mat& src, cv::Mat& dst,
                              const size_t xorder, const size_t yorder) {
  //cv::Scharr(src, dst, CV_32F, xorder, yorder, 1.0, 0, cv::BORDER_DEFAULT);

  cv::Sobel( src, dst, CV_32F, xorder, yorder, 3, 1.0, 0, cv::BORDER_DEFAULT );
}


/* ************************************************************************* */
void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float k) {

  cv::Size sz = Lx.size();
  float inv_k = 1.0 / (k*k);
  for (int y = 0; y < sz.height; y++) {
    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);
    float* dst_row = dst.ptr<float>(y);
    for (int x = 0; x < sz.width; x++)
      dst_row[x] = (-inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]));
  }

  cv::exp(dst, dst);
}

/* ************************************************************************* */
void pm_g2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float k) {
#if NO_CUDA
  cv::Size sz = Lx.size();
  float inv_k = 1.0 / (k*k);
  for (int y = 0; y < sz.height; y++) {
    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);
    float* dst_row = dst.ptr<float>(y);
    for (int x = 0; x < sz.width; x++)
      dst_row[x] = 1.0 / (1.0+inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]));
  }
#else
    {
        float* s_lx;
        float* s_ly;
        float* d_flow;

        size_t img_size = dst.cols*dst.rows*sizeof(float);
        cudaMalloc(&s_lx, img_size);
        cudaMalloc(&s_ly, img_size);
        cudaMalloc(&d_flow, img_size);

        //float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(s_lx, Lx.data, img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(s_ly, Ly.data, img_size, cudaMemcpyHostToDevice);

        PM_G2Cuda( s_lx, s_ly, d_flow, dst.cols, dst.rows, k );
        cudaMemcpy(dst.data, d_flow, img_size, cudaMemcpyDeviceToHost);

#if 0
        int start_offset = 128*dst.cols;

        for(int i = 0; i < dst.cols; ++i)
        {
            const float* src_x_u  = dst.ptr<float>(128);
            const float* cuda_s   = client_dst+start_offset;
            printf("-- %.5f/%.5f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
        }
#endif
        cudaFree(s_lx);
        cudaFree(s_ly);
        cudaFree(d_flow);

    }
#endif
}

/* ************************************************************************* */
void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float k) {

  cv::Size sz = Lx.size();
  float inv_k = 1.0 / (k*k);
  for (int y = 0; y < sz.height; y++) {
    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);
    float* dst_row = dst.ptr<float>(y);
    for (int x = 0; x < sz.width; x++) {
      float dL = inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]);
      dst_row[x] = -3.315/(dL*dL*dL*dL);
    }
  }

  cv::exp(dst, dst);
  dst = 1.0 - dst;
}

/* ************************************************************************* */
void charbonnier_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float k) {

  cv::Size sz = Lx.size();
  float inv_k = 1.0 / (k*k);
  for (int y = 0; y < sz.height; y++) {
    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);
    float* dst_row = dst.ptr<float>(y);
    for (int x = 0; x < sz.width; x++) {
      float den = sqrt(1.0+inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]));
      dst_row[x] = 1.0 / den;
    }
  }
}

/* ************************************************************************* */
float compute_k_percentile(const cv::Mat& img, float perc, float gscale,
                           size_t nbins, size_t ksize_x, size_t ksize_y) {

  size_t nbin = 0, nelements = 0, nthreshold = 0, k = 0;
  float kperc = 0.0, modg = 0.0, npoints = 0.0, hmax = 0.0;

  // Create the array for the histogram
  float* hist = new float[nbins];

  // Create the matrices
  cv::Mat gaussian = cv::Mat::zeros(img.rows, img.cols, CV_32F);

  cv::Mat Lx = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat Ly = cv::Mat::zeros(img.rows, img.cols, CV_32F);

  // Set the histogram to zero
  for (size_t i = 0; i < nbins; i++)
    hist[i] = 0.0;

  // Perform the Gaussian convolution
  gaussian_2D_convolution(img, gaussian, ksize_x, ksize_y, gscale);

  // Compute the Gaussian derivatives Lx and Ly

    image_derivatives_scharr( gaussian, Lx, 1, 0);
    image_derivatives_scharr( gaussian, Ly, 0, 1);


  // Skip the borders for computing the histogram
  for (int y = 1; y < gaussian.rows-1; y++) {

    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);

    for (int x = 1; x < gaussian.cols-1; x++) {

      modg = sqrt(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]);

      // Get the maximum
      if (modg > hmax)
        hmax = modg;
    }
  }

  // Skip the borders for computing the histogram
  for (int y = 1; y < gaussian.rows-1; y++) {

    const float* Lx_row = Lx.ptr<float>(y);
    const float* Ly_row = Ly.ptr<float>(y);

    for (int x = 1; x < gaussian.cols-1; x++) {

      modg = sqrt(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]);

      // Find the correspondent bin
      if (modg != 0.0) {
        nbin = floor(nbins*(modg/hmax));

        if (nbin == nbins) {
          nbin--;
        }

        hist[nbin]++;
        npoints++;
      }
    }
  }

  // Now find the perc of the histogram percentile
  nthreshold = (size_t)(npoints*perc);

  for (k = 0; nelements < nthreshold && k < nbins; k++)
    nelements = nelements + hist[k];

  if (nelements < nthreshold)
    kperc = 0.03;
  else
    kperc = hmax*((float)(k)/(float)nbins);

  delete [] hist;
  return kperc;
}

/* ************************************************************************* */
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t xorder,
                                const size_t yorder, const size_t scale) {

  cv::Mat kx, ky;
  compute_derivative_kernels(kx, ky, xorder, yorder, scale);
#if (!SHARR_DER_CUDA)

#if 0
    cv::Mat temp = cv::Mat( src.rows, src.cols, CV_32F, 1.0f );

        float* a_row = temp.ptr<float>(0);
    for(int i = 0; i < src.cols; ++i)
    {
        a_row[i] = 0;
    }

    ky.ptr<float>(0)[0 ] = 0.01f;
    ky.ptr<float>(0)[kx.rows/2 ] = 0.1f;
    ky.ptr<float>(0)[kx.rows-1 ] = 0.01f;

    kx.ptr<float>(0)[0 ] = 1;
    kx.ptr<float>(0)[1 ] = 0;
    kx.ptr<float>(0)[2] =0 ;
    kx.ptr<float>(0)[3 ] = 0;
    kx.ptr<float>(0)[4 ] = 1;
#endif
  cv::sepFilter2D(src, dst, CV_32F, kx, ky);
#else
    //TODO this is 20+ms slower than sepFilter2D!!

    {

     //   cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);



        float* s_src;
        float* d_dst;

        size_t img_size = src.cols*src.rows*sizeof(float);
        cudaMalloc(&s_src, img_size);
        cudaMalloc(&d_dst, img_size);

        //float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(s_src, src.data, img_size, cudaMemcpyHostToDevice);

        int hsize = kx.rows ;

        cv::Mat& ref =  yorder == 1 ? kx : ky;

        float mk = ref.ptr<float>(0)[hsize/2 ];
        float ek = ref.ptr<float>(0)[0];

     //   cudaEventRecord(start);

        SharrDerivativeCUDA( s_src, d_dst, mk, ek,yorder,hsize, src.cols, src.rows );

     //   cudaDeviceSynchronize();
     //   cudaEventRecord(stop);

        cudaMemcpy(dst.data, d_dst, img_size, cudaMemcpyDeviceToHost);


    //    float milliseconds = 0;
    //    cudaEventElapsedTime(&milliseconds, start, stop);
   //     cudaEventDestroy(start);
   //     cudaEventDestroy(stop);
        //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
   //     printf("-- compute_scharr_derivatives %.4f \n",milliseconds );

       // cudaMemcpy(client_dst, d_dst, img_size, cudaMemcpyDeviceToHost);
#if 0
        //
        int start_offset = 2*dst.cols;

        for(int i = 0; i < dst.cols; ++i)
        {
            const float* src_x_u  = dst.ptr<float>(2);
            const float* cuda_s   = client_dst+start_offset;
            printf("-- %.5f/%.5f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
        }
#endif

        cudaFree(s_src);
        cudaFree(d_dst);
    }
#endif
}

void cal_determinant_hessian(std::vector<TEvolution>& evo, bool verb )
{
    for (size_t i = 0; i < evo.size(); i++) {
        if (verb == true)
            cout << "Computing detector response. Determinant of Hessian. Evolution time: " << evo[i].etime << endl;

#if (!HASSIAN_CUDA)

        float sigma_size_ = evo[i].multiDerSigmaSize;
        float sigma_size_sqr = sigma_size_*sigma_size_;
        float sigma_size_quad = sigma_size_sqr*sigma_size_sqr;
 //       evo[i].Lx = evo[i].Lx*((sigma_size_));
  //      evo[i].Ly = evo[i].Ly*((sigma_size_));
 //       evo[i].Lxx = evo[i].Lxx*(sigma_size_sqr);
 //       evo[i].Lxy = evo[i].Lxy*(sigma_size_sqr);
 //       evo[i].Lyy = evo[i].Lyy*(sigma_size_sqr);

        int height = evo[i].Ldet.rows;
        int width = evo[i].Ldet.cols;
        for (int ix = 0; ix < height; ix++) {
            const float* lxx = evo[i].Lxx.ptr<float>(ix);
            const float* lxy = evo[i].Lxy.ptr<float>(ix);
            const float* lyy = evo[i].Lyy.ptr<float>(ix);
            float* ldet = evo[i].Ldet.ptr<float>(ix);
            for (int jx = 0; jx < width; jx++)
                ldet[jx] = ( lxx[jx]*sigma_size_sqr*lyy[jx]*sigma_size_sqr-lxy[jx]*sigma_size_sqr*lxy[jx]*sigma_size_sqr);
        }

#else
        {

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            float* s_lxx;
            float* s_lyy;
            float* s_lxy;
            float* ldet;

            size_t img_size = evo[i].Lxx.cols*evo[i].Lxx.rows*sizeof(float);
            cudaMalloc(&s_lxx, img_size);
            cudaMalloc(&s_lyy, img_size);
            cudaMalloc(&s_lxy, img_size);
            cudaMalloc(&ldet, img_size);

         //   float* client_dst = (float*)malloc( img_size);

            cudaMemcpy(s_lxx, evo[i].Lxx.data, img_size, cudaMemcpyHostToDevice);
            cudaMemcpy(s_lyy, evo[i].Lyy.data, img_size, cudaMemcpyHostToDevice);
            cudaMemcpy(s_lxy, evo[i].Lxy.data, img_size, cudaMemcpyHostToDevice);

            float sigma_size_ = evo[i].multiDerSigmaSize;
            float sigma_size_sqr = sigma_size_*sigma_size_;
            float sigma_size_quad = sigma_size_sqr*sigma_size_sqr;


            cudaEventRecord(start);

            DeterminantHessianCUDA( s_lxx, s_lyy, s_lxy, ldet, sigma_size_quad, evo[i].Lxx.cols, evo[i].Lxx.rows);

            cudaDeviceSynchronize();
            cudaEventRecord(stop);

            cudaMemcpy(evo[i].Ldet.data, ldet, img_size, cudaMemcpyDeviceToHost);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
            printf("-- cal_determinant_hessian %.4f \n",milliseconds );

    #if 0
            cudaMemcpy(client_dst, ldet, img_size, cudaMemcpyDeviceToHost);
            //
            int width =  evo[i].Ldet.cols;
            int start_offset = 16* width;
            const float* src_x_u  = evo[i].Ldet.ptr<float>(16);
            for(int i = 0; i < width; ++i)
            {

                const float* cuda_s   = client_dst+start_offset;
                printf("-- %.5f/%.5f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
            }
    #endif

            cudaFree(s_lxx);
            cudaFree(s_lxy);
            cudaFree(s_lyy);
            cudaFree(ldet);
        }
#endif
    }
}

/* ************************************************************************* */
void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, const float stepsize) {

#if (!NLD_CUDA)
  // Diffusion all the image except borders
#ifdef _OPENMP_NO
  omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for schedule(dynamic)
#endif

  for (int y = 1; y < Lstep.rows-1; y++) {
    const float* c_row = c.ptr<float>(y);
    const float* c_row_p = c.ptr<float>(y+1);
    const float* c_row_m = c.ptr<float>(y-1);

    float* Ld_row = Ld.ptr<float>(y);
    float* Ld_row_p = Ld.ptr<float>(y+1);
    float* Ld_row_m = Ld.ptr<float>(y-1);
    float* Lstep_row = Lstep.ptr<float>(y);

    for (int x = 1; x < Lstep.cols-1; x++) {
      float xpos =  (c_row[x]+c_row[x+1])*(Ld_row[x+1]-Ld_row[x]);
      float xneg =  (c_row[x-1]+c_row[x])*(Ld_row[x]-Ld_row[x-1]);
      float ypos =  (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
      float yneg =  (c_row_m[x]+c_row[x])*(Ld_row[x]-Ld_row_m[x]);
      Lstep_row[x] = 0.5*stepsize*(xpos-xneg + ypos-yneg);
    }
  }


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
    float ypos = (c_row[x]+c_row_p[x])*(Ld_row_p[x]-Ld_row[x]);
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
#else
    {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float* s_flow;
        float* s_lt;
        float* d_lt;

        size_t img_size = Lstep.cols*Lstep.rows*sizeof(float);
        cudaMalloc(&s_flow, img_size);
        cudaMalloc(&s_lt, img_size);
        cudaMalloc(&d_lt, img_size);

        //float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(s_flow, c.data, img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(s_lt, Ld.data, img_size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);

        NLDStepScalarCUDA( s_flow, s_lt, d_lt, stepsize, Lstep.cols, Lstep.rows );

        cudaDeviceSynchronize();
        cudaEventRecord(stop);

        cudaMemcpy(Ld.data, d_lt, img_size, cudaMemcpyDeviceToHost);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
        printf("-- nld_step_scalar  %.5f \n",milliseconds );


      //  cudaMemcpy(client_dst, d_lt, img_size, cudaMemcpyDeviceToHost);
#if 0
        // Ld = Ld + Lstep
        for (int y = 0; y < Lstep.rows; y++) {
            float* Ld_row = Ld.ptr<float>(y);
            float* Lstep_row = Lstep.ptr<float>(y);
            for (int x = 0; x < Lstep.cols; x++) {
                Ld_row[x] = Ld_row[x] + Lstep_row[x];
            }
        }

        //
        int start_offset = 128*Lstep.cols;

        for(int i = 0; i < Lstep.cols; ++i)
        {
            const float* src_x_u  = Ld.ptr<float>(128);
            const float* cuda_s   = client_dst+start_offset;
            printf("-- %.5f/%.5f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
        }
#endif

        cudaFree(s_flow);
        cudaFree(s_lt);
        cudaFree(d_lt);
    }
#endif
}

/* ************************************************************************* */
void halfsample_image(const cv::Mat& src, cv::Mat& dst) {

#if (!HALF_SIZE_CUDA)
  // Make sure the destination image is of the right size
  cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_AREA);
#else
    {
        float* s_src;
        float* d_dst;

        size_t img_size = dst.cols*dst.rows*sizeof(float);

        cudaMalloc(&s_src, img_size*4);
        cudaMalloc(&d_dst, img_size);

       // float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(s_src, src.data, img_size*4, cudaMemcpyHostToDevice);

        DownSize2x2AreaFastCUDA( s_src, d_dst,  dst.cols, dst.rows);

        cudaMemcpy(dst.data, d_dst, img_size, cudaMemcpyDeviceToHost);
       // cudaMemcpy(client_dst, d_dst, img_size, cudaMemcpyDeviceToHost);
#if 0
        //
        int start_offset = 128*dst.cols;

        for(int i = 0; i < dst.cols; ++i)
        {
            const float* src_x_u  = dst.ptr<float>(128);
            const float* cuda_s   = client_dst+start_offset;
            printf("-- %.5f/%.5f diff %.4f\n",src_x_u[i], cuda_s[i], src_x_u[i]-cuda_s[i] );
        }
#endif

        cudaFree(s_src);
        cudaFree(d_dst);
    }
#endif
}

/* ************************************************************************* */
void compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_,
                                const size_t dx, const size_t dy, const size_t scale) {

  const int ksize = 3 + 2*(scale-1);

  // The usual Scharr kernel
  if (scale == 1) {
    cv::getDerivKernels(kx_, ky_, dx, dy, 0, true, CV_32F);
    return;
  }

  kx_.create(ksize,1,CV_32F,-1,true);
  ky_.create(ksize,1,CV_32F,-1,true);
  cv::Mat kx = kx_.getMat();
  cv::Mat ky = ky_.getMat();

  float w = 10.0/3.0;
  float norm = 1.0/(2.0*scale*(w+2.0));

  for (int k = 0; k < 2; k++) {
    cv::Mat* kernel = k == 0 ? &kx : &ky;
    int order = k == 0 ? dx : dy;
    float kerI[1000];

    for (int t = 0; t<ksize; t++)
      kerI[t] = 0;

    if (order == 0) {
      kerI[0] = norm;
      kerI[ksize/2] = w*norm;
      kerI[ksize-1] = norm;
    }
    else if (order == 1) {
      kerI[0] = -1;
      kerI[ksize/2] = 0;
      kerI[ksize-1] = 1;
    }

    cv::Mat temp(kernel->rows, kernel->cols, CV_32F, &kerI[0]);
    temp.copyTo(*kernel);
  }
}

/* ************************************************************************* */
bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value,
                                 int row, int col, bool same_img) {

  bool response = true;

  for (int i = row-dsize; i <= row+dsize; i++) {
    for (int j = col-dsize; j <= col+dsize; j++) {
      if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) {
        if (same_img == true) {
          if (i != row || j != col) {
            if ((*(img.ptr<float>(i)+j)) > value) {
              response = false;
              return response;
            }
          }
        }
        else {
          if ((*(img.ptr<float>(i)+j)) > value) {
            response = false;
            return response;
          }
        }
      }
    }
  }

  return response;
}



void local_extrema(AKAZEOptions& options, std::vector<TEvolution>& evolution)
{

    double t1 = 0.0, t2 = 0.0;
    float value = 0.0;
    float dist = 0.0, ratio = 0.0, smax = 0.0;
    int npoints = 0, id_repeated = 0;
    int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
    bool is_extremum = false, is_repeated = false, is_out = false;
    cv::KeyPoint point;
    vector<cv::KeyPoint> kpts_aux;

    // Set maximum size
    if (options.descriptor == SURF_UPRIGHT || options.descriptor == SURF ||
        options.descriptor == MLDB_UPRIGHT || options.descriptor == MLDB) {
        smax = 10.0*sqrtf(2.0f);
    }
    else if (options.descriptor == MSURF_UPRIGHT || options.descriptor == MSURF) {
        smax = 12.0*sqrtf(2.0f);
    }

    t1 = cv::getTickCount();

    float dthresh = options.dthreshold;

    for (size_t i = 0; i < evolution.size(); i++) {

        float* d_src;
        float* d_dst;
        float* d_intem;

        float kp_size = evolution[i].esigma*options.derivative_factor;

        float ratio = pow(2.0f, evolution[i].octave);
        float sigma_size_ = fRound(kp_size/ratio);

        float smax_sigma = smax*sigma_size_;

        printf("--  smax*sigma_size at %.2f \n",kp_size);

        const float* src_read = (const float*)evolution[i].Ldet.data;
#if 1
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t img_size =  evolution[i].Ldet.rows * evolution[i].Ldet.cols*sizeof(float);
        int code = cudaMalloc(&d_src, img_size);
        code = cudaMalloc(&d_intem, img_size);
        code = cudaMalloc(&d_dst, img_size);

        float* client_dst = (float*)malloc( img_size);

        cudaMemcpy(d_src, evolution[i].Ldet.data , img_size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);

        KeyPointExtract( d_src , d_intem, d_dst, kp_size,
                          smax_sigma, dthresh , ratio, evolution[i].Ldet.cols, evolution[i].Ldet.rows);

        cudaDeviceSynchronize();

        cudaEventRecord(stop);

        cudaMemcpy(client_dst, d_dst, img_size, cudaMemcpyDeviceToHost);


        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //float time = 1000.0*(t2-t1) / cv::getTickFrequency();
        printf("-- local_extrema %.4f \n",milliseconds  );



        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_intem);
#endif
        for (int ix = 1; ix < evolution[i].Ldet.rows-1; ix++) {

            float* ldet_m = evolution[i].Ldet.ptr<float>(ix-1);
            float* ldet = evolution[i].Ldet.ptr<float>(ix);
            float* ldet_p = evolution[i].Ldet.ptr<float>(ix+1);

            for (int jx = 1; jx < evolution[i].Ldet.cols-1; jx++) {

                is_extremum = false;
                is_repeated = false;
                is_out = false;
                value = ldet[jx];

                //printf("-- 0000 at %.2f \n", ldet[jx]);

                // Filter the points with the detector threshold
                if (value > options.dthreshold && value >= options.min_dthreshold &&
                    value > ldet[jx-1] && value > ldet[jx+1] &&
                    value > ldet_m[jx-1] && value > ldet_m[jx] && value > ldet_m[jx+1] &&
                    value > ldet_p[jx-1] && value > ldet_p[jx] && value > ldet_p[jx+1]) {

                    is_extremum = true;
                    point.response = fabs(value);
                    point.size = evolution[i].esigma*options.derivative_factor;
                    point.octave = evolution[i].octave;
                    point.class_id = i;
                    ratio = pow(2.0f, point.octave);
                    sigma_size_ = fRound(point.size/ratio);
                    point.pt.x = jx;
                    point.pt.y = ix;

                    // Compare response with the same and lower scale
                    for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

                   //     if ((point.class_id-1) == kpts_aux[ik].class_id ||
                        if ( point.class_id == kpts_aux[ik].class_id) {

                            dist = (point.pt.x*ratio-kpts_aux[ik].pt.x)*(point.pt.x*ratio-kpts_aux[ik].pt.x) +
                                   (point.pt.y*ratio-kpts_aux[ik].pt.y)*(point.pt.y*ratio-kpts_aux[ik].pt.y);

                            if (dist <= point.size*point.size) {
                                if (point.response > kpts_aux[ik].response) {
                                    id_repeated = ik;
                                    is_repeated = true;
                                }
                                else {
                                    is_extremum = false;
                                }
                                break;
                            }
                        }
                    }

                    // Check out of bounds
                    if (is_extremum == true) {

                        // Check that the point is under the image limits for the descriptor computation
                        left_x = fRound(point.pt.x-smax*sigma_size_)-1;
                        right_x = fRound(point.pt.x+smax*sigma_size_) +1;
                        up_y = fRound(point.pt.y-smax*sigma_size_)-1;
                        down_y = fRound(point.pt.y+smax*sigma_size_)+1;

                        if (left_x < 0 || right_x >= evolution[i].Ldet.cols ||
                            up_y < 0 || down_y >= evolution[i].Ldet.rows) {
                            is_out = true;
                        }

                        int idx = jx + ix*evolution[i].Ldet.cols;

                        if (is_out == false) {

                            float value = client_dst[idx];

                            if (is_repeated == false) {

                                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                kpts_aux.push_back(point);
                                npoints++;

                               // PrintSurroundPixels(jx, ix, client_dst, evolution[i].Ldet.cols );
                               // printf("-- %.5f/%.5f diff %.4f\n",point.response, value, point.response-value );
                            }
                            else {
                                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                kpts_aux[id_repeated] = point;

                            //    printf("is_repeated -- %.5f/%.5f diff %.4f\n",point.response, value, point.response-value );
                            }
                        } // if is_out
                    } //if is_extremum
                }
            } // for jx
        } // for ix
        free(client_dst);
    } // for i
}