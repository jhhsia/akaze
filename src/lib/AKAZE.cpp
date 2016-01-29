    //=============================================================================
    //
    // AKAZE.cpp
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
    * @file AKAZE.cpp
    * @brief Main class for detecting and describing binary features in an
    * accelerated nonlinear scale space
    * @date Oct 07, 2014
    * @author Pablo F. Alcantarilla, Jesus Nuevo
    */

    #include "AKAZE.h"
    #include <stdio.h>
    #include <opencv2/highgui/highgui.hpp>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "../gpu/helper_cuda.h"
    #include "../gpu/ImageOperation.h"

    using namespace std;
    using namespace libAKAZE;

    /* ************************************************************************* */
    AKAZE::AKAZE(const AKAZEOptions& options) : options_(options) {

        ncycles_ = 0;
        reordering_ = true;

        if (options_.descriptor_size > 0 && options_.descriptor >= MLDB_UPRIGHT) {
            generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options_.descriptor_size,
                                options_.descriptor_pattern_size, options_.descriptor_channels);
        }

        Allocate_Memory_Evolution();
    }

    /* ************************************************************************* */
    AKAZE::~AKAZE() {
        evolution_.clear();
    }

    /* ************************************************************************* */
    void AKAZE::Allocate_Memory_Evolution() {

        float rfactor = 0.0;
        int level_height = 0, level_width = 0;

        // Allocate the dimension of the matrices for the evolution
        for (int i = 0; i <= options_.omax-1; i++) {
            rfactor = 1.0/pow(2.0f, i);
            level_height = (int)(options_.img_height*rfactor);
            level_width = (int)(options_.img_width*rfactor);

        // Smallest possible octave and allow one scale if the image is small
            if ((level_width < 80 || level_height < 40) && i != 0) {
                options_.omax = i;
                break;
            }

            for (int j = 0; j < options_.nsublevels; j++) {
                TEvolution step;
                cv::Size size(level_width, level_height);
                step.imgWidth = level_width;
                step.imgHeight = level_height;


                int line_size = level_width*sizeof(float);
                int total_cuda_mem_size = (line_size*level_height)*10;

                int img_pix_count = level_width*level_height;

                step.imgSize = line_size*level_height ;
                step.ltBufIdx = 0;
#if 0
                float* lx_mem ;
                float* ly_mem ;
                cudaHostAlloc( (void**)&lx_mem, step.imgSize+128, cudaHostAllocDefault  |
                                                              cudaHostAllocMapped );
                cudaHostAlloc( (void**)&ly_mem, step.imgSize+128, cudaHostAllocDefault  |
                                                              cudaHostAllocMapped );
                step.Lx = cv::Mat(level_height, level_width, CV_32F, lx_mem); //create(size, CV_32F);
                step.Ly = cv::Mat(level_height, level_width, CV_32F, ly_mem); //create(size, CV_32F);
#endif
                step.Lx.create(size, CV_32F);
                step.Ly.create(size, CV_32F);
#if GPU_MEM
                cudaMalloc(&step.memBase , total_cuda_mem_size);
                step.LxGpu       = step.memBase  + level_width * 6;
                step.LyGpu       = step.LxGpu    + img_pix_count;
                step.LxxGpu      = step.LyGpu    + img_pix_count;            ///< Second order spatial derivatives
                step.LxyGpu      = step.LxxGpu   + img_pix_count;
                step.LyyGpu      = step.LxyGpu   + img_pix_count;
                step.LflowGpu    = step.LyyGpu   + img_pix_count + level_width*4;
                step.LtGpu[0]    = step.LflowGpu + img_pix_count + level_width*4;
                step.LtGpu[1]    = step.LtGpu[0] + img_pix_count + level_width*4;
                step.LsmoothGpu  = step.LtGpu[1] + img_pix_count + level_width*4;
#else

                step.Lxx.create(size, CV_32F);
                step.Lxy.create(size, CV_32F);
                step.Lyy.create(size, CV_32F);
                step.Lflow.create(size, CV_32F);
                step.Lt.create(size, CV_32F);
#endif

                step.Ldet.create(size, CV_32F);

                step.Lstep.create(size, CV_32F);
                step.esigma = options_.soffset*pow(2.0f, (float)(j)/(float)(options_.nsublevels) + i);
                step.sigma_size = fRound(step.esigma);
                step.etime = 0.5*(step.esigma*step.esigma);
                step.octave = i;
                step.sublevel = j;
                evolution_.push_back(step);
            }
        }

        // Allocate memory for the number of cycles and time steps
        for (size_t i = 1; i < evolution_.size(); i++) {
            int naux = 0;
            vector<float> tau;
            float ttime = 0.0;
            ttime = evolution_[i].etime-evolution_[i-1].etime;
            naux = fed_tau_by_process_time(ttime, 1, 0.25, reordering_,tau);
            nsteps_.push_back(naux);
            tsteps_.push_back(tau);
            ncycles_++;
        }

        //alloc 512 mem ahead
        InitCudaMemory( 128*1024*0124 );
    }

    /* ************************************************************************* */
    int AKAZE::Create_Nonlinear_Scale_Space(const cv::Mat& img) {

        double t1 = 0.0, t2 = 0.0;

        if (evolution_.size() == 0) {
            cerr << "Error generating the nonlinear scale space!!" << endl;
            cerr << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()" << endl;
            return -1;
        }

        t1 = cv::getTickCount();

        // Copy the original image to the first level of the evolution
       // img.copyTo(evolution_[0].Lt);

        //gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
        //one shot blur.

        // Lsmooth ever used?
#if GPU_MEM
        evolution_[0].ltBufIdx = 1;
        gaussian_2D_convolutionGMEM(img, evolution_[0].LtGpu[0], 0, 0, options_.soffset);

        cudaMemcpy( evolution_[0].LsmoothGpu , evolution_[0].LtGpu[0], evolution_[0].imgSize, cudaMemcpyDeviceToDevice);

       // cudaMemcpy( evolution_[0].Lt.data , evolution_[0].LsmoothGpu, evolution_[0].imgSize, cudaMemcpyDeviceToHost);
#else
        img.copyTo(evolution_[0].Lt);
        //one shot blur.
        gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);

        evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);

#endif

        // First compute the kcontrast factor
        // with gscal 1.0 blur?

        options_.kcontrast = -1.0f;
        //  options_.kcontrast = compute_k_percentile(img, options_.kcontrast_percentile,
        //                                            1.0, options_.kcontrast_nbins, 0, 0);

        t2 = cv::getTickCount();
        timing_.kcontrast = 1000.0*(t2-t1) / cv::getTickFrequency();

        // Now generate the rest of evolution levels
        for (size_t i = 1; i < evolution_.size(); i++) {

            int src_front = 1-evolution_[i-1].ltBufIdx;
            int dst_front = 1-evolution_[i].ltBufIdx;

            int img_width = evolution_[i].imgWidth;
            int img_height = evolution_[i].imgHeight;

            if (evolution_[i].octave > evolution_[i-1].octave) {

                halfsample_image( evolution_[i-1].LtGpu[src_front], evolution_[i].LtGpu[dst_front],
                                 img_width,  img_height );

                options_.kcontrast = options_.kcontrast*0.75;
            }
            else {
#if GPU_MEM
                cudaMemcpy( evolution_[i].LtGpu[dst_front] , evolution_[i-1].LtGpu[src_front], evolution_[i].imgSize, cudaMemcpyDeviceToDevice);
#else
                evolution_[i-1].Lt.copyTo(evolution_[i].Lt);
#endif
            }

#if 0
            //copy lt buffer to front
            int front_idx = 1 - evolution_[i].ltBufIdx;
            cudaMemcpy(evolution_[i].LtGpu[front_idx], evolution_[i].Lt.data, evolution_[i].imgSize, cudaMemcpyHostToDevice);
#endif

#if GPU_MEM
            gaussian_2D_convolution(evolution_[i].LtGpu[dst_front], evolution_[i].LsmoothGpu, 0, 0, 1.0,
                                    img_width,  img_height);
#else
            gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0);
#endif
            // Compute the Gaussian derivatives Lx and Ly
#if GPU_MEM
            image_derivatives_scharr( evolution_[i].LsmoothGpu, evolution_[i].LxGpu,  0, img_width, img_height);
            image_derivatives_scharr( evolution_[i].LsmoothGpu, evolution_[i].LyGpu,  1, img_width, img_height);
#else
            image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx,  0);
            image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly,  1);
#endif

            if(options_.kcontrast < 0.0f)
            {
               //download to CV mat before using...
#if GPU_MEM
                cudaMemcpy(evolution_[1].Lx.data, evolution_[1].LxGpu, evolution_[1].imgSize, cudaMemcpyDeviceToHost);
                cudaMemcpy(evolution_[1].Ly.data, evolution_[1].LyGpu, evolution_[1].imgSize, cudaMemcpyDeviceToHost);
#endif
                options_.kcontrast = compute_k_percentile_share(evolution_[i].Lx, evolution_[i].Ly,
                                                              options_.kcontrast_percentile , 1.0, options_.kcontrast_nbins);
            }

            // Compute the conductivity equation
            switch (options_.diffusivity) {
                case PM_G1:
                   // pm_g1(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
                break;
                case PM_G2:
#if GPU_MEM
                    pm_g2(evolution_[i].LxGpu, evolution_[i].LyGpu, evolution_[i].LflowGpu,
                          options_.kcontrast,img_width,img_height);
#else
                    pm_g2(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
#endif
                break;
                case WEICKERT:
                   // weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
                break;
                case CHARBONNIER:
                   // charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
                break;
                default:
                    cerr << "Diffusivity: " << options_.diffusivity << " is not supported" << endl;
            }

        // Perform FED n inner steps
            for (int j = 0; j < nsteps_[i-1]; j++)
            {
#if GPU_MEM
                int back_idx = evolution_[i].ltBufIdx;
                float* cur_dst = evolution_[i].LtGpu[back_idx];
                float* cur_src = evolution_[i].LtGpu[1-back_idx];
                nld_step_scalar(cur_dst, cur_src, evolution_[i].LflowGpu ,  tsteps_[i-1][j], img_width, img_height);

                evolution_[i].ltBufIdx = 1-back_idx;
#else
                nld_step_scalar(evolution_[i].Lt, evolution_[i].Lflow, evolution_[i].Lstep, tsteps_[i-1][j]);
#endif
            }

#if 0
            //copy lt buffer to front
             front_idx = 1 - evolution_[i].ltBufIdx;
            cudaMemcpy(evolution_[i].Lt.data, evolution_[i].LtGpu[front_idx], evolution_[i].imgSize, cudaMemcpyDeviceToHost);

            // TODO:: for later descritpor gpu use clean me up
            //cudaMemcpy(evolution_[i].LtGpu[1-front_idx], evolution_[i].Lt.data, evolution_[i].imgSize, cudaMemcpyHostToDevice);
#endif
        //     cv::namedWindow( "Display window",   cv::WINDOW_AUTOSIZE );// Create a window for display.
        //     cv::imshow( "Display window", evolution_[i].Lt );                   // Show our image inside it.
        //    cv::waitKey(0);

        }

        t2 = cv::getTickCount();
        timing_.scale = 1000.0*(t2-t1) / cv::getTickFrequency();

        return 0;
    }

    /* ************************************************************************* */
    void AKAZE::Feature_Detection(std::vector<cv::KeyPoint>& kpts) {

        double t1 = 0.0, t2 = 0.0;

        t1 = cv::getTickCount();
        vector<cv::KeyPoint>().swap(kpts);

        Compute_Determinant_Hessian_Response();

        Find_Scale_Space_Extrema(kpts);
        Do_Subpixel_Refinement(kpts);

        t2 = cv::getTickCount();
        timing_.detector = 1000.0*(t2-t1) / cv::getTickFrequency();

        timing_.cudaAsync = 0;
#if 0

        double t3 = cv::getTickCount();
        cudaStream_t copyStream0, copyStream1;
        cudaStreamCreate(&copyStream0);
        cudaStreamCreate(&copyStream1);
        // we copy the remaining level of Lx data back to cpu in async way
        for (size_t i = 0; i < evolution_.size(); i++) {

            // we already copied first during k percentile compute
            //if ( i != 1)
            {
                cudaMemcpyAsync(evolution_[i].Lx.data, evolution_[i].LxGpu, evolution_[i].imgSize, cudaMemcpyDeviceToHost, copyStream0);
                cudaMemcpyAsync(evolution_[i].Ly.data, evolution_[i].LyGpu, evolution_[i].imgSize, cudaMemcpyDeviceToHost, copyStream1);

                //cudaMemcpy(evolution_[i].LtGpu[0], evolution_[i].Lt.data, evolution_[i].imgSize, cudaMemcpyHostToDevice);
            }
        }
        double calltime = cv::getTickCount() - t3;
        //  cout << " cudaMemcpyAsync call time: " << calltime*1000.0 / cv::getTickFrequency() << endl;

        timing_.cudaAsync = calltime*1000.0 / cv::getTickFrequency();
#endif
    }

    /* ************************************************************************* */
    void AKAZE::Compute_Multiscale_Derivatives() {

        double t1 = 0.0, t2 = 0.0;

        t1 = cv::getTickCount();

        #ifdef _OPENMP_NO
        omp_set_num_threads(OMP_MAX_THREADS);
        #pragma omp parallel for
        #endif

        for (int i = 0; i < (int) evolution_.size(); i++) {

            float ratio = pow(2.0f,(float)evolution_[i].octave);
            int sigma_size_ = fRound(evolution_[i].esigma*options_.derivative_factor/ratio);

            int width = evolution_[i].imgWidth;
            int height = evolution_[i].imgHeight;
#if GPU_MEM
            compute_scharr_derivatives(evolution_[i].LsmoothGpu, evolution_[i].LxGpu, 1, 0, sigma_size_, width, height);
            compute_scharr_derivatives(evolution_[i].LsmoothGpu, evolution_[i].LyGpu, 0, 1, sigma_size_, width, height);
            compute_scharr_derivatives(evolution_[i].LxGpu, evolution_[i].LxxGpu, 1, 0, sigma_size_, width, height);
            compute_scharr_derivatives(evolution_[i].LyGpu, evolution_[i].LyyGpu, 0, 1, sigma_size_, width, height);
            compute_scharr_derivatives(evolution_[i].LxGpu, evolution_[i].LxyGpu, 0, 1, sigma_size_, width, height);

#else
            compute_scharr_derivatives(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0, sigma_size_);
            compute_scharr_derivatives(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1, sigma_size_);
            compute_scharr_derivatives(evolution_[i].Lx, evolution_[i].Lxx, 1, 0, sigma_size_);
            compute_scharr_derivatives(evolution_[i].Ly, evolution_[i].Lyy, 0, 1, sigma_size_);
            compute_scharr_derivatives(evolution_[i].Lx, evolution_[i].Lxy, 0, 1, sigma_size_);
#endif
            //TODO combine this pass with determine calculation
            evolution_[i].multiDerSigmaSize = sigma_size_;
#if 0
            evolution_[i].Lx = evolution_[i].Lx*((sigma_size_));
            evolution_[i].Ly = evolution_[i].Ly*((sigma_size_));
            evolution_[i].Lxx = evolution_[i].Lxx*((sigma_size_)*(sigma_size_));
            evolution_[i].Lxy = evolution_[i].Lxy*((sigma_size_)*(sigma_size_));
            evolution_[i].Lyy = evolution_[i].Lyy*((sigma_size_)*(sigma_size_));
#endif
        }

        t2 = cv::getTickCount();
        timing_.derivatives = 1000.0*(t2-t1) / cv::getTickFrequency();
    }

    /* ************************************************************************* */
    void AKAZE::Compute_Determinant_Hessian_Response() {

        // Firstly compute the multiscale derivatives
        Compute_Multiscale_Derivatives();

        cal_determinant_hessian( evolution_, options_.verbosity );
    }

    /* ************************************************************************* */


    #define CUDA_TEST 1
    void AKAZE::Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts) {

        double t1 = 0.0, t2 = 0.0;
        float value = 0.0;
        float dist = 0.0, ratio = 0.0, smax = 0.0;
        int npoints = 0, id_repeated = 0;
        int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
        bool is_extremum = false, is_repeated = false, is_out = false;
        cv::KeyPoint point;
        vector<cv::KeyPoint> kpts_aux;

        // Set maximum size
        if (options_.descriptor == SURF_UPRIGHT || options_.descriptor == SURF ||
        options_.descriptor == MLDB_UPRIGHT || options_.descriptor == MLDB) {
            smax = 10.0*sqrtf(2.0f);
        }
        else if (options_.descriptor == MSURF_UPRIGHT || options_.descriptor == MSURF) {
            smax = 12.0*sqrtf(2.0f);
        }

        t1 = cv::getTickCount();

        #if 0
        local_extrema(options_, evolution_ );
        #else

        for (size_t i = 0; i < evolution_.size(); i++) {

            for (int ix = 1; ix < evolution_[i].Ldet.rows-1; ix++) {

                float* ldet_m = evolution_[i].Ldet.ptr<float>(ix-1);
                float* ldet = evolution_[i].Ldet.ptr<float>(ix);
                float* ldet_p = evolution_[i].Ldet.ptr<float>(ix+1);

                for (int jx = 1; jx < evolution_[i].Ldet.cols-1; jx++) {

                    is_extremum = false;
                    is_repeated = false;
                    is_out = false;
                    value = ldet[jx];

                    //printf("-- 0000 at %.2f \n", ldet[jx]);

                    // Filter the points with the detector threshold
                    if (value > options_.dthreshold && value >= options_.min_dthreshold &&
                    value > ldet[jx-1] && value > ldet[jx+1] &&
                    value > ldet_m[jx-1] && value > ldet_m[jx] && value > ldet_m[jx+1] &&
                    value > ldet_p[jx-1] && value > ldet_p[jx] && value > ldet_p[jx+1]) {

                        is_extremum = true;
                        point.response = fabs(value);
                        point.size = evolution_[i].esigma*options_.derivative_factor;
                        point.octave = evolution_[i].octave;
                        point.class_id = i;
                        ratio = pow(2.0f, point.octave);
                        sigma_size_ = fRound(point.size/ratio);
                        point.pt.x = jx;
                        point.pt.y = ix;

                        // Compare response with the same and lower scale
                        for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

                            if ((point.class_id-1) == kpts_aux[ik].class_id ||
                            point.class_id == kpts_aux[ik].class_id) {

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

                            if (left_x < 0 || right_x >= evolution_[i].Ldet.cols ||
                                up_y < 0 || down_y >= evolution_[i].Ldet.rows) {
                                is_out = true;
                            }

                            if (is_out == false) {
                                if (is_repeated == false) {
                                    point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                    point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                    kpts_aux.push_back(point);
                                    npoints++;
                                }
                            else {
                                    point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                    point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                    kpts_aux[id_repeated] = point;
                                }
                            } // if is_out
                        } //if is_extremum
                    }
                } // for jx
            } // for ix
        } // for i
        #endif


        // Now filter points with the upper scale level
        for (size_t i = 0; i < kpts_aux.size(); i++) {

            is_repeated = false;
            const cv::KeyPoint& point = kpts_aux[i];
            for (size_t j = i+1; j < kpts_aux.size(); j++) {

                // Compare response with the upper scale
                if ((point.class_id+1) == kpts_aux[j].class_id) {

                dist = (point.pt.x-kpts_aux[j].pt.x)*(point.pt.x-kpts_aux[j].pt.x) +
                    (point.pt.y-kpts_aux[j].pt.y)*(point.pt.y-kpts_aux[j].pt.y);

                if (dist <= point.size*point.size) {
                  if (point.response < kpts_aux[j].response) {
                    is_repeated = true;
                    break;
                  }
                }
            }
        }

        if (is_repeated == false)
            kpts.push_back(point);
        }

        //  int kp_found = kpts.size();
        t2 = cv::getTickCount();
        timing_.extrema = 1000.0*(t2-t1) / cv::getTickFrequency();
    }

    /* ************************************************************************* */
    void AKAZE::Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts) {

        double t1 = 0.0, t2 = 0.0;
        float Dx = 0.0, Dy = 0.0, ratio = 0.0;
        float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
        int x = 0, y = 0;
        cv::Matx22f A(0.0, 0.0, 0.0, 0.0);
        cv::Vec2f b(0.0, 0.0);
        cv::Vec2f dst(0.0, 0.0);

        t1 = cv::getTickCount();

        for (size_t i = 0; i < kpts.size(); i++) {
            ratio = pow(2.f,kpts[i].octave);
            x = fRound(kpts[i].pt.x/ratio);
            y = fRound(kpts[i].pt.y/ratio);

            // Compute the gradient
            Dx = (0.5)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x+1)
            -*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x-1));
            Dy = (0.5)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x)
            -*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x));

            // Compute the Hessian
            Dxx = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x+1)
            + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x-1)
            -2.0*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

            Dyy = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x)
            + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x)
            -2.0*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

            Dxy = (0.25)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x+1)
            +(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x-1)))
            -(0.25)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x+1)
            +(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x-1)));

            // Solve the linear system
            A(0,0) = Dxx;
            A(1,1) = Dyy;
            A(0,1) = A(1,0) = Dxy;
            b(0) = -Dx;
            b(1) = -Dy;

            cv::solve(A, b, dst, cv::DECOMP_LU);

            if (fabs(dst(0)) <= 1.0 && fabs(dst(1)) <= 1.0) {
                kpts[i].pt.x = x + dst(0);
                kpts[i].pt.y = y + dst(1);
                int power = powf(2, evolution_[kpts[i].class_id].octave);
                kpts[i].pt.x *= power;
                kpts[i].pt.y *= power;
                kpts[i].angle = 0.0;

                // In OpenCV the size of a keypoint its the diameter
                kpts[i].size *= 2.0;
            }
            // Delete the point since its not stable
            else {
                kpts.erase(kpts.begin()+i);
                i--;
            }
        }

        t2 = cv::getTickCount();
        timing_.subpixel = 1000.0*(t2-t1) / cv::getTickFrequency();
    }

    /* ************************************************************************* */
    /**
    * @brief This method  computes the set of descriptors through the nonlinear scale space
    * @param kpts Vector of detected keypoints
    * @param desc Matrix to store the descriptors
    */
    void AKAZE::Compute_Descriptors( std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {

        double t1 = 0.0, t2 = 0.0;

        t1 = cv::getTickCount();

        // Allocate memory for the matrix with the descriptors
        if (options_.descriptor < MLDB_UPRIGHT) {
                desc = cv::Mat::zeros(kpts.size(), 64, CV_32FC1);
            }
            else {
            // We use the full length binary descriptor -> 486 bits
            if (options_.descriptor_size == 0) {
                int t = (6+36+120)*options_.descriptor_channels;
                desc = cv::Mat::zeros(kpts.size(), ceil(t/8.), CV_8UC1);
            }
            else {
                // We use the random bit selection length binary descriptor
                desc = cv::Mat::zeros(kpts.size(), ceil(options_.descriptor_size/8.), CV_8UC1);
            }
        }

        // right before we need the extra lx data, we syncornize.
#if GPU_MEM
        double t3 = cv::getTickCount();
        cudaDeviceSynchronize();
        double sync_time = cv::getTickCount()- t3;
        cout<< "cudaDeviceSynchronize: " << sync_time * 1000.0 / cv::getTickFrequency() << endl;
#endif
        //
        switch (options_.descriptor) {

        case SURF_UPRIGHT : // Upright descriptors, not invariant to rotation
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < (int)(kpts.size()); i++) {
                Get_SURF_Descriptor_Upright_64(kpts[i],desc.ptr<float>(i));
            }
        }
        break;
        case SURF :
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < (int)(kpts.size()); i++) {
                Compute_Main_Orientation(kpts[i]);
                Get_SURF_Descriptor_64(kpts[i],desc.ptr<float>(i));
            }
        }
        break;
        case MSURF_UPRIGHT : // Upright descriptors, not invariant to rotation
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < (int)(kpts.size()); i++) {
                Get_MSURF_Upright_Descriptor_64(kpts[i],desc.ptr<float>(i));
            }
        }
        break;
        case MSURF :
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < (int)(kpts.size()); i++) {
                Compute_Main_Orientation(kpts[i]);
                Get_MSURF_Descriptor_64(kpts[i],desc.ptr<float>(i));
            }
        }
        break;
        case MLDB_UPRIGHT : // Upright descriptors, not invariant to rotation
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < (int)(kpts.size()); i++) {
            if (options_.descriptor_size == 0)
              Get_Upright_MLDB_Full_Descriptor(kpts[i], desc.ptr<unsigned char>(i));
            else
              Get_Upright_MLDB_Descriptor_Subset(kpts[i], desc.ptr<unsigned char>(i));
            }
        }
        break;
        case MLDB :
        {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif

            for (int i = 0; i < (int)(kpts.size()); i++) {

                Compute_Main_Orientation(kpts[i]);
                if (options_.descriptor_size == 0)
                  Get_MLDB_Full_Descriptor(kpts[i], desc.ptr<unsigned char>(i));
                else
                  Get_MLDB_Descriptor_Subset(kpts[i], desc.ptr<unsigned char>(i));

            }
        }
        break;
        }

        t2 = cv::getTickCount();
        timing_.descriptor = 1000.0*(t2-t1) / cv::getTickFrequency();
    }


    static DspKPCUDAEntry CUDA_DESCRIPTOR_ENTRY[2048];
    void AKAZE::Compute_DescriptorsGpu( std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {

        // Allocate memory for the matrix with the descriptors

        double t1 = 0.0, t2 = 0.0;
        t1 = cv::getTickCount();
        desc = cv::Mat::zeros(kpts.size(), 64, CV_8UC1);

        //MLDB only
        // hack throw last one out to make dividable by 2

        int kpt_count = kpts.size();

        char* cuda_base = GetCudaMem( 128 );

        int data_size = kpt_count*sizeof(DspKPCUDAEntry) + 128;
        DspKPCUDAEntry* d_dst = (DspKPCUDAEntry*)( cuda_base  );
        unsigned char* des_dst = (unsigned char*)( cuda_base + data_size);

        cudaMemsetAsync( des_dst, 0, 64*kpt_count );
        //float* client_dst = (float*)malloc( img_size);

        const int max_channels = 3;

        for (int i = 0; i < kpt_count; i++) {

            int level = kpts[i].class_id;
            float ratio = (float)(1<<evolution_[level].octave);

            int front_idx = level == 0 ? 0 :  1-evolution_[level].ltBufIdx;

            CUDA_DESCRIPTOR_ENTRY[i].imgWidth = evolution_[level].imgWidth;
            CUDA_DESCRIPTOR_ENTRY[i].pt_x = kpts[i].pt.x;
            CUDA_DESCRIPTOR_ENTRY[i].pt_y = kpts[i].pt.y;
            CUDA_DESCRIPTOR_ENTRY[i].angle = 0.0f;
            CUDA_DESCRIPTOR_ENTRY[i].lx = evolution_[level].LxGpu;
            CUDA_DESCRIPTOR_ENTRY[i].ly = evolution_[level].LyGpu;
            CUDA_DESCRIPTOR_ENTRY[i].lt = evolution_[level].LtGpu[front_idx];
            CUDA_DESCRIPTOR_ENTRY[i].size = kpts[i].size;
            CUDA_DESCRIPTOR_ENTRY[i].ratio = ratio;
            //CUDA_DESCRIPTOR_ENTRY[i].vals = des_dst + i*16*max_channels;
        }
        cudaMemcpy(d_dst, CUDA_DESCRIPTOR_ENTRY , kpt_count*sizeof(DspKPCUDAEntry), cudaMemcpyHostToDevice);

        //getLastCudaError("ComputeMainOrientationGPU");
        ComputeMainOrientationGPU(d_dst, des_dst, kpt_count);
        //cudaError_t error = cudaMemcpy(CUDA_DESCRIPTOR_ENTRY, d_dst , kpt_count*sizeof(DspKPCUDAEntry)  , cudaMemcpyDeviceToHost);

#if 0
        double t3 = cv::getTickCount();
        cudaDeviceSynchronize();
        double sync_time = cv::getTickCount()- t3;
        cout<< "cudaDeviceSynchronize: " << sync_time * 1000.0 / cv::getTickFrequency() << endl;
#endif
        //static unsigned char gpu_values[64*1024];

        cudaError_t error = cudaMemcpy(desc.data, des_dst , 64*kpt_count , cudaMemcpyDeviceToHost);

        t2 = cv::getTickCount();
        timing_.descriptor = 1000.0*(t2-t1) / cv::getTickFrequency();
        getLastCudaError("ComputeMainOrientationGPU");

#if 0
        int idx = 0;
        for (int i = 0; i < kpt_count; i++) {
            idx += 64;
            const unsigned char* reader = desc.ptr<unsigned char>(i);
            cout << " >>> first 4 byte <<< " <<  (short)reader[0] << (short)reader[1] << (short)reader[2] << (short)reader[3] <<endl;
        }
        getLastCudaError("ComputeMainOrientationGPU");
        for (int i = 0; i < (int)(kpts.size()); i++) {

            Compute_Main_Orientation(kpts[i]);

            unsigned char* desp = desc.ptr<unsigned char>(i);

           //cout<< "kp_angle" << kpts[i].angle << " / " << CUDA_DESCRIPTOR_ENTRY[i].angle << endl;

            Get_MLDB_Full_Descriptor(kpts[i], desc.ptr<unsigned char>(i));


            CV_Assert(options_.descriptor_channels <= max_channels);
            float values[16*max_channels];
            const float size_mult[3] = {1, 2.0/3.0, 1.0/2.0};

            float ratio = (float)(1 << kpts[i].octave);
            float scale = (float)fRound(0.5f*kpts[i].size / ratio);
            float xf = kpts[i].pt.x / ratio;
            float yf = kpts[i].pt.y / ratio;
            float co = cos(kpts[i].angle);
            float si = sin(kpts[i].angle);
            int pattern_size = options_.descriptor_pattern_size;


            int dpos = 0;
            for(int lvl = 0; lvl < 3; lvl++) {

                int val_count = (lvl + 2) * (lvl + 2);
                int sample_step = static_cast<int>(ceil(pattern_size * size_mult[lvl]));

                ////////////////
                int pattern_size = options_.descriptor_pattern_size;
                int nr_channels = options_.descriptor_channels;
                int valpos = 0;

                //    cout << "=========== one MLDB ================ " << endl;
                //    cout << "xf: " << xf << " yf:" << xf << " level: " <<  level << endl;

                for (int f = -pattern_size; f < pattern_size; f += sample_step) {
                    for (int j = -pattern_size; j < pattern_size; j += sample_step) {

                        float di = 0.0, dx = 0.0, dy = 0.0;
                        int nsamples = 0;

                        for (int k = f; k < f + sample_step; k++) {
                            for (int l = j; l < j + sample_step; l++) {

                                float sample_y = yf + (l*co*scale + k*si*scale);
                                float sample_x = xf + (-l*si*scale + k*co*scale);

                                int y1 = fRound(sample_y);
                                int x1 = fRound(sample_x);

                                float ri = *(evolution_[kpts[i].class_id].Lt.ptr<float>(y1)+x1);

                               // if( ri < 10.0f )
                                {

                                    di += ri;

                                    if(nr_channels > 1) {
                                        float rx = *(evolution_[kpts[i].class_id].Lx.ptr<float>(y1)+x1);
                                        float ry = *(evolution_[kpts[i].class_id].Ly.ptr<float>(y1)+x1);

                                        if (nr_channels == 2) {
                                            dx += sqrtf(rx*rx + ry*ry);
                                        }
                                        else {
                                            float rry = rx*co + ry*si;
                                            float rrx = -rx*si + ry*co;
                                            dx += rrx;
                                            dy += rry;


                                        }
                                    }

                                    nsamples++;

                                }


                            }
                        }

                        di /= nsamples;
                        dx /= nsamples;
                        dy /= nsamples;

                        values[valpos] = di;

                        if (nr_channels > 1)
                            values[valpos + 1] = dx;

                        if (nr_channels > 2)
                            values[valpos + 2] = dy;

                        valpos += nr_channels;

                        //               cout << " dx: "<< dx << " dy: " << dy << " di: "   << di << endl;
                    }
                }


                /////////////////////

               // int nr_channels = options_.descriptor_channels;

                for(int pos = 0; pos < nr_channels; pos++) {
                    for (int k = 0; k < val_count; k++) {
                        float ival = values[nr_channels * k + pos];
                        for (int j = k + 1; j < val_count; j++) {
                            int res = ival > values[nr_channels * j + pos];
                            desp[dpos >> 3] |= (res << (dpos & 7));
                            dpos++;
                        }
                    }
                }
            }

/*
            for(int p = 0; p < 48; ++p)
            {
                float g_di = gpu_values[ i*48 + p];

                printf(" -- values %f  \n", values[p] - g_di);
                //  printf(" -- values %f - %f\n", dx, g_dx);
                //  printf(" -- values %f - %f\n", dy, g_dy);
            }
*/

#if 1
            int byte_diff = 0;
            unsigned char* descgpup = &gpu_values[i*64];
            for(int h = 0; h<61;++h)
            {
                int diff = desp[h] - descgpup[h];
                //cout<< desp[h] ;
                if( diff != 0)
                {
                    byte_diff++;
                }
            }
            if( byte_diff > 3)
            {
                cout<< "descriptor " << i << " diff: " << byte_diff  << " level: " << kpts[i].class_id << " val: " << values[0] <<endl;
            }
#endif



           // float diff = values[0] - CUDA_DESCRIPTOR_ENTRY[i].angle;
          //  cout<< "values: " << diff << " / "  << kpts[i].class_id <<endl;
        }
#endif

    }

    /* ************************************************************************* */
    void AKAZE::Compute_Main_Orientation(cv::KeyPoint& kpt) const {

        int ix = 0, iy = 0, idx = 0, s = 0, level = 0;
        float xf = 0.0, yf = 0.0, gweight = 0.0, ratio = 0.0;
        float resX[109], resY[109], Ang[109];
        const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

        // Variables for computing the dominant direction
        float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

        // Get the information from the keypoint
        level = kpt.class_id;
        ratio = (float)(1<<evolution_[level].octave);
        s = fRound(0.5*kpt.size/ratio);
        xf = kpt.pt.x/ratio;
        yf = kpt.pt.y/ratio;

        // Calculate derivatives responses for points within radius of 6*scale
        //cout<< "kp_angle: ";
        for (int i = -6; i <= 6; ++i) {
            for (int j = -6; j <= 6; ++j) {
                if (i*i + j*j < 36) {
                    iy = fRound(yf + j*s);
                    ix = fRound(xf + i*s);

                    gweight = gauss25[id[i+6]][id[j+6]];
                    resX[idx] = gweight*(*(evolution_[level].Lx.ptr<float>(iy)+ix));
                    resY[idx] = gweight*(*(evolution_[level].Ly.ptr<float>(iy)+ix));
                    Ang[idx] = cv::fastAtan2(resY[idx], resX[idx])*(CV_PI/180.0);
                    //float angle = atan2f(resY[idx], resX[idx])*(CV_PI/180.0);
                    ++idx;
                }
            }
        }

        // Loop slides pi/3 window around feature point
        for (ang1 = 0; ang1 < 2.0*CV_PI;  ang1+=0.15f) {

            ang2 =(ang1+CV_PI/3.0f > 2.0*CV_PI ? ang1-5.0f*CV_PI/3.0f : ang1+CV_PI/3.0f);
            sumX = sumY = 0.f;

            for (size_t k = 0; k < 109; ++k) {
                // Get angle from the x-axis of the sample point
                const float& ang = Ang[k];

                // Determine whether the point is within the window
                if (ang1 < ang2 && ang1 < ang && ang < ang2) {
                    sumX+=resX[k];
                    sumY+=resY[k];
                }
                else if (ang2 < ang1 && ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2.0*CV_PI))) {
                    sumX+=resX[k];
                    sumY+=resY[k];
                }
            }

            // if the vector produced from this window is longer than all
            // previous vectors then this forms the new dominant direction
            if (sumX*sumX + sumY*sumY > max) {
            // store largest orientation
                max = sumX*sumX + sumY*sumY;
                kpt.angle =  atan2f(sumY, sumX);
               // kpt.angle =  cv::fastAtan2(sumY, sumX)*(CV_PI/180.0);
            }
        }
    }

    /* ************************************************************************* */
    void AKAZE::Get_SURF_Descriptor_Upright_64(const cv::KeyPoint& kpt, float *desc) const {
    #if 1
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
    float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
    float sample_x = 0.0, sample_y = 0.0;
    float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
    int scale = 0, dsize = 0, level = 0;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 10;

    // Get the information from the keypoint
    ratio = (float)(1<<kpt.octave);
    scale = fRound(0.5*kpt.size/ratio);
    level = kpt.class_id;
    yf = kpt.pt.y/ratio;
    xf = kpt.pt.x/ratio;

    // Calculate descriptor for this interest point
    for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {

    dx=dy=mdx=mdy=0.0;

    for (int k = i; k < i + sample_step; k++) {
    for (int l = j; l < j + sample_step; l++) {

      // Get the coordinates of the sample point on the rotated axis
      sample_y = yf + l*scale;
      sample_x = xf + k*scale;

      y1 = (int)(sample_y-.5);
      x1 = (int)(sample_x-.5);

      y2 = (int)(sample_y+.5);
      x2 = (int)(sample_x+.5);

      fx = sample_x-x1;
      fy = sample_y-y1;

      res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
      rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
      ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      // Sum the derivatives to the cumulative descriptor
      dx += rx;
      dy += ry;
      mdx += fabs(rx);
      mdy += fabs(ry);
    }
    }

    // Add the values to the descriptor vector
    desc[dcount++] = dx;
    desc[dcount++] = dy;
    desc[dcount++] = mdx;
    desc[dcount++] = mdy;

    // Store the current length^2 of the vector
    len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
    }

    // convert to unit vector
    len = sqrt(len);

    for (int i = 0; i < dsize; i++) {
    desc[i] /= len;
    }
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc) const {
    #if 1
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
    float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
    float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
    float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
    int scale = 0, dsize = 0, level = 0;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 10;

    // Get the information from the keypoint
    ratio = (float)(1<<kpt.octave);
    scale = fRound(0.5*kpt.size/ratio);
    angle = kpt.angle;
    level = kpt.class_id;
    yf = kpt.pt.y/ratio;
    xf = kpt.pt.x/ratio;
    co = cos(angle);
    si = sin(angle);

    // Calculate descriptor for this interest point
    for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {

    dx=dy=mdx=mdy=0.0;

    for (int k = i; k < i + sample_step; k++) {
    for (int l = j; l < j + sample_step; l++) {

      // Get the coordinates of the sample point on the rotated axis
      sample_y = yf + (l*scale*co + k*scale*si);
      sample_x = xf + (-l*scale*si + k*scale*co);

      y1 = (int)(sample_y-.5);
      x1 = (int)(sample_x-.5);

      y2 = (int)(sample_y+.5);
      x2 = (int)(sample_x+.5);

      fx = sample_x-x1;
      fy = sample_y-y1;

      res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
      rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
      ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      // Get the x and y derivatives on the rotated axis
      rry = rx*co + ry*si;
      rrx = -rx*si + ry*co;

      // Sum the derivatives to the cumulative descriptor
      dx += rrx;
      dy += rry;
      mdx += fabs(rrx);
      mdy += fabs(rry);
    }
    }

    // Add the values to the descriptor vector
    desc[dcount++] = dx;
    desc[dcount++] = dy;
    desc[dcount++] = mdx;
    desc[dcount++] = mdy;

    // Store the current length^2 of the vector
    len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
    }

    // convert to unit vector
    len = sqrt(len);

    for (int i = 0; i < dsize; i++)
    desc[i] /= len;
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float *desc) const {

    #if 1
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0;
    int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
    int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int scale = 0, dsize = 0, level = 0;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5, cy = 0.5;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    ratio = (float)(1<<kpt.octave);
    scale = fRound(0.5*kpt.size/ratio);
    level = kpt.class_id;
    yf = kpt.pt.y/ratio;
    xf = kpt.pt.x/ratio;

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {
    j = -8;
    i = i-4;

    cx += 1.0;
    cy = -0.5;

    while (j < pattern_size) {
    dx=dy=mdx=mdy=0.0;
    cy += 1.0;
    j = j-4;

    ky = i + sample_step;
    kx = j + sample_step;

    ys = yf + (ky*scale);
    xs = xf + (kx*scale);

    for (int k = i; k < i+9; k++) {
    for (int l = j; l < j+9; l++) {
      sample_y = k*scale + yf;
      sample_x = l*scale + xf;

      //Get the gaussian weighted x and y responses
      gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.50*scale);

      y1 = (int)(sample_y-.5);
      x1 = (int)(sample_x-.5);

      y2 = (int)(sample_y+.5);
      x2 = (int)(sample_x+.5);

      fx = sample_x-x1;
      fy = sample_y-y1;

      res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
      rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
      ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      rx = gauss_s1*rx;
      ry = gauss_s1*ry;

      // Sum the derivatives to the cumulative descriptor
      dx += rx;
      dy += ry;
      mdx += fabs(rx);
      mdy += fabs(ry);
    }
    }

    // Add the values to the descriptor vector
    gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

    desc[dcount++] = dx*gauss_s2;
    desc[dcount++] = dy*gauss_s2;
    desc[dcount++] = mdx*gauss_s2;
    desc[dcount++] = mdy*gauss_s2;

    len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

    j += 9;
    }

    i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (int i = 0; i < dsize; i++)
    desc[i] /= len;
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc) const {
    #if 1
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
    float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
    int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    int scale = 0, dsize = 0, level = 0;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5, cy = 0.5;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    ratio = (float)(1<<kpt.octave);
    scale = fRound(0.5*kpt.size/ratio);
    angle = kpt.angle;
    level = kpt.class_id;
    yf = kpt.pt.y/ratio;
    xf = kpt.pt.x/ratio;
    co = cos(angle);
    si = sin(angle);

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {
    j = -8;
    i = i-4;

    cx += 1.0;
    cy = -0.5;

    while (j < pattern_size) {
    dx=dy=mdx=mdy=0.0;
    cy += 1.0;
    j = j - 4;

    ky = i + sample_step;
    kx = j + sample_step;

    xs = xf + (-kx*scale*si + ky*scale*co);
    ys = yf + (kx*scale*co + ky*scale*si);

    for (int k = i; k < i + 9; ++k) {
    for (int l = j; l < j + 9; ++l) {
      // Get coords of sample point on the rotated axis
      sample_y = yf + (l*scale*co + k*scale*si);
      sample_x = xf + (-l*scale*si + k*scale*co);

      // Get the gaussian weighted x and y responses
      gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5*scale);

      y1 = fRound(sample_y-.5);
      x1 = fRound(sample_x-.5);

      y2 = fRound(sample_y+.5);
      x2 = fRound(sample_x+.5);

      fx = sample_x-x1;
      fy = sample_y-y1;

      res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
      rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
      res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
      res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
      res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
      ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

      // Get the x and y derivatives on the rotated axis
      rry = gauss_s1*(rx*co + ry*si);
      rrx = gauss_s1*(-rx*si + ry*co);

      // Sum the derivatives to the cumulative descriptor
      dx += rrx;
      dy += rry;
      mdx += fabs(rrx);
      mdy += fabs(rry);
    }
    }

    // Add the values to the descriptor vector
    gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);
    desc[dcount++] = dx*gauss_s2;
    desc[dcount++] = dy*gauss_s2;
    desc[dcount++] = mdx*gauss_s2;
    desc[dcount++] = mdy*gauss_s2;

    len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

    j += 9;
    }

    i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (int i = 0; i < dsize; i++)
    desc[i] /= len;
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const {

        const int max_channels = 3;
        CV_Assert(options_.descriptor_channels <= max_channels);
        float values[16*max_channels];
        const float size_mult[3] = {1, 2.0/3.0, 1.0/2.0};

        float ratio = (float)(1 << kpt.octave);
        float scale = (float)fRound(0.5f*kpt.size / ratio);
        float xf = kpt.pt.x / ratio;
        float yf = kpt.pt.y / ratio;
        float co = cos(kpt.angle);
        float si = sin(kpt.angle);
        int pattern_size = options_.descriptor_pattern_size;

        int dpos = 0;
        for(int lvl = 0; lvl < 3; lvl++) {
            int val_count = (lvl + 2) * (lvl + 2);
            int sample_step = static_cast<int>(ceil(pattern_size * size_mult[lvl]));
            MLDB_Fill_Values(values, sample_step, kpt.class_id, xf, yf, co, si, scale);
            MLDB_Binary_Comparisons(values, desc, val_count, dpos);
        }
    }

    /* ************************************************************************* */
    void AKAZE::Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const {

    const int max_channels = 3;
    CV_Assert(options_.descriptor_channels <= max_channels);
    float values[16*max_channels];
    const double size_mult[3] = {1, 2.0/3.0, 1.0/2.0};

    float ratio = (float)(1 << kpt.octave);
    float scale = (float)fRound(0.5f*kpt.size / ratio);
    float xf = kpt.pt.x / ratio;
    float yf = kpt.pt.y / ratio;
    int pattern_size = options_.descriptor_pattern_size;

    int dpos = 0;
    for(int lvl = 0; lvl < 3; lvl++) {
    int val_count = (lvl + 2) * (lvl + 2);
    int sample_step = static_cast<int>(ceil(pattern_size * size_mult[lvl]));
    MLDB_Fill_Upright_Values(values, sample_step, kpt.class_id, xf, yf, scale);
    MLDB_Binary_Comparisons(values, desc, val_count, dpos);
    }
    }

    /* ************************************************************************* */
    void AKAZE::MLDB_Fill_Values(float* values, int sample_step, int level,
                         float xf, float yf, float co, float si, float scale) const {
#if !GPU_MEM
        int pattern_size = options_.descriptor_pattern_size;
        int nr_channels = options_.descriptor_channels;
        int valpos = 0;

    //    cout << "=========== one MLDB ================ " << endl;
    //    cout << "xf: " << xf << " yf:" << xf << " level: " <<  level << endl;

        for (int i = -pattern_size; i < pattern_size; i += sample_step) {
            for (int j = -pattern_size; j < pattern_size; j += sample_step) {

                float di = 0.0, dx = 0.0, dy = 0.0;
                int nsamples = 0;

                for (int k = i; k < i + sample_step; k++) {
                    for (int l = j; l < j + sample_step; l++) {

                      float sample_y = yf + (l*co*scale + k*si*scale);
                      float sample_x = xf + (-l*si*scale + k*co*scale);

                      int y1 = fRound(sample_y);
                      int x1 = fRound(sample_x);

                      float ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
                      di += ri;

                      if(nr_channels > 1) {
                        float rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
                        float ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);
                        if (nr_channels == 2) {
                          dx += sqrtf(rx*rx + ry*ry);
                        }
                        else {
                          float rry = rx*co + ry*si;
                          float rrx = -rx*si + ry*co;
                          dx += rrx;
                          dy += rry;
                        }
                      }
                      nsamples++;
                    }
                }

                di /= nsamples;
                dx /= nsamples;
                dy /= nsamples;

                values[valpos] = di;

                if (nr_channels > 1)
                values[valpos + 1] = dx;

                if (nr_channels > 2)
                values[valpos + 2] = dy;

                valpos += nr_channels;

 //               cout << " dx: "<< dx << " dy: " << dy << " di: "   << di << endl;
            }
        }

#endif
    }


    /* ************************************************************************* */
    void AKAZE::MLDB_Fill_Upright_Values(float* values, int sample_step, int level,
                                 float xf, float yf, float scale) const {
#if !GPU_MEM
    int pattern_size = options_.descriptor_pattern_size;
    int nr_channels = options_.descriptor_channels;
    int valpos = 0;

    for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {

    float di = 0.0, dx = 0.0, dy = 0.0;
    int nsamples = 0;

    for (int k = i; k < i + sample_step; k++) {
    for (int l = j; l < j + sample_step; l++) {

      float sample_y = yf + l*scale;
      float sample_x = xf + k*scale;

      int y1 = fRound(sample_y);
      int x1 = fRound(sample_x);

      float ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
      di += ri;

      if(nr_channels > 1) {
        float rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
        float ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);
        if (nr_channels == 2) {
          dx += sqrtf(rx*rx + ry*ry);
        }
        else {
          dx += rx;
          dy += ry;
        }
      }
      nsamples++;
    }
    }

    di /= nsamples;
    dx /= nsamples;
    dy /= nsamples;

    values[valpos] = di;

    if (nr_channels > 1)
    values[valpos + 1] = dx;

    if (nr_channels > 2)
    values[valpos + 2] = dy;

    valpos += nr_channels;
    }
    }
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::MLDB_Binary_Comparisons(float* values, unsigned char* desc,
                                int count, int& dpos) const {

        int nr_channels = options_.descriptor_channels;

        for(int pos = 0; pos < nr_channels; pos++) {
            for (int i = 0; i < count; i++) {
                    float ival = values[nr_channels * i + pos];
                    for (int j = i + 1; j < count; j++) {
                    int res = ival > values[nr_channels * j + pos];
                    desc[dpos >> 3] |= (res << (dpos & 7));
                    dpos++;
                }
            }
        }
    }

    /* ************************************************************************* */
    void AKAZE::Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char* desc) {
#if !GPU_MEM
        float di = 0.f, dx = 0.f, dy = 0.f;
        float rx = 0.f, ry = 0.f;
        float sample_x = 0.f, sample_y = 0.f;
        int x1 = 0, y1 = 0;

        // Get the information from the keypoint
        float ratio = (float)(1<<kpt.octave);
        int scale = fRound(0.5*kpt.size/ratio);
        float angle = kpt.angle;
        float level = kpt.class_id;
        float yf = kpt.pt.y/ratio;
        float xf = kpt.pt.x/ratio;
        float co = cos(angle);
        float si = sin(angle);

        // Allocate memory for the matrix of values
        cv::Mat values = cv::Mat_<float>::zeros((4+9+16)*options_.descriptor_channels, 1);

        // Sample everything, but only do the comparisons
        vector<int> steps(3);
        steps.at(0) = options_.descriptor_pattern_size;
        steps.at(1) = ceil(2.f*options_.descriptor_pattern_size/3.f);
        steps.at(2) = options_.descriptor_pattern_size/2;

        for (int i=0; i < descriptorSamples_.rows; i++) {
            int *coords = descriptorSamples_.ptr<int>(i);
            int sample_step = steps.at(coords[0]);
            di=0.0f;
            dx=0.0f;
            dy=0.0f;

            for (int k = coords[1]; k < coords[1] + sample_step; k++) {
                for (int l = coords[2]; l < coords[2] + sample_step; l++) {

                    // Get the coordinates of the sample point
                    sample_y = yf + (l*scale*co + k*scale*si);
                    sample_x = xf + (-l*scale*si + k*scale*co);

                    y1 = fRound(sample_y);
                    x1 = fRound(sample_x);

                    di += *(evolution_[level].Lt.ptr<float>(y1)+x1);

                    if (options_.descriptor_channels > 1) {
                      rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
                      ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

                      if (options_.descriptor_channels == 2) {
                        dx += sqrtf(rx*rx + ry*ry);
                      }
                      else if (options_.descriptor_channels == 3) {
                        // Get the x and y derivatives on the rotated axis
                        dx += rx*co + ry*si;
                        dy += -rx*si + ry*co;
                      }
                    }
                }
            }

            *(values.ptr<float>(options_.descriptor_channels*i)) = di;

            if (options_.descriptor_channels == 2) {
                *(values.ptr<float>(options_.descriptor_channels*i+1)) = dx;
            }
            else if (options_.descriptor_channels == 3) {
                *(values.ptr<float>(options_.descriptor_channels*i+1)) = dx;
                *(values.ptr<float>(options_.descriptor_channels*i+2)) = dy;
            }
        }

        // Do the comparisons
        const float *vals = values.ptr<float>(0);
        const int *comps = descriptorBits_.ptr<int>(0);

        for (int i=0; i<descriptorBits_.rows; i++) {
            if (vals[comps[2*i]] > vals[comps[2*i +1]]) {
                desc[i/8] |= (1<<(i%8));
            }
        }
#endif
    }

    /* ************************************************************************* */
    void AKAZE::Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc) {
#if !GPU_MEM
    float di = 0.0f, dx = 0.0f, dy = 0.0f;
    float rx = 0.0f, ry = 0.0f;
    float sample_x = 0.0f, sample_y = 0.0f;
    int x1 = 0, y1 = 0;

    // Get the information from the keypoint
    float ratio = (float)(1<<kpt.octave);
    int scale = fRound(0.5*kpt.size/ratio);
    float level = kpt.class_id;
    float yf = kpt.pt.y/ratio;
    float xf = kpt.pt.x/ratio;

    // Allocate memory for the matrix of values
    cv::Mat values = cv::Mat_<float>::zeros((4+9+16)*options_.descriptor_channels, 1);

    vector<int> steps(3);
    steps.at(0) = options_.descriptor_pattern_size;
    steps.at(1) = ceil(2.f*options_.descriptor_pattern_size/3.f);
    steps.at(2) = options_.descriptor_pattern_size/2;

    for (int i=0; i < descriptorSamples_.rows; i++) {
    int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di=0.0f, dx=0.0f, dy=0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
    for (int l = coords[2]; l < coords[2] + sample_step; l++) {

    // Get the coordinates of the sample point
    sample_y = yf + l*scale;
    sample_x = xf + k*scale;

    y1 = fRound(sample_y);
    x1 = fRound(sample_x);
    di += *(evolution_[level].Lt.ptr<float>(y1)+x1);

    if (options_.descriptor_channels > 1) {
      rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
      ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

      if (options_.descriptor_channels == 2) {
        dx += sqrtf(rx*rx + ry*ry);
      }
      else if (options_.descriptor_channels == 3) {
        dx += rx;
        dy += ry;
      }
    }
    }
    }

    *(values.ptr<float>(options_.descriptor_channels*i)) = di;

    if (options_.descriptor_channels == 2) {
    *(values.ptr<float>(options_.descriptor_channels*i+1)) = dx;
    }
    else if (options_.descriptor_channels == 3) {
    *(values.ptr<float>(options_.descriptor_channels*i+1)) = dx;
    *(values.ptr<float>(options_.descriptor_channels*i+2)) = dy;
    }
    }

    // Do the comparisons
    const float *vals = values.ptr<float>(0);
    const int *comps = descriptorBits_.ptr<int>(0);
    for (int i=0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2*i]] > vals[comps[2*i +1]]) {
    desc[i/8] |= (1<<(i%8));
    }
    }
    #endif
    }

    /* ************************************************************************* */
    void AKAZE::Save_Scale_Space() {
#if !GPU_MEM
    cv::Mat img_aux;
    string outputFile;

    for (size_t i = 0; i < evolution_.size(); i++) {
    convert_scale(evolution_[i].Lt);
    evolution_[i].Lt.convertTo(img_aux,CV_8U,255.0,0);
    outputFile = "../output/evolution_" + to_formatted_string(i, 2) + ".jpg";
    // cv::imwrite(outputFile, img_aux);
    }
#endif
    }

    /* ************************************************************************* */
    void AKAZE::Save_Detector_Responses() {

    cv::Mat img_aux;
    string outputFile;
    float ttime = 0.0;
    int nimgs = 0;

    for (size_t i = 0; i < evolution_.size(); i++) {
    ttime = evolution_[i+1].etime-evolution_[i].etime;
    if (ttime > 0) {
    convert_scale(evolution_[i].Ldet);
    evolution_[i].Ldet.convertTo(img_aux,CV_8U,255.0,0);
    outputFile = "../output/images/detector_" + to_formatted_string(nimgs, 2) + ".jpg";
    // imwrite(outputFile.c_str(), img_aux);
    nimgs++;
    }
    }
    }

    /* ************************************************************************* */
    void AKAZE::Show_Computation_Times() const {
    cout << "(*) Time k percentile: " << timing_.kcontrast << endl;
    cout << "(*) Time Scale Space: " << timing_.scale << endl;
    cout << "(*) Time Detector: " << timing_.detector << endl;
    cout << "   - Time Derivatives: " << timing_.derivatives << endl;
    cout << "   - Time cudaAsync: " << timing_.cudaAsync << endl;
    cout << "   - Time Extrema: " << timing_.extrema << endl;
    cout << "   - Time Subpixel: " << timing_.subpixel << endl;
    cout << "(*) Time Descriptor: " << timing_.descriptor << endl;
    cout << endl;
    }

    /* ************************************************************************* */
    void libAKAZE::generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons, int nbits,
                                       int pattern_size, int nchannels) {

    int ssz = 0;
    for (int i=0; i<3; i++) {
    int gz = (i+2)*(i+2);
    ssz += gz*(gz-1)/2;
    }
    ssz *= nchannels;

    CV_Assert(nbits<=ssz && "descriptor size can't be bigger than full descriptor");

    // Since the full descriptor is usually under 10k elements, we pick
    // the selection from the full matrix.  We take as many samples per
    // pick as the number of channels. For every pick, we
    // take the two samples involved and put them in the sampling list

    cv::Mat_<int> fullM(ssz/nchannels,5);
    for (size_t i=0, c=0; i<3; i++) {
    int gdiv = i+2; //grid divisions, per row
    int gsz = gdiv*gdiv;
    int psz = ceil(2.*pattern_size/(float)gdiv);

    for (int j=0; j<gsz; j++) {
    for (int k=j+1; k<gsz; k++,c++) {
    fullM(c,0) = i;
    fullM(c,1) = psz*(j % gdiv) - pattern_size;
    fullM(c,2) = psz*(j / gdiv) - pattern_size;
    fullM(c,3) = psz*(k % gdiv) - pattern_size;
    fullM(c,4) = psz*(k / gdiv) - pattern_size;
    }
    }
    }

    srand(1024);
    cv:: Mat_<int> comps = cv::Mat_<int>(nchannels*ceil(nbits/(float)nchannels),2);
    comps = 1000;

    // Select some samples. A sample includes all channels
    int count =0;
    size_t npicks = ceil(nbits/(float)nchannels);
    cv::Mat_<int> samples(29,3);
    cv::Mat_<int> fullcopy = fullM.clone();
    samples = -1;

    for (size_t i=0; i<npicks; i++) {
    size_t k = rand() % (fullM.rows-i);
    if (i < 6) {
    // Force use of the coarser grid values and comparisons
    k = i;
    }

    bool n = true;

    for (int j=0; j<count; j++) {
    if (samples(j,0) == fullcopy(k,0) && samples(j,1) == fullcopy(k,1) && samples(j,2) == fullcopy(k,2)) {
    n = false;
    comps(i*nchannels,0) = nchannels*j;
    comps(i*nchannels+1,0) = nchannels*j+1;
    comps(i*nchannels+2,0) = nchannels*j+2;
    break;
    }
    }

    if (n) {
    samples(count,0) = fullcopy(k,0);
    samples(count,1) = fullcopy(k,1);
    samples(count,2) = fullcopy(k,2);
    comps(i*nchannels,0) = nchannels*count;
    comps(i*nchannels+1,0) = nchannels*count+1;
    comps(i*nchannels+2,0) = nchannels*count+2;
    count++;
    }

    n = true;
    for (int j=0; j<count; j++) {
    if (samples(j,0) == fullcopy(k,0) && samples(j,1) == fullcopy(k,3) && samples(j,2) == fullcopy(k,4)) {
    n = false;
    comps(i*nchannels,1) = nchannels*j;
    comps(i*nchannels+1,1) = nchannels*j+1;
    comps(i*nchannels+2,1) = nchannels*j+2;
    break;
    }
    }

    if (n) {
    samples(count,0) = fullcopy(k,0);
    samples(count,1) = fullcopy(k,3);
    samples(count,2) = fullcopy(k,4);
    comps(i*nchannels,1) = nchannels*count;
    comps(i*nchannels+1,1) = nchannels*count+1;
    comps(i*nchannels+2,1) = nchannels*count+2;
    count++;
    }

    cv::Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows-i-1).copyTo(tmp);
    }

    sampleList = samples.rowRange(0,count).clone();
    comparisons = comps.rowRange(0,nbits).clone();
    }

    /* ************************************************************************* */
    void libAKAZE::check_descriptor_limits(int &x, int &y, int width, int height) {

    if (x < 0)
    x = 0;

    if (y < 0)
    y = 0;

    if (x > width-1)
    x = width-1;

    if (y > height-1)
    y = height-1;
    }

