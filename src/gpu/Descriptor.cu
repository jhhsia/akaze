#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "helper_device_thread.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "ImageOperation.h"

/// This function computes the value of a 2D Gaussian function
__constant__ float gauss25[7][7] = {
        {0.02546481f,	0.02350698f,	0.01849125f,	0.01239505f,	0.00708017f,	0.00344629f,	0.00142946f},
        {0.02350698f,	0.02169968f,	0.01706957f,	0.01144208f,	0.00653582f,	0.00318132f,	0.00131956f},
        {0.01849125f,	0.01706957f,	0.01342740f,	0.00900066f,	0.00514126f,	0.00250252f,	0.00103800f},
        {0.01239505f,	0.01144208f,	0.00900066f,	0.00603332f,	0.00344629f,	0.00167749f,	0.00069579f},
        {0.00708017f,	0.00653582f,	0.00514126f,	0.00344629f,	0.00196855f,	0.00095820f,	0.00039744f},
        {0.00344629f,	0.00318132f,	0.00250252f,	0.00167749f,	0.00095820f,	0.00046640f,	0.00019346f},
        {0.00142946f,	0.00131956f,	0.00103800f,	0.00069579f,	0.00039744f,	0.00019346f,	0.00008024f}
};


__global__ void ComputeMainOrientation(const DspKPCUDAEntry* kptAngle,  unsigned int* out  )
{

    int   indx  = GetGlobalIdx();
    float size  = kptAngle[indx].size;
    float ratio = kptAngle[indx].ratio;
    float pt_x  = kptAngle[indx].pt_x;
    float pt_y  = kptAngle[indx].pt_y;
    int width = kptAngle[indx].imgWidth;
    const float* lx = kptAngle[indx].lx;
    const float* ly = kptAngle[indx].ly;
    const float* lt = kptAngle[indx].lt;

    //float* desp = out + (indx*48);

    int ix = 0, iy = 0, idx = 0, s = 0;
    float xf = 0.0, yf = 0.0, gweight = 0.0;
    float resX[109], resY[109], Ang[109];
    const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};
    // Variables for computing the dominant direction
    float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

    // Get the information from the keypoint
    //level = kpt.class_id;
   // ratio = (float)(1<<evolution_[level].octave);
    s = floorf( (0.5*size/ratio) + 0.5);
    xf = pt_x/ratio;
    yf = pt_y/ratio;

    //debug debug debug debug ------

 //   printf(" ===========  MLDB ================ \n");
   // printf(" pt_x: %f, pt_y %f, ratio: %f \n", pt_x, pt_y, ratio );

    // Calculate derivatives responses for points within radius of 6*scale
    for (int i = -6; i <= 6; ++i) {
        for (int j = -6; j <= 6; ++j) {
            if (i*i + j*j < 36) {
                iy = floorf(yf + j*s + 0.5);
                ix = floorf(xf + i*s + 0.5);

                gweight = gauss25[id[i+6]][id[j+6]];
                resX[idx] = gweight*( lx[iy*width + ix] );
                resY[idx] = gweight*( ly[iy*width + ix] );
                float tan_angle = atan2f(resY[idx], resX[idx]);
                Ang[idx]  = tan_angle < 0.0f ? tan_angle + M_PI*2.0f : tan_angle;
                ++idx;
            }
        }
    }

    // Loop slides pi/3 window around feature point
    float kpt_angle = 0.0f;
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
            float tan_angle = atan2f(sumY, sumX);

            kpt_angle =  tan_angle < 0.0f ? tan_angle + M_PI*2.0f : tan_angle;

        }
    }

    const int max_channels = 3;
    //CV_Assert(options_.descriptor_channels <= max_channels);
    float values[16*max_channels];
    // init to zero , we need this?
    unsigned char bin_des[64];
#if 1

    {
        unsigned int* alias = (unsigned int*)bin_des;
        for(int i = 0; i < 64/sizeof(unsigned int); ++i)
        {
         //   printf("  values %d  \n",alias[i]);
            alias[i] = 0;
        }
    }
#endif

    const float size_mult[3] = {1, 2.0/3.0, 1.0/2.0};
    float co = cosf(kpt_angle);
    float si = sinf(kpt_angle);

    //TODO: controled from option!!
    int pattern_size = 10;
    int nr_channels = 3;

    int dpos = 0;
    for(int lvl = 0; lvl < 3; lvl++) {

        int val_count = (lvl + 2) * (lvl + 2);
        int sample_step = (int)(ceil(pattern_size * size_mult[lvl]));

        //    cout << "=========== one MLDB ================ " << endl;
        //    cout << "xf: " << xf << " yf:" << xf << " level: " <<  level << endl;
        int valpos = 0;
        for (int i = -pattern_size; i < pattern_size; i += sample_step) {
            for (int j = -pattern_size; j < pattern_size; j += sample_step) {

                float di = 0.0, dx = 0.0, dy = 0.0;
                int nsamples = 0;

                for (int k = i; k < i + sample_step; k++) {
                    for (int l = j; l < j + sample_step; l++) {

                        float sample_y = yf + (l*co*s + k*si*s);
                        float sample_x = xf + (-l*si*s + k*co*s);

                        int y1 = floorf(sample_y + 0.5f);
                        int x1 = floorf(sample_x + 0.5f);

                        float ri = lt[y1*width+x1];

                        //printf(" ComputeMainOrientation %f\n", ri);
                        //float ri = 1.0f;
                        if( !isnan(ri) )
                        {
                            float rx = lx[y1*width+x1];
                            float ry = ly[y1*width+x1];
                            if( !isnan(rx) && !isnan(ry) )
                            {
                                di += ri;
                                float rry = rx*co + ry*si;
                                float rrx = -rx*si + ry*co;
                                dx += rrx;
                                dy += rry;

                                nsamples++;
                            }
                        }
                    }
                }

                di /= nsamples;
                dx /= nsamples;
                dy /= nsamples;

                values[valpos] = di;
                values[valpos + 1] = dx;
                values[valpos + 2] = dy;

                valpos += nr_channels;
               // if(indx == 331)
               // {
               //     printf("dx: %f, dy: %f, di: %f \n", dx, dy, di);
               // }
            }
        }


        for(int pos = 0; pos < nr_channels; pos++) {
            for (int k = 0; k < val_count; k++) {
                float ival = values[nr_channels * k + pos];
                for (int j = k + 1; j < val_count; j++) {
                    int res = ival > values[nr_channels * j + pos];

                    bin_des[ dpos >> 3 ] |= (res << (dpos & 7));
                    //out[ indx*64 + (dpos >> 3) ] |= (res << (dpos & 7));
                    dpos++;
                }
            }
        }

#if 0
        if(indx == 331)
        {
            for(int p = 0; p < 48; ++p)
            {

                printf("  values %f  \n",values[p]);
                //  printf(" -- values %f - %f\n", dx, g_dx);
                //  printf(" -- values %f - %f\n", dy, g_dy);
            }
        }

#endif

    }

    const unsigned int* src = (const unsigned int*)bin_des;
   // unsigned int* dst = (unsigned int*)out;
    int offset = indx * 64/sizeof(unsigned int);
    for(int i = 0; i < 64/sizeof(unsigned int); ++i)
    {
        //   printf("  values %d  \n",alias[i]);
        out[offset+i] = src[i];
    }

    // kptAngle[indx].angle =  values[0];
}


void ComputeMainOrientationGPU( DspKPCUDAEntry* kptAngle,  unsigned char* out, int count )
{

    ComputeMainOrientation<<< 2, count/2 >>>( kptAngle , (unsigned int*)out);
}