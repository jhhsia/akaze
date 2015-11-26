
#pragma once

void ScharrFilerCuda( const float* src, float* dst, int width, int height, int direction  );
void PM_G2Cuda( const float* lx, const float* ly, float* dst, int width, int height, float k_percent  );
void GaussianCUDA5x5( const float* src, float* dst, float sigma, int width, int height);
void GaussianCUDA5x5TwoPass( const float* src, float* interm, float* dst, float sigma, int width, int height);
void GaussianCUDA9x9( const float* src, float* dst, float sigma, int width, int height);
void GaussianCUDA9x9TwoPass( const float* src, float* interm, float* dst, float sigma, int width, int height);
void NLDStepScalarCUDA(const float* src, const float* stLd, float* dlt, float stepSize, int width, int height);
void DownSize2x2AreaFastCUDA( const float* src, float* dst, int width, int height  );
void SharrDerivativeCUDA(const float* src, float* dst, float mkernel,
                         float ekernel, int direction,int step, int width, int height);

void DeterminantHessianCUDA(const float* lxx, const float* lyy, const float* lxy, float* det, float sigma_quad, int width, int height);