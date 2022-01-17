#ifndef UTIL_BLAS_H_
#define UTIL_BLAS_H_

#include <mkl.h>
#include <string.h>
#include <complex>

inline void cblas_saxpby(const MKL_INT n, float *dest, float a, float *x, float b, float *y)
{
  const MKL_INT inc = 1;
  cblas_scopy (n, y, inc, dest, inc);  
  if (1 != b) cblas_sscal (n, b, dest, inc);
  cblas_saxpy (n, a, x, inc, dest, inc);
}

inline void cblas_saxpby(const MKL_INT n, std::complex<float> *dest, float a, float *x, float b, float *y)
{
  // slowest
  //TIMING(memset(dest, 0, n*sizeof(std::complex<float>)););
  // faster
  //TIMING(std::fill((float*)dest, ((float*)dest)+2*n,0.0););   
  // fastest is multiplying imaginary part with zero
  cblas_sscal (n, 0.0f, ((float*)dest)+1, 2);
  cblas_scopy(n, y, 1, (float*)dest, 2);  
  if (1 != b) cblas_sscal (n, b, (float*)dest, 1);
  cblas_saxpy (n, a, x, 1, (float*)dest, 2);
}
 
inline void cblas_threshold(const MKL_INT n, float threshold, float *src, float *dest, float *mask)
{
  float max_src = src[cblas_isamax(n, src, 1)];
  if (0==max_src) {
		if (NULL!=dest) cblas_scopy(n, src, 1, dest, 1);
		return; 
	}
  vsLinearFrac(n, src, src, 1.0f/max_src, -threshold/max_src, 0.0f, 1.0f, mask);
  // this works but is 2-3 times slower
  //float val=-threshold/max_src;
  //TIMING(
  //cblas_scopy(n, &val, 0, mask, 1);
  //cblas_saxpy(n, 1.0/max_src, src, 1, mask, 1);
  //);  
  vsCeil(n, mask, mask);
  if (NULL != dest) vsMul(n, src, mask, dest);
}

inline void cblas_threshold_frac(const MKL_INT n, float threshold_frac, float *src, float *dest, float *mask)
{
  float max_src = src[cblas_isamax(n, src, 1)];
  if (0==max_src) {
		if (NULL!=dest) cblas_scopy(n, src, 1, dest, 1);
		return; 
	}
  vsLinearFrac(n, src, src, 1.0f/max_src, -threshold_frac, 0.0f, 1.0f, mask);
  // this works but is 2-3 times slower
  //float val=-threshold/max_src;
  //TIMING(
  //cblas_scopy(n, &val, 0, mask, 1);
  //cblas_saxpy(n, 1.0/max_src, src, 1, mask, 1);
  //);  
  vsCeil(n, mask, mask);
  if (NULL != dest) vsMul(n, src, mask, dest);
}

// add a scalar, uses zero stride trick
inline void cblas_sapy(const MKL_INT n, float a, float *y, MKL_INT inc)
{
  cblas_saxpy(n, 1.0f, &a, 0, y, inc);
}

inline float cblas_sum(const MKL_INT n, const float *y, MKL_INT inc)
{
  float val = 1.0f;
  return cblas_sdot(n, y, inc, &val, 0);
}
 
//inline void vcMuls(const MKL_INT n, MKL_Complex8 *a, float* b, MKL_Complex8* result)
//{
//  // set result to zero
//  cblas_sscal (2*n, 0.0, (float*)result, 1);
//  cblas_scopy(n, b, 1, (float*)result, 2);  
//  
//vcMul(m_lens, (MKL_Complex8*)numer,(MKL_Complex8*)denom, (MKL_Complex8*)ctmp);

#endif
