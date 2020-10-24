#ifndef _ANGLE_H_
#define _ANGLE_H_
#include <complex>
#include <fftw3.h>
#define pi 3.141592653589793238462643383279

float angle(fftwf_complex sig);
float angle(const std::complex<float>& sig);
float abs(fftwf_complex sig);

#endif
