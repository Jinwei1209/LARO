#include "angle.h"
#include <math.h>
float angle(fftwf_complex sig)
{
  float phs;

  if (sig[0]>0) {
    phs = ::atan ( sig[1]/sig[0] );
  }
  else if (sig[0] ==0)
  {
    if (sig[1]>0)
      phs = (float)pi/2;
    else
      phs = -(float)pi/2;
  }
  else
  {
    if (sig[1]>0)
      phs = (float)pi + ::atan ( sig[1]/sig[0] );
    else if (sig[1]<0)
      phs = ::atan ( sig[1]/sig[0] )-(float)pi;
    else
      phs = -(float)pi;
  }

  return phs;
}

float angle(const std::complex<float>& sig)
{
  return angle((float*)&sig);
}

float abs(fftwf_complex sig)
{
  float mag;
  mag = sqrtf(sig[0]*sig[0]+sig[1]*sig[1]);
  return mag;
}

