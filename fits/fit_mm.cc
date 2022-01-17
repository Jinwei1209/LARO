#define NOMINMAX
#include <iostream>
#include <string.h>
#include <fftw3.h>
#include <omp.h>
#include <vector>
#include "Util.hh"
#include "util_blas.h"
#include "any_matrix.h"
#include "angle.h"
#include <fit_mm.h>
#include "mkl_lapacke.h"
using namespace std;


int unwrap1d( float *phase, long numel, long numte)
{
  long i,j,k;   /* counters */
  float c;   /* phase difference */
  float cd;  /* estimate of phase slope  */

  cout << "unwrap1d!";
  for(i=0; i<numel; i++)
  {
    c = phase[i*numte+1] - phase[i*numte];
    if (c > pi) 
      c = c - 2*pi;
    if (c < -pi)
      c = c + 2*pi;


    for( j=0; j<numte-1; j++)  
    {
      cd = (phase[i*numte + j + 1]-phase[i*numte + j]) - c;  


      if (cd<-pi)
        for (k=j+1; k<numte; k++)
          phase[i*numte+k] = phase[i*numte+k] + fabsf(truncf( (cd-pi)/(2.0f*pi) ))*2.0f*pi;


      if (cd>pi)
        for (k=j+1; k<numte; k++)
          phase[i*numte+k] = phase[i*numte+k] - fabsf(truncf( (cd+pi)/(2.0f*pi) ))*2.0f*pi;

    }
  }
  return 0;
}


float nonlin_fit(long numte, const float *data, float* wp, float tol, int max_iter)
{
  int num_unknown = 2;
  int iter=0;
  float p[3];
  float dp[3];
  memcpy(p, wp, num_unknown*sizeof(float)); 
  memcpy(dp, p, num_unknown*sizeof(float)); 
  int FREQ=num_unknown-1;

  float *A;    /* linear regression matrix    */
  float *res;    /* residual between observation and model */

  //int i,j;    /* counter variables  */
  float echo_mag;

  //CBLAS_ORDER order;
  //CBLAS_TRANSPOSE transA, transB;

  int nrhs, lwork, info,  M, N; //, MN;
  float *work=NULL;
  char trans = 'N';

  /*allocate memory spaces for variables  */
  res = (float *)  calloc(numte * 2 , sizeof(float));      /* 2 is for complex numbers */
  A = (float *)    calloc(numte*num_unknown*2 , sizeof(float));    /* 2 is for complex number */

  /****************************************************************************************/

  /*initialize parameters  */

  M = numte;
  N = num_unknown;         
  //MN =  numte;
  nrhs = 1;
  //lwork = MN + MN*3;
  //work = (float *)  calloc(lwork*2, sizeof(float));  /* allocate working memory, 2 is for complex number  */
  lwork = -1;

  while ((fabsf(dp[FREQ])>fabsf(p[FREQ]*tol)) && (iter<max_iter))
  {
    iter = iter+1;
    for(long j=0; j<numte; j++)
    {
      echo_mag = sqrt(data[j*2+1]*data[j*2+1] + data[j*2+0]*data[j*2+0]);
      //res[j*2+0] = data[j*2+0]-echo_mag*cosf(p*j+p0);  res[j*2+1] = data[j*2+1]-echo_mag*sinf(p*j+p0);

      /* Based on the following calculation

         A[(0*numte + j)*2+0] = 0;  A[(0*numte + j)*2+1] = 1;
         A[(1*numte + j)*2+0] = 0;  A[(1*numte + j)*2+1] = j;
         W[(j*numte + j)*2+0] = echo_mag*cosf(p*j+p0);  W[(j*numte + j)*2+1] = echo_mag*sinf(p*j+p0);

       */
      float a = p[1]*j+p[0], ec = echo_mag*cosf(a), es = echo_mag*sinf(a); 
      res[j*2+0] = data[j*2+0]-ec;  res[j*2+1] = data[j*2+1]-es;
      A[(0*numte + j)*2+0] = -es;    A[(0*numte + j)*2+1] = ec;
      A[(1*numte + j)*2+0] = -es*j;  A[(1*numte + j)*2+1] = ec*j;

      //A[(0*numte + j)*2+0] = -echo_mag*sinf(p*j+p0);    A[(0*numte + j)*2+1] = echo_mag*cosf(p*j+p0);
      //A[(1*numte + j)*2+0] = -echo_mag*sinf(p*j+p0)*j;    A[(1*numte + j)*2+1] = echo_mag*cosf(p*j+p0)*j;


    }      

    //order = CblasColMajor;      /* column major  */  
    //transA = CblasNoTrans;       /* no transpose  */
    //transB = CblasNoTrans;       /* no transpose  */


    if (-1==lwork) {
      float optlwork[2];
      cgels_(&trans, &M,     &N,     &nrhs, (MKL_Complex8 *) A,  &M,     (MKL_Complex8 *) res,     &M,  (MKL_Complex8 *)&optlwork, &lwork, &info);
      lwork=(int)optlwork[0];
      //PRINT(info);
      //PRINT(lwork);
      work = (float *)  calloc(lwork*2, sizeof(float));  /* allocate working memory, 2 is for complex number  */
    }

    cgels_(&trans, &M,     &N,     &nrhs, (MKL_Complex8 *) A,  &M,     (MKL_Complex8 *) res,     &M,   (MKL_Complex8 *) work, &lwork, &info);
    for(int k=0; k<num_unknown; k++) {
      dp[k] = res[k*2+0];
      p[k] += dp[k];
    }
    //dp0 = res[0*2+0];
    //dp = res[1*2+0];
    //p = p+dp;
    //p0 = p0+dp0;
  }

  if (NULL != work) {free(work);  work = NULL;}
  free(res);     res = NULL;
  free(A);    A = NULL;

  return p[FREQ];
}



float noise_estimate(long numte, const float *data)
{
  float mag_sum =0;
  float t_mag_sum = 0;
  float t2_mag_sum = 0;
  float echo_mag;
  float noise;

  for (long t = 0; t<numte; t++)
  {
    echo_mag = sqrt(data[t*2+1]*data[t*2+1] + data[t*2+0]*data[t*2+0]);
    if (echo_mag==0)
      return 0;
    mag_sum = mag_sum + echo_mag*echo_mag;
    t_mag_sum = t_mag_sum + t*echo_mag*echo_mag;
    t2_mag_sum = t2_mag_sum + t*t*echo_mag*echo_mag;
  }

  if   ((mag_sum*t2_mag_sum - t_mag_sum*t_mag_sum)>0)
  {
    noise = mag_sum/(mag_sum*t2_mag_sum - t_mag_sum*t_mag_sum);
    noise = sqrt(noise);
  }
  else
    noise = 0;

  return noise;  
}


int fit_ge( float *s, const int *mask, long numte, long numel, float *field, float *noise_prop)
{
  //float *A;  /* linear regression matrix  */
  //float *W;  /* weights  */
  //float *WA;  /* weighted A */
  float *phase;  /* phases of input signal  */

  //unsigned int i,j;  /* counter variables  */
  long te4use;
  long total_points = 0;
  long points = 0;
  int num_unknown = 2;

  /*allocate memory spaces for variables  */
  phase = (float *)  calloc(numel * numte , sizeof(float));  
  //A = (float *)  calloc(numte*num_unknown , sizeof(float));  
  //W = (float *)  calloc(numte*numte , sizeof(float));  
  //WA = (float *)  calloc(numte*num_unknown , sizeof(float));  

  /****************************************************************************************/

  /*initialize parameters  */
  for(long i=0; i<numel * numte; i++)
    phase[i] = angle(&s[i*2]);

  unwrap1d(phase, numel, numte);

  /* start frequency searching  */
  /* LAPACK initializations  */
  {
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA, transB;

    int nrhs, lwork, info,  M, N, MN;
    //float *work;
    char trans = 'N';
    //float *wp;  /* weighted phase */ 

    //wp = (float *) calloc(numte, sizeof(float));

    if (numte>2)
      te4use =3;
    else
      te4use = 2;
    
    M = te4use;  
    N = num_unknown;     
    MN = te4use;
    nrhs = 1;
    lwork = MN + MN*3;
    order = CblasColMajor;  /* column major  */  
    transA = CblasNoTrans;   /* no transpose  */
    transB = CblasNoTrans;   /* no transpose  */

    for (long i=0; i<numel; i++)
      if (mask[i])
        total_points = total_points+1; 

#pragma omp parallel 
  {
    std::vector<float> A(numte*num_unknown);  
    std::vector<float> W(numte*numte);  
    std::vector<float> WA(numte*num_unknown);  
    std::vector<float> work(lwork);  /* allocate working memory  */
    std::vector<float> wp(numte);
  
#pragma for    
    for (long i= 0; i<numel; i++)
      if (mask[i])  
      {
        points = points+1;

        for (long j=0;j<numte*numte;j++) W[j]=0;

        for(long j=0; j<te4use; j++)
        {
          A[0*te4use + j] = 1;
          A[1*te4use + j] = j;
          float echo_mag = sqrt(s[ (i*numte + j)*2+1]*s[ (i*numte + j)*2+1] + s[(i*numte + j)*2+0]*s[(i*numte + j)*2+0]);
          W[j*te4use + j] = echo_mag;  /* weight the matrix */  
          wp[j] = phase[i*numte+j]*echo_mag;  /* weight the phase */
        }  

        cblas_sgemm(order, transA, transB, M, N, M, 1, &W[0], M, &A[0], M, 0, &WA[0], M); 

        sgels_(&trans, &M,   &N,   &nrhs, &WA[0],  &M,   &wp[0],   &M,   &work[0], &lwork, &info);
        field[i] = wp[N-1];

        field[i] = nonlin_fit(numte, &s[i*numte*2], &wp[0], 1e-4, 10);
        noise_prop[i] = noise_estimate(numte, &s[i*numte*2]);
      }

    //free(wp);  wp = NULL;
    //free(work);  work = NULL;
  }
  } /*End of using LAPACK */

  /* clean up memory  */
  //free(A);  A = NULL;
  //free(WA);  WA = NULL;
  //free(W);  W = NULL;
  free(phase);  phase = NULL;
  /****************************/
  return 0;
}
