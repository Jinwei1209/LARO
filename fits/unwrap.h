#ifndef __UNWRAP_H__
#define __UNWRAP_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


#define PI 3.1415926535897931f
//#define min(a,b) ((a)<(b)?(a):(b))
//#define max(a,b) ((a)>(b)?(a):(b))


struct MRI_point {
    long pos[3];
	long ind;
	float mag;
	float phase;
	long planned;
	long pre;
};


 

int unwrap(struct MRI_point *, long,  long *, float);
int Laplacian_unwrap(float *, long*, float *);
struct MRI_point * assemble_data( float *, float *, long *, long *);

float fround(float num);

void get_neighbors(long *,long *,long *);

long * heap_create(long, long * );
void heap_add(struct MRI_polong *, long *, long, long *);
long heap_pop(struct MRI_polong *, long *, long *);

void swap(long *, long *);
int qualityguidedunwrapping(float *mag, float *phase, long* matrix_size, float unwrap_noise_ratio);

#ifdef __cplusplus
}
#endif
#endif /*__UNWRAP_H__*/

