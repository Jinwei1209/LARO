#ifndef __FIT_MM_H__
#define __FIT_MM_H__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#define _USE_MATH_DEFINES
#include <math.h>

int unwrap1d( float *phase, long numel, long numte);

/*
	Estimate field Map using nonlinear fitting
	numte -- number of TEs
	ncoil -- number of coils
	numel -- number of elements in s
	field -- output field map
	noise_est -- estimated noise level for each voxel
*/
int fit_ge( float *s, const int *mask, long numte, long numel, float *field, float *noise_est);

#endif /*__FIT_MM_H__*/
