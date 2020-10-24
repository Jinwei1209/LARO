#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "any_matrix.h"
#include "debug.h"
// #include "emem.h"
// #define cleanup do { line = __LINE__; goto label_cleanup; } while (0)

#if defined(_WIN32) || defined(_WIN64)
#include <basetsd.h>
#define voids2ulong(x) HandleToUlong(x)
#else
#define voids2ulong(x) ((unsigned long)(x))
#endif

#define EFALSE(cond) \
do { if (!(cond)) {line=__LINE__; goto label_cleanup; }} while (0)

#define ENULL(a) EFALSE(NULL != (a))

#define  EALLOC(var, type, num)                           \
     do {                                                 \
       ENULL ((var) = (type*)malloc((num)*sizeof(type))); \
       memset((var), 0, (num)*sizeof(type));              \
     } while (0)

#define EFREE(var, extrafunc)                                 \
     do {                                                     \
       if (NULL != (var)) { extrafunc; free(var);var=NULL;}   \
     } while (0)

#define DEF_VAR

/*******************************************************************
create a matrix of any dimension. The return must be cast correctly.
********************************************************************/
void * _any_matrix(int dimension, int el_size, ...)
{
	DEF_VAR int *dims=NULL;
	int i;
	va_list ap;
	DEF_VAR void *mat=NULL;
	int line = 0;

	EFALSE(dimension > 0);
	EFALSE(el_size > 0);
	
	EALLOC(dims, int, dimension);

	/* gather the arguments */
	va_start(ap, el_size);
	for (i=0;i<dimension;i++) {
		dims[i] = va_arg(ap, int);
	}
	i = va_arg(ap, int);
	EFALSE(ANY_MARKER == i);
	va_end(ap);
	
	/* now we've disected the arguments we can go about the real
	   business of creating the matrix */

        ENULL(mat = any_matrix_2(dimension, el_size, dims));
	
label_cleanup:
	EFREE(dims, );
	if (line > 0)
	{
		EFREE(mat, );
		printf("Error in file %s on line %d\n", __FILE__, line);
	};
	return mat;
}

void * _any_matrix_realloc(void *ptr, int dimension, int el_size, ...)
{
	DEF_VAR int *dims=NULL;
	int i;
	va_list ap;
	DEF_VAR void *mat=NULL;
	int line = 0;
	
	
	EFALSE(dimension > 0);
	EFALSE(el_size > 0);
	
	EALLOC(dims, int, dimension);

	/* gather the arguments */
	va_start(ap, el_size);
	for (i=0;i<dimension;i++) {
		dims[i] = va_arg(ap, int);
	}
	i = va_arg(ap, int);
	EFALSE (ANY_MARKER == i);
	va_end(ap);
	
	/* now we've disected the arguments we can go about the real
	   business of creating the matrix */

        ENULL(mat = any_matrix_2_realloc(ptr, dimension, el_size, dims));
	
label_cleanup:
	EFREE(dims, );
	if (line > 0)
	{
		EFREE(mat, );
		printf("Error in file %s on line %d\n", __FILE__, line);
	}

	return mat;
}

void * _any_matrix_wrap(void *ptr, int dimension, int el_size, ...)
{
	DEF_VAR int *dims=NULL;
	int i;
	va_list ap;
	DEF_VAR void *mat=NULL;
	int line=0;
	
	EFALSE(dimension > 0);
	EFALSE(el_size > 0);
	
	EALLOC(dims, int, dimension);

	/* gather the arguments */
	va_start(ap, el_size);
	for (i=0;i<dimension;i++) {
		dims[i] = va_arg(ap, int);
	}
	i = va_arg(ap, int);
	EFALSE(ANY_MARKER == i);
	va_end(ap);
	
	/* now we've disected the arguments we can go about the real
	   business of creating the matrix */

        ENULL(mat = any_matrix_2_wrap(ptr, dimension, el_size, dims));

label_cleanup:
	EFREE(dims, );
	if (line > 0)
	{
		EFREE(mat, );
		printf("Error in file %s on line %d\n", __FILE__, line);
	}
	return mat;
}

/* padding depends also on ptr to ensure 16 byte alignment of the
   actual data
 */  
unsigned long any_matrix_data_offset(void *ptr, int dimension, int el_size, int *dims)
{
	int i, j;
	unsigned long ptr_size,prod, padding;
	
	if (dimension <= 0) return(-1);
	if (el_size <= 0) return(-1);
	
	/* calculate how much space all the pointers will take up */
	ptr_size = 0;
	for (i=0;i<(dimension-1);i++) {
		prod=sizeof(void *);
		for (j=0;j<=i;j++) prod *= dims[j];
		ptr_size += prod;
		//showi(ptr_size);
	}

	/* padding overcomes potential alignment errors */
	//padding = (el_size - (ptr_size % el_size)) % el_size;
	padding =  (ANY_ALIGN -
              ((voids2ulong(ptr) + ptr_size)%ANY_ALIGN)
             ) % ANY_ALIGN;

  return ptr_size + padding;
}

void *any_matrix_2(int dimension, int el_size, int *dims) {
  return any_matrix_2_realloc(NULL, dimension, el_size, dims);
}

void *any_matrix_2_realloc(void *ptr, int dimension, int el_size, int *dims)
{
	void **mat;
	int i, j, ppos;
	unsigned long k, ptr_size, size, prod, padding;
	void *next_ptr;
	
	if (dimension <= 0) return(NULL);
	if (el_size <= 0) return(NULL);
	
	/* calculate how much space all the pointers will take up */
	ptr_size = 0;
	for (i=0;i<(dimension-1);i++) {
		prod=sizeof(void *);
		for (j=0;j<=i;j++) prod *= dims[j];
		ptr_size += prod;
	}

	/* padding overcomes potential alignment errors */
	//padding = (el_size - (ptr_size % el_size)) % el_size;
	//initial size for padding is elsize extra
  padding = ANY_ALIGN;
	
        //printf("padding = %l\n", padding);
	
	/* now calculate the total memory taken by the array */
	prod=el_size;
	for (i=0;i<dimension;i++) prod *= dims[i];
	size = prod + ptr_size + padding;
        //showl(ptr_size);
	//showl(padding);
	//showl(size);
	/* allocate the matrix memory */
	mat = (void **)realloc(ptr, size);

	if (mat == NULL) {
		fprintf(stdout,"Error allocating %d dim matrix of size %ld\n",dimension,size);
		return(NULL);
	}

  //showl(ANY_ALIGN);
  //showl(voids2ulong(mat));
  //showl(ptr_size);
  /* final size of padding */
  padding =  (ANY_ALIGN -
              ((voids2ulong(mat) + ptr_size)%ANY_ALIGN)
             ) % ANY_ALIGN;
  //showl(padding);

	/* now fill in the pointer values */
	next_ptr = (void *)&mat[dims[0]];
	ppos = 0;
	prod = 1;
	for (i=0;i<(dimension-1);i++) {
		int skip;
		if (i == dimension-2) {
			skip = el_size*dims[i+1];
			next_ptr = (void *)(((char *)next_ptr) + padding); /* add in the padding */
		} else {
			skip = sizeof(void *)*dims[i+1];
		}

		for (k=0;k<(dims[i]*prod);k++) {
			mat[ppos++] = next_ptr;
			next_ptr = (void *)(((char *)next_ptr) + skip);
		}
		prod *= dims[i];
	}

	return((void *)mat);
}

void *any_matrix_2_wrap(void *ptr, int dimension, int el_size, int *dims)
{
	void **mat;
	int i, j, ppos;
	unsigned long k, ptr_size, prod;
	void *next_ptr;
	
	if (dimension <= 0) return(NULL);
	if (el_size <= 0) return(NULL);
	
	/* calculate how much space all the pointers will take up */
	ptr_size = 0;
	for (i=0;i<(dimension-1);i++) {
		prod=sizeof(void *);
		for (j=0;j<=i;j++) prod *= dims[j];
		ptr_size += prod;
	}

	mat = (void **)malloc(ptr_size);

	if (mat == NULL) {
		fprintf(stdout,"Error allocating %d dim pointer space of size %ld\n",dimension,ptr_size);
		return(NULL);
	}

	/* now fill in the pointer values */
	next_ptr = (void *)&mat[dims[0]];
	ppos = 0;
	prod = 1;
	for (i=0;i<(dimension-1);i++) {
		int skip;
		if (i == dimension-2) {
			skip = el_size*dims[i+1];
			next_ptr = ptr;
		} else {
			skip = sizeof(void *)*dims[i+1];
		}

		for (k=0;k<(dims[i]*prod);k++) {
			mat[ppos++] = next_ptr;
			next_ptr = (void *)(((char *)next_ptr) + skip);
		}
		prod *= dims[i];
	}

	return((void *)mat);
}




/*
  double ****x;
  double ***y;

  x = (double ****)any_matrix(4, sizeof(double), 3, 6, 8, 2);

  x[0][3][4][1] = 5.0;

  y = x[1];

  y[2][3][4] = 7.0;

  free(x);
*/
