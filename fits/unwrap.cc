/*=================================================================
 * 3d unwrapping algorithm
 * Similar to R. Cusack, N. Papadakis NeuroImage 16, 754 - 764 (2002)
 * Use intensity guided region growing algorithm;
 * Starts from a seed struct MRI_point, grow(unwrap) the edge. 
 * Always grow the edge struct MRI_point with the greatest intensity. 
 * Algorithm is slow
 *=================================================================*/
/* $Revision: 1.5.6.2 $ */
#include "unwrap.h"
#include "laplacian.h"
#define INVALID ((long)(-1))

int Laplacian_unwrap(float* phase, long *matrix_size, float* voxel_size)
{
	long m_lens = matrix_size[0]*matrix_size[1]*matrix_size[2];
	float *first_term = new float[m_lens]();
	float *second_term = new float[m_lens]();
	for (long i=0;i<m_lens;i++)
	{
		first_term[i] = sinf(phase[i]);
		second_term[i] = cosf(phase[i]);
	}
	Laplacian* L =  new Laplacian(matrix_size[0], matrix_size[1], matrix_size[2], voxel_size); 	
	L->GenLpKernel();
	L->Conv(first_term, first_term);
	L->Conv(second_term, second_term);

	for (long i=0;i<m_lens;i++)
	{
		first_term[i] = cosf(phase[i])*first_term[i];
		second_term[i] = sinf(phase[i])*second_term[i];
		phase[i] = first_term[i] - second_term[i];	
	}
	L->Deconv(phase, phase);

	delete L;
	delete [] first_term; first_term = NULL;
	delete [] second_term; second_term = NULL;
	return 0;
}	

void swap(long *a, long *b)
{
long c;
	c = *a;
	*a = *b;
	*b = c;
}

struct MRI_point* assemble_data( float *mag, float *phase, long *dim, long *max_ind)
{
long ind, numel;

struct MRI_point * ret_image;

	numel = dim[0]*dim[1]*dim[2];
	ret_image = (struct MRI_point *) calloc(numel, sizeof(struct MRI_point));

	for (ind=0; ind< numel; ind++)
	{
    	ret_image[ind].pos[2]=(long)((ind)/(dim[1]*dim[0]));
    	ret_image[ind].pos[1]=(long)((ind - ret_image[ind].pos[2] *dim[1]*dim[0])/dim[0]);
    	ret_image[ind].pos[0]=(long)(ind - ret_image[ind].pos[1]*dim[0] - ret_image[ind].pos[2]*dim[1]*dim[0]);
		ret_image[ind].ind = ind;
		ret_image[ind].mag = mag[ind];
		if (phase != NULL)
			ret_image[ind].phase = phase[ind];
		ret_image[ind].planned = 0;
		ret_image[ind].pre = 0;
		if (mag[ind] > mag[*max_ind]) *max_ind = ind;
	}	
	
	return ret_image;

}






void get_neighbors(long *nei,long *pos,long *dim)
{
    
    long x,y,z;


	x = pos[0];
	y = pos[1];
	z = pos[2];

	if(x>0)
        nei[0]=x-1+y*dim[0]+z*dim[0]*dim[1];
    else
        nei[0]=INVALID;
    
    if (x<(dim[0]-1))
        nei[1]=x+1+y*dim[0]+z*dim[0]*dim[1];
    else
        nei[1]=INVALID;

    if (y>0)
        nei[2]=x+(y-1)*dim[0]+z*dim[0]*dim[1];
    else
        nei[2]=INVALID;
    
    if (y<(dim[1]-1))
        nei[3]=x+(1+y)*dim[0]+z*dim[0]*dim[1];
    else
        nei[3]=INVALID;

    if (z>0)
        nei[4]=x+y*dim[0]+(z-1)*dim[0]*dim[1];
    else
        nei[4]=INVALID;
    
    if (z<(dim[2]-1))
        nei[5]=x+y*dim[0]+(z+1)*dim[0]*dim[1];
    else
        nei[5]=INVALID;

}






float fround(float num)
{
    if (num>=0)
    {
        if ((num-floorf(num))>=0.5f)
            return floorf(num)+1;
        else
            return floorf(num);
    }
    else
    {
        if ((num-floorf(num))<=0.5f)
            return floorf(num);
        else 
            return floorf(num)+1;
    }
}



long * heap_create(long seed, long * dim)
{
long numel;
long *heap;
	numel = dim[0]*dim[1]*dim[2];
	heap = (long *) calloc(numel, sizeof(long));
	heap[0] = seed;
	return heap;
}


void heap_add(struct MRI_point *ori_image, long * heap, long index, long * length)
{
long ind, parent, goon;
	
	heap[*length] = index;
	goon = 1;
	
	ind = *length;
	while ((ind>0) && (goon))
	{
		parent = (ind-1)/2;
	
		if (ori_image[heap[parent]].mag < ori_image[heap[ind]].mag) 
		{
			swap( &heap[parent], &heap[ind]);
			ind = parent;
		}
		else
		{
			goon = 0;
		}
	}
	*length = *length + 1;
}


long heap_pop( struct MRI_point * ori_image, long * heap, long * length)
{
long left, right, ind, goon, child;
long ret_val;	
	*length = *length - 1;
	ret_val = heap[0];
	heap[0] = heap[*length];
	heap[*length] = 0;
	
	ind = 0;
	goon = 1;
	while ((ind*2+1<= *length - 1) && (goon))
	{
		left = ind * 2 +1;
		right = ind*2 +2;
		if ( ori_image[heap[left]].mag > ori_image[heap[right]].mag )
			child = left;
		else
			child = right;
		if (child > (*length - 1)) child = left;
		
		if (ori_image[heap[ind]].mag < ori_image[heap[child]].mag)
		{
			swap( &heap[ind], &heap[child]);
			ind = child;
		}
		else
		{
			goon = 0;
		}
	}
	return ret_val;
}


int unwrap(struct MRI_point *ori_image, long seed, long *dim, float noise)
{
long nei[6];
long curr_ind, counter;
long i;
long * heap;
long heap_length = 0;

	counter = 0;
	curr_ind = seed;

	/* beginning of the searching algorithm	*/
    
	heap = heap_create(seed, dim);
	heap_length++;
	ori_image[heap[0]].pre = INVALID;
	ori_image[heap[0]].planned = 1;
		
	do
	{

		counter ++; 
/*		
		if (ori_image[heap[0]].phase < 10)
			ori_image[heap[0]].phase = 10;
		else 
			ori_image[heap[0]].phase = ori_image[heap[0]].phase + 10;
*/		

	
//Original code. Compare the difference with 1 previous voxel	
		if (INVALID != ori_image[heap[0]].pre) 
			ori_image[heap[0]].phase = ori_image[heap[0]].phase - fround( ( ori_image[heap[0]].phase - ori_image[ ori_image[heap[0]].pre ].phase )/(2*PI) )   *2*PI;


/*2011-12-12 LJQ Replace the above codes with the following codes. Use 3 previous points to improve robustness********/
/*
	if ( (ori_image[heap[0]].pre >=0) && (ori_image[ori_image[heap[0]].pre].pre >=0) && (ori_image[ori_image[ori_image[heap[0]].pre].pre].pre) >=0)
	{
		if(fabs(round( ( ori_image[heap[0]].phase - ori_image[ ori_image[heap[0]].pre ].phase )/(2*PI) ))>0 && fabs(round( ( ori_image[heap[0]].phase - ori_image[ori_image[ori_image[heap[0]].pre].pre ].phase )/(2*PI) ))>0 && fabs(round( ( ori_image[heap[0]].phase -ori_image[ori_image[ori_image[ori_image[heap[0]].pre].pre ].pre].phase )/(2*PI) ))>0)
		ori_image[heap[0]].phase = ori_image[heap[0]].phase - round( ( ori_image[heap[0]].phase - ori_image[ ori_image[heap[0]].pre ].phase )/(2*PI) ) *2*PI;
		else
			ori_image[heap[0]].phase = ori_image[heap[0]].phase;

	}		
*/

		get_neighbors(nei, ori_image[heap[0]].pos, dim);

		curr_ind = heap_pop( ori_image, heap, &heap_length);

		for (i = 0; i<6; i++)
			if ( (INVALID!=nei[i]) && (0 == ori_image[nei[i]].planned ) && (ori_image[nei[i]].mag > noise) )
			{
				ori_image[nei[i]].pre = curr_ind;
				heap_add(ori_image, heap, nei[i], &heap_length);
				
				ori_image[nei[i]].planned = 1;
			}


		
	} while (heap_length>0 );

	free(heap); heap = NULL;
	
	return 0;
    
}

int qualityguidedunwrapping(float *mag, float *phase, long *matrix_size, float unwrap_noise_ratio)
{ 
  long max_ind = 0;
  struct MRI_point *MRimage = assemble_data(mag, phase, matrix_size, &max_ind);
  float noise_level = mag[max_ind] * unwrap_noise_ratio;
  long seed_pixel = 0;
  { 
    //calculated the center of mass
    float total_mass = 0;
    float total_weight_x = 0;
    float total_weight_y = 0;
    float total_weight_z = 0;

    for (long k = 0; k<matrix_size[2]; k++) {
      for (long j = 0 ; j<matrix_size[1]; j++) {
        for (long i=0; i<matrix_size[0];i++) {
          long idx = k*matrix_size[0]*matrix_size[1] + j*matrix_size[0] + i;
          total_weight_x += mag[idx]*i;
          total_weight_y += mag[idx]*j;
          total_weight_z += mag[idx]*k;
          total_mass += mag[idx];					
        }
      }
    }
    long com_x = total_weight_x/total_mass;
    long com_y = total_weight_y/total_mass;
    long com_z = total_weight_z/total_mass;
    seed_pixel = com_z*matrix_size[1]*matrix_size[0]
      + com_y*matrix_size[0] 
      + com_x;
  }

  /* choose the center of mass as the seeding point	*/
  unwrap(MRimage, seed_pixel, matrix_size, noise_level);	
  for (long i= 0; i<matrix_size[0]*matrix_size[1]*matrix_size[2]; i++) {
    phase[i] = MRimage[i].phase;
  }
  free(MRimage); MRimage = NULL;
  return 0;
}

