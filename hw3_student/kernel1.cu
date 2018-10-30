#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
	extern __shared__ float s_data[];
    // TODO, implement this kernel below
    // global thread(data) row index 
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	i = i + 1; //because the edge of the data is not processed
	
	
	// global thread(data) column index
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	j = j + 1; //because the edge of the data is not processed
	
	//block relative row index
	int by = threadIdx.y + 1;
	
	//block relative column index
	int bx = threadIdx.x + 1;
	
	// check the boundary
	//This will be insufficient to deal with edge cases because of padding -- will somehow need to incorporate pitch
	//if i + pitch > width - 1 || j + pitch >= width - 1 <-- why is this width and not height for i????
	if( i >= width - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
	//Note: s_data should be small, so want to start indexing from 0
	//Hence, must account for the fact that block is offset by blockIdx * blockDim

	if(bx == 1) {
		s_data[((bx - 1) * 3)] = g_dataA[(i-1) * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 1] = g_dataA[i * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 2] = g_dataA[(i+1) * floatpitch + (j-1)];
		s_data[(bx * 3)] = g_dataA[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = g_dataA[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = g_dataA[(i+1) * floatpitch +  j];
	}
	
	else if(bx == blockDim.x || j == width - 2) {
		s_data[(bx * 3)] = g_dataA[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = g_dataA[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = g_dataA[(i+1) * floatpitch +  j];
		s_data[(bx + 1) * 3] = g_dataA[(i-1) * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 1] = g_dataA[i * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 2] = g_dataA[(i+1) * floatpitch + (j+1)];
	}
	
	else {
		s_data[(bx * 3)] = g_dataA[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = g_dataA[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = g_dataA[(i+1) * floatpitch +  j];
	}
	
	__syncthreads();
	
	g_dataB[i * floatpitch + j] = (
                              0.2f * s_data[(bx * 3) + 1      ] +          //itself
                              0.1f * s_data[(bx * 3)          ] +          //N
                              0.1f * s_data[(bx + 1) * 3      ] +          //NE
                              0.1f * s_data[((bx + 1) * 3) + 1] +          //E
                              0.1f * s_data[((bx + 1) * 3) + 2] +          //SE
                              0.1f * s_data[(bx * 3) + 2      ] +          //S
                              0.1f * s_data[((bx - 1)* 3) + 2 ] +          //SW
                              0.1f * s_data[((bx - 1)* 3) + 1 ] +          //W
                              0.1f * s_data[(bx - 1)* 3       ]            //NW
                           ) * 0.95f;
}

