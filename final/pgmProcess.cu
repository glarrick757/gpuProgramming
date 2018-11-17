#include "pgmProcess.h"

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
 
__global__ void pgmSobel(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width) {
	extern __shared__ int s_data[];
  
    // global thread(data) row index 
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	i = i + 1; //because the edge of the data is not processed
		
	// global thread(data) column index
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	j = j + 1; //because the edge of the data is not processed
	
	//block relative column index
	int bx = threadIdx.x + 1;
	
	// check the boundary
	if( i >= width - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
	//Left edge case
	if(bx == 1) {
		s_data[((bx - 1) * 3)] = d_pixelsIn[(i-1) * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 1] = d_pixelsIn[i * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 2] = d_pixelsIn[(i+1) * floatpitch + (j-1)];
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
	}
	
	//Right edge Case
	else if(bx == blockDim.x || j == width - 2) {
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
		s_data[(bx + 1) * 3] = d_pixelsIn[(i-1) * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 1] = d_pixelsIn[i * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 2] = d_pixelsIn[(i+1) * floatpitch + (j+1)];
	}
	
	else {
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
	}
	
	__syncthreads();
	
	double dx = (
		  1 * s_data[(bx + 1) * 3      ] +          //NE
		  2 * s_data[((bx + 1) * 3) + 1] +          //E
		  1 * s_data[((bx + 1) * 3) + 2] -          //SE
		  1 * s_data[((bx - 1)* 3) + 2 ] -          //SW
		  2 * s_data[((bx - 1)* 3) + 1 ] -          //W
		  1 * s_data[(bx - 1)* 3       ]            //NW
	);
	
	double dy = (
		  2 * s_data[(bx * 3)          ] +          //N
		  1 * s_data[(bx + 1) * 3      ] -          //NE
		  1 * s_data[((bx + 1) * 3) + 2] -          //SE
		  2 * s_data[(bx * 3) + 2      ] -          //S
		  1 * s_data[((bx - 1)* 3) + 2 ] +          //SW
		  1 * s_data[(bx - 1)* 3       ]            //NW
	);
	
	d_pixelsOut[i * floatpitch + j] = sqrt(dx * dx + dy * dy);
	d_orientation[i * floatpitch + j] = atan(dy / dx) + 3.14159;
}

__global__ void pgmNonMaximumSupression(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width) {
	extern __shared__ int s_data[];
  
    // global thread(data) row index 
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	i = i + 1; //because the edge of the data is not processed
		
	// global thread(data) column index
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	j = j + 1; //because the edge of the data is not processed
	
	//block relative column index
	int bx = threadIdx.x + 1;
	
	// check the boundary
	if( i >= width - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
	//Left edge case
	if(bx == 1) {
		s_data[((bx - 1) * 3)] = d_pixelsIn[(i-1) * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 1] = d_pixelsIn[i * floatpitch + (j-1)];
		s_data[((bx - 1) * 3) + 2] = d_pixelsIn[(i+1) * floatpitch + (j-1)];
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
	}
	
	//Right edge Case
	else if(bx == blockDim.x || j == width - 2) {
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
		s_data[(bx + 1) * 3] = d_pixelsIn[(i-1) * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 1] = d_pixelsIn[i * floatpitch + (j+1)];
		s_data[((bx + 1) * 3) + 2] = d_pixelsIn[(i+1) * floatpitch + (j+1)];
	}
	
	else {
		s_data[(bx * 3)] = d_pixelsIn[(i-1) * floatpitch +  j];
		s_data[(bx * 3) + 1] = d_pixelsIn[i * floatpitch +  j];
		s_data[(bx * 3) + 2] = d_pixelsIn[(i+1) * floatpitch +  j];
	}
	
	__syncthreads();
	
	//Gradient orientation of this pixel in degrees
	float orientation = d_orientation[i * floatpitch + j] * 180 / 3.14159;
	
	if((orientation >= 0 && orientation < 22.5) || (orientation >= 157.5 && orientation < 180)) {
	
	}
	
	else if(orientation >= 22.5 && orientation < 67.5) {
	
	}
	
	else if(orientation >= 67.5 && orientation < 112.5) {
	
	}
	
	else if(orientation >= 112.5 && orientation < 157.5) {
	
	}
	
	else {
		//assign 0 to output pixel
	}
	
	double dx = (
		  1 * s_data[(bx + 1) * 3      ] +          //NE
		  2 * s_data[((bx + 1) * 3) + 1] +          //E
		  1 * s_data[((bx + 1) * 3) + 2] -          //SE
		  1 * s_data[((bx - 1)* 3) + 2 ] -          //SW
		  2 * s_data[((bx - 1)* 3) + 1 ] -          //W
		  1 * s_data[(bx - 1)* 3       ]            //NW
	);
	
	double dy = (
		  2 * s_data[(bx * 3)          ] +          //N
		  1 * s_data[(bx + 1) * 3      ] -          //NE
		  1 * s_data[((bx + 1) * 3) + 2] -          //SE
		  2 * s_data[(bx * 3) + 2      ] -          //S
		  1 * s_data[((bx - 1)* 3) + 2 ] +          //SW
		  1 * s_data[(bx - 1)* 3       ]            //NW
	);
	
	0.2f * s_data[(bx * 3) + 1      ] +          //itself
	0.1f * s_data[(bx * 3)          ] +          //N
	0.1f * s_data[(bx + 1) * 3      ] +          //NE
	0.1f * s_data[((bx + 1) * 3) + 1] +          //E
	0.1f * s_data[((bx + 1) * 3) + 2] +          //SE
	0.1f * s_data[(bx * 3) + 2      ] +          //S
	0.1f * s_data[((bx - 1)* 3) + 2 ] +          //SW
    0.1f * s_data[((bx - 1)* 3) + 1 ] +          //W
	0.1f * s_data[(bx - 1)* 3       ]            //NW
	
	d_pixelsOut[i * floatpitch + j] = sqrt(dx * dx + dy * dy);
	d_orientation[i * floatpitch + j] = atan(dy / dx) + 3.14159;
}
