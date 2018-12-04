#include "pgmProcess.h"
#include "pgmUtility.h"

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */ 
 
// Stack is full when top is equal to the last index 
__device__ __host__ int isFull(int *top, unsigned capacity) 
{   return *top == capacity - 1; } 
  
// Stack is empty when top is equal to -1 
__device__ __host__ int isEmpty(int *top) 
{   return *top == -1;  } 
  
// Function to add an item to stack.  It increases top by 1 
__device__ __host__ void push(int *stack, int *top, int capacity, int item) 
{ 
    if (isFull(top, capacity)) 
        return; 
    stack[++*top] = item; 
} 
  
// Function to remove an item from stack.  It decreases top by 1 
__device__ __host__ int pop(int *stack, int *top, unsigned capacity) 
{ 
    if (isEmpty(top)) 
        return INT_MIN; 
    return stack[*top--]; 
}


__device__ __host__ void pushNeighbors(int currentPixelIndex, int lower_thresh, int upper_thresh, int floatpitch, int width, int height, int *stack, int *top, int capacity, int *d_pixelsIn, int *d_pixelsOut) {
	int x_index = currentPixelIndex % floatpitch;
	int y_index = (currentPixelIndex - x_index) / floatpitch;
	
	int n_pixel = (y_index - 1) * floatpitch + (x_index);
	int nw_pixel = (y_index - 1) * floatpitch + (x_index + 1);
	int w_pixel = (y_index) * floatpitch + (x_index + 1);
	int sw_pixel = (y_index + 1) * floatpitch + (x_index + 1);
	int s_pixel = (y_index + 1) * floatpitch + (x_index);
	int se_pixel = (y_index + 1) * floatpitch + (x_index - 1);
	int e_pixel = (y_index) * floatpitch + (x_index - 1);
	int ne_pixel = (y_index - 1) * floatpitch + (x_index - 1);
	
	
	if(x_index + 1 < width - 1) {
		if(d_pixelsIn[w_pixel] >= lower_thresh) {
			push(stack, top, capacity, w_pixel);
		}
		d_pixelsOut[w_pixel] = 255;
	}
	
	if(y_index + 1 < height - 1) {
		if(d_pixelsIn[s_pixel] >= lower_thresh) {
			push(stack, top, capacity, s_pixel);
		}
		d_pixelsOut[s_pixel] = 255;
	}
	
	if(x_index - 1 >= 1) {
		if(d_pixelsIn[e_pixel] >= lower_thresh) {
			push(stack, top, capacity, e_pixel);
		}
		d_pixelsOut[e_pixel] = 255;
	}
	
	if(y_index - 1 >= 1) {
		if(d_pixelsIn[n_pixel] >= lower_thresh) {
			push(stack, top, capacity, n_pixel);
		}
		d_pixelsOut[n_pixel] = 255;
	}
	
	if(x_index - 1 >=1 && y_index - 1 >= 1) {
		if(d_pixelsIn[ne_pixel] >= lower_thresh) {
			push(stack, top, capacity, ne_pixel);
		}
		d_pixelsOut[ne_pixel] = 255;
	}
	
	if(x_index + 1 < width - 1 && y_index + 1 < height - 1) {
		if(d_pixelsIn[sw_pixel] >= lower_thresh) {
			push(stack, top, capacity, sw_pixel);
		}
		d_pixelsOut[sw_pixel] = 255;
	}
	
	if(x_index + 1 < width - 1 && y_index - 1 >= 1) {
		if(d_pixelsIn[nw_pixel] >= lower_thresh) {
			push(stack, top, capacity, nw_pixel);
		}
		d_pixelsOut[nw_pixel] = 255;
	}
	
	if(x_index - 1 >=1 && y_index + 1 < height - 1) {
		if(d_pixelsIn[se_pixel] >= lower_thresh) {
			push(stack, top, capacity, se_pixel);
		}
		d_pixelsOut[se_pixel] = 255;
	}
}

__global__ void gaussianBlur(int *d_pixelsIn, int *d_pixelsOut, double *guassian, int floatpitch, int width, int height) {
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
	if( i >= height - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
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
	
	d_pixelsOut[i * floatpitch + j] = (
                              guassian[4] * s_data[(bx * 3) + 1      ] +          //itself
                              guassian[1] * s_data[(bx * 3)          ] +          //N
                              guassian[0] * s_data[(bx + 1) * 3      ] +          //NE
                              guassian[3] * s_data[((bx + 1) * 3) + 1] +          //E
                              guassian[6] * s_data[((bx + 1) * 3) + 2] +          //SE
                              guassian[7] * s_data[(bx * 3) + 2      ] +          //S
                              guassian[8] * s_data[((bx - 1)* 3) + 2 ] +          //SW
                              guassian[5] * s_data[((bx - 1)* 3) + 1 ] +          //W
                              guassian[2] * s_data[(bx - 1)* 3       ]            //NW
                           );
}

__global__ void pgmSobel(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width, int height) {
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
	if( i >= height - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
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
	
	int n = s_data[(bx * 3)];
	int nw = s_data[(bx - 1)* 3];
	int w = s_data[((bx - 1)* 3) + 1];
	int sw = s_data[((bx - 1)* 3) + 2];
	int s = s_data[(bx * 3) + 2];
	int se = s_data[((bx + 1) * 3) + 2];
	int e = s_data[((bx + 1) * 3) + 1];
	int ne = s_data[(bx + 1) * 3];
	
	double dx = (
		  1 * ne +         //NE
		  2 * e  +       //E
		  1 * se -        //SE
		  1 * sw -         //SW
		  2 * w  -       //W
		  1 * nw          //NW
	);
	
	double dy = (
		  1 * se +         //SE
		  2 * s  +        //S
		  1 * sw -         //SW
		  2 * n  -        //N
		  1 * ne -        //NE
		  1 * nw            //NW
	);
	
	d_pixelsOut[i * floatpitch + j] = sqrt(dx * dx + dy * dy);
	float angle = atan2(dy, dx);
	if(angle < 0) {
		angle += 3.14159;
	}
	d_orientation[i * floatpitch + j] = angle;
}

__global__ void pgmNonMaximumSupression(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width, int height) {
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
	if( i >= height - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
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
	float orientation = d_orientation[i * floatpitch + j] * 180.0 / 3.14159;
	
	//Surrounding values
	int value = s_data[(bx * 3) + 1];
	int n = s_data[(bx * 3)];
	int nw = s_data[(bx - 1)* 3];
	int w = s_data[((bx - 1)* 3) + 1];
	int sw = s_data[((bx - 1)* 3) + 2];
	int s = s_data[(bx * 3) + 2];
	int se = s_data[((bx + 1) * 3) + 2];
	int e = s_data[((bx + 1) * 3) + 1];
	int ne = s_data[(bx + 1) * 3];
	
    if(orientation > 22.5 && orientation <= 67.5) {
		d_pixelsOut[i * floatpitch + j] = (value >= nw && value >= se) ? value : 0;
	}
	
	else if(orientation > 67.5 && orientation <= 112.5) {
		d_pixelsOut[i * floatpitch + j] = (value >= n && value >= s) ? value : 0;
	}
	
	else if(orientation > 112.5 && orientation <= 157.5) {
		d_pixelsOut[i * floatpitch + j] = (value >= sw && value >= ne) ? value : 0;
	}
	
	else if((orientation > 0 && orientation <= 22.5) || (orientation > 157.5 && orientation <= 180)) {
		d_pixelsOut[i * floatpitch + j] = (value >= e && value >= w) ? value : 0;
	}
	
	else {
		d_pixelsOut[i * floatpitch + j] = 0;
	}
}

__global__ void pgmHysterisisThresholding(int *d_pixelsIn, int *d_pixelsOut, int lower_thresh, int upper_thresh, int floatpitch, int width, int height) {
	//stack for storing weak edges
	const int capacity = 1;
	int stack[capacity];
	
	// global thread(data) row index 
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	i = i + 1; //because the edge of the data is not processed
		
	// global thread(data) column index
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	j = j + 1; //because the edge of the data is not processed
	
	// check the boundary
	if( i >= height - 1 || j >= width - 1 || i < 1 || j < 1 ) return;
	
	int top = -1;
	
	if( d_pixelsIn[i * floatpitch + j] >= upper_thresh ) {
		d_pixelsOut[i * floatpitch + j] = 255;
		
		int currentPixelIndex = i * floatpitch + j;
		
		do {
			pushNeighbors(currentPixelIndex, lower_thresh, upper_thresh, floatpitch, width, height, stack, &top, capacity, d_pixelsIn, d_pixelsOut);
			currentPixelIndex = pop(stack, &top, capacity);
			if(isFull(&top, capacity)) {
				break;
			}
		} while(!isEmpty(&top));
	}
	
	else {
		d_pixelsOut[i * floatpitch + j] = 0;
	}
}

__host__ void pgmSobelSqequential(int *pixelsIn, int *pixelsOut, float *orientation, int width, int height) {
	int i, j;
	for(i = 1; i < height - 1; i ++) {
		for(j = 1; j < width - 1; j++) {
			int n = pixelsIn[(i - 1) * width + j];
 			int s = pixelsIn[(i + 1) * width + j];
			int e = pixelsIn[i * width + 1 + j];
			int w = pixelsIn[i * width - 1 + j];
			int nw = pixelsIn[(i - 1) * width - 1 + j];
			int ne = pixelsIn[(i - 1) * width + 1 + j];
			int sw = pixelsIn[(i + 1) * width - 1 + j];
			int se = pixelsIn[(i + 1) * width + 1 + j];
			double dy = sw + (2 * s) + se - nw - (2 * n) - ne;
			double dx = ne + (2 * e) + se - nw - (2 * w) - sw;
			pixelsOut[i * width + j] = sqrt(dx * dx + dy * dy);
			float angle = atan2(dy, dx);
			if(angle < 0) {
				angle += 3.14159;
			}
			orientation[i * width + j] = angle;
		}
	}
}

__host__ void pgmGuassianBlurSequential(int *pixelsIn, int *pixelsOut, double *guassian, int width, int height) {
	int i, j;
	for(i = 1; i < height - 1; i ++) {
		for(j = 1; j < width - 1; j++) {
			int value = pixelsIn[i * width + j];
			int n = pixelsIn[(i - 1) * width + j];
 			int s = pixelsIn[(i + 1) * width + j];
			int e = pixelsIn[i * width + j +1];
			int w = pixelsIn[i * width - 1 + j];
			int nw = pixelsIn[(i - 1) * width - 1 + j];
			int ne = pixelsIn[(i - 1) * width + 1 + j];
			int sw = pixelsIn[(i + 1) * width - 1 + j];
			int se = pixelsIn[(i + 1) * width + 1 + j];
			pixelsOut[i * width + j] = (
                              guassian[4] * value +          //itself
                              guassian[1] * n +          //N
                              guassian[0] * ne +          //NE
                              guassian[3] * e +          //E
                              guassian[6] * se +          //SE
                              guassian[7] * s +          //S
                              guassian[8] * sw +          //SW
                              guassian[5] * w +          //W
                              guassian[2] * nw            //NW
                           );
		}
	}
}

__host__ void pgmNonMaximumSupressionSequential(int *pixelsIn, int *pixelsOut, float *h_orientation, int width, int height) {
	int i, j;
	for(i = 1; i < height - 1; i ++) {
		for(j = 1; j < width - 1; j++) {
			int value = pixelsIn[i * width + j];
			int n = pixelsIn[(i - 1) * width + j];
 			int s = pixelsIn[(i + 1) * width + j];
			int e = pixelsIn[i * width + j +1];
			int w = pixelsIn[i * width - 1 + j];
			int nw = pixelsIn[(i - 1) * width - 1 + j];
			int ne = pixelsIn[(i - 1) * width + 1 + j];
			int sw = pixelsIn[(i + 1) * width - 1 + j];
			int se = pixelsIn[(i + 1) * width + 1 + j];
			float orientation = h_orientation[i * width + j] * 180.0 / 3.14159;
			
			if(orientation > 22.5 && orientation <= 67.5) {
				pixelsOut[i * width + j] = (value >= nw && value >= se) ? value : 0;
			}
			
			else if(orientation > 67.5 && orientation <= 112.5) {
				pixelsOut[i * width + j] = (value >= n && value >= s) ? value : 0;
			}
			
			else if(orientation > 112.5 && orientation <= 157.5) {
				pixelsOut[i * width + j] = (value >= sw && value >= ne) ? value : 0;
			}
			
			else if((orientation > 0 && orientation <= 22.5) || (orientation > 157.5 && orientation <= 180)) {
				pixelsOut[i * width + j] = (value >= e && value >= w) ? value : 0;
			}
			
			else {
				pixelsOut[i * width + j] = 0;
			}
		}
	}
}

__host__ void pgmHysterisisThresholdingSequential(int *pixelsIn, int *pixelsOut, int lower_thresh, int upper_thresh, int width, int height) {
	const int capacity = 1;
	int stack[capacity];
	int top = -1;
	
	int i, j;
	for(i = 0; i < height - 1; i++) {
		for(j = 0; j < width; j++) {
			if( pixelsIn[i * width + j] >= upper_thresh ) {
				pixelsOut[i * width + j] = 255;
				
				int currentPixelIndex = i * width + j;
				
				do {
					pushNeighbors(currentPixelIndex, lower_thresh, upper_thresh, width, width, height, stack, &top, capacity, pixelsIn, pixelsOut);
					currentPixelIndex = pop(stack, &top, capacity);
					if(isFull(&top, capacity)) {
						break;
					}
				} while(!isEmpty(&top));
			}
			
			else {
				pixelsOut[i * width + j] = 0;
			}
		}
	}
}