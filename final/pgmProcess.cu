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
 
//A structure to represent a stack 
struct Stack 
{ 
    int top; 
    unsigned capacity; 
    int* array; 
}; 
  
//function to create a stack of given capacity. It initializes size of stack as 0 
__device__ struct Stack* createStack(unsigned capacity) 
{ 
    struct Stack* stack = (struct Stack*) malloc(sizeof(struct Stack)); 
    stack->capacity = capacity; 
    stack->top = -1; 
    stack->array = (int*) malloc(stack->capacity * sizeof(int)); 
    return stack; 
} 
  
// Stack is full when top is equal to the last index 
__device__ int isFull(struct Stack* stack) 
{   return stack->top == stack->capacity - 1; } 
  
// Stack is empty when top is equal to -1 
__device__ int isEmpty(struct Stack* stack) 
{   return stack->top == -1;  } 
  
// Function to add an item to stack.  It increases top by 1 
__device__ void push(struct Stack* stack, int item) 
{ 
    if (isFull(stack)) 
        return; 
    stack->array[++stack->top] = item; 
} 
  
// Function to remove an item from stack.  It decreases top by 1 
__device__ int pop(struct Stack* stack) 
{ 
    if (isEmpty(stack)) 
        return INT_MIN; 
    return stack->array[stack->top--]; 
}

__global__ void gaussianBlur(int *d_pixelsIn, int *d_pixelsOut, double *guassian, int floatpitch, int width) {
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
		  1 * s_data[((bx + 1) * 3) + 2] +          //SE
		  2 * s_data[(bx * 3) + 2      ] +          //S
		  1 * s_data[((bx - 1)* 3) + 2 ] -          //SW
		  2 * s_data[(bx * 3)          ] -          //N
		  1 * s_data[(bx + 1) * 3      ] -          //NE
		  1 * s_data[(bx - 1)* 3       ]            //NW
	);
	
	d_pixelsOut[i * floatpitch + j] = sqrt(dx * dx + dy * dy);
	d_orientation[i * floatpitch + j] = atan2(dy, dx);
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
	
	if((orientation >=0 && orientation <=45) || (orientation <-135 && orientation >=-180)) {
		d_pixelsOut[i * floatpitch + j] = (value >= n && value >= s) ? value : 0;
	}
	
	else if((orientation > 45 && orientation <=90) || (orientation <-90 && orientation >=-135)) {
		d_pixelsOut[i * floatpitch + j] = (value >= nw && value >= se) ? value : 0;
	}
	
	else if((orientation > 90 && orientation <=135) || (orientation <-45 && orientation >=-90)) {
		d_pixelsOut[i * floatpitch + j] = (value >= w && value >= e) ? value : 0;
	}
	
	else if((orientation >135 && orientation <=180) || (orientation <0 && orientation >=-45)) {
		d_pixelsOut[i * floatpitch + j] = (value >= sw && value >= ne) ? value : 0;
	}
	
	else {
		d_pixelsOut[i * floatpitch + j] = 0;
	}
}
