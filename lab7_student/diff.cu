#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCKSIZE 4 // Number of threads in each thread block
 
// CUDA kernel. Each thread takes care of one element of a 
__global__ void diffKernel( float *in, float *out, int n )
{
    // Wrtie the kernel to implement the diff operation on an array Using shared memory
	unsigned int tx = threadIdx.x;
	
	__shared__ int s_data[BLOCKSIZE];
	
	unsigned int i = blockDim.x * blockIdx.x + tx;
	
	if(i < n) {
		s_data[tx] = in[i];
	}
	
	__syncthreads();
	
	if(tx > 0) {
		out[i] = s_data[tx] - s_data[tx - 1];
	}
	
	else if(i > 0) {
		out[i] = s_data[tx] - in[i - 1];
	}
}  
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int i;
    float input[] = {4, 5, 6, 7, 19, 10, 0, 4, 2, 3, 1, 7, 9, 11, 45, 23, 99, 29};
    int n = sizeof(input) / sizeof(float); //careful, this usage only works with statically allocated arrays, NOT dynamic arrays

    // Host input vectors
    float *h_in = input;
    //Host output vector
    float *h_out = (float *) malloc(n * sizeof(float));
 
    // Device input vectors
    float *d_in;
    //Device output vector
    float *d_out;
 
    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
 
    // Copy host data to device
    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // TODO: setup the blocksize and gridsize and launch the kernel below.
     
    // Number of threads in each thread block
	int blockSize = BLOCKSIZE;
	
    // Number of thread blocks in grid
	int gridSize = (int)ceil((float)n/blockSize);
	
	// Size of shared memory
	int sharedSize = blockSize * sizeof(float);
	
    // Execute the kernel
	diffKernel<<<gridSize, blockSize, sharedSize>>>(d_in, d_out, n);
 
    // Copy array back to host
    cudaMemcpy( h_out, d_out, bytes, cudaMemcpyDeviceToHost );
 
    // Show the result
    printf("The original array is: ");
    for(i = 0; i < n; i ++)
        printf("%4.0f,", h_in[i] );    
    
    printf("\n\nThe diff     array is: ");
    for(i = 0; i < n; i++)
        printf("%4.0f,", h_out[i] );    
    puts("");
    
    // Release device memory
    cudaFree(d_in);
    cudaFree(d_out);
 
    // Release host memory
    free(h_out);
 
    return 0;
}
