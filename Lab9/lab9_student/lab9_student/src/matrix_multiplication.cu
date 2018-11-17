#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "timing.h"
#include "mul.h"
#include "arrayUtils.h"


// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(float *a, float *b, float *ab, size_t width)
{
  // calculate the row & column index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  float result = 0;

  // do dot product between row of a and column of b
  for(int k = 0; k < width; ++k)
  {
    result += a[row*width+k] * b[k*width+col];
  }

  // write out this thread's result
  ab[row*width+col] = result;

}

// a improved tiled matrix-matrix multiplication kernel
__global__ void matrix_multiply_improved(float *a, float *b, float *ab, size_t width)
{
    // TODO: write the kernel to perform matrix a times b, store results into ab.
    // You are required to use shared memory to do tiled matrix-matrix multiplication
    // width is the size of the square matrix along one dimension.
	int TILE_WIDTH = blockDim.x;
	
	extern __shared__ float Mds[];
	extern __shared__ float Nds[];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// Identify the row and column of the ab element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	// Loop over the a and b tiles required to compute ab element
	for(int m = 0; m < width/TILE_WIDTH; ++m) {
		//Coolaborative loading of a and b tiles required to compute ab element
		Mds[ty * TILE_WIDTH + tx] = a[Row * width + m * TILE_WIDTH + tx];
		Nds[ty * TILE_WIDTH + tx] = b[(m * TILE_WIDTH + ty) * width + Col];
		__syncthreads();
		
		for(int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
		}
		__syncthreads();
	}
	ab[Row * width + Col] = Pvalue;
}


void usage()
{
   printf("Usage: ./progName blockWidth matrixFileName p \n");
}


int main(int argc, char *argv[])
{
   // create a large workload so we can easily measure the
   // performance difference on CPU and GPU

   // to run this program: ./a.out blockWidth matrixFileName p
   int shouldPrint = 0;
   if(argc < 3 || argc > 4) {
      usage();
      return 1;
   } else  if(argc == 3){
         shouldPrint = 0;
   } else if(argv[3][0]=='p'){
         shouldPrint=1;
   } else {
         usage();
         return 1;
   }
  
   //
   int tile_width = atoi(argv[1]);
   if ( ! tile_width )
   {
       printf("Wrong argument passed in for blockWidth\n");
       exit(-1);
   }
   //
   float *h_a, *h_c, *h_b;  // In this application, we test matrix multiplication h_a * h_a and store results into h_c
   int rows = 0, cols = 0, n = 0;  // n is the width of the one side of the input square matrix


   //read in the matrix data from file
   h_a = readNewArray(&rows, &cols, argv[2]);
   h_b = readNewArray(&rows, &cols, argv[2]);//for host test
   n = rows; //initialize n

   if( ! h_a || ! h_b )
   {
       printf("Error in file I/O, check your file name or path to file!\n");
       exit(-1);
   }
   // set up host memory for results
   h_c = (float *)calloc(n * n, sizeof(float));
  
  const dim3 block_size(tile_width, tile_width);
  const dim3 num_blocks(ceil(n / (float)block_size.x), ceil(n / (float)block_size.y));

  // allocate storage for the device
  float *d_a = 0, *d_b = 0, *d_c = 0;
  cudaMalloc((void**)&d_a, sizeof(float) * n * n);
  cudaMalloc((void**)&d_b, sizeof(float) * n * n);
  cudaMalloc((void**)&d_c, sizeof(float) * n * n);

  // copy input to the device
  cudaMemcpy(d_a, h_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

  // time the kernel launches using CUDA events
  cudaEvent_t launch_begin, launch_end;
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);

  // to get accurate timings, launch a single "warm-up" kernel
  matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  writeArray(h_c, n, n, "gpuout2");
  if(shouldPrint)
      printArray(h_c, n, n); 

  // time many kernel launches and take the average time
  const size_t num_launches = 10;
  float average_simple_time = 0;
  printf("Timing simple GPU implementation... \n");
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
    
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_simple_time += time;
  }
  average_simple_time /= num_launches;
  printf(" done! GPU time cost in second: %f\n", average_simple_time / 1000);
  
  //Now do the improved GPU solution
  // to get accurate timings, launch a single "warm-up" kernel
  matrix_multiply_improved<<<num_blocks,block_size,block_size * sizeof(float)>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  writeArray(h_c, n, n, "gpuout3");
  if(shouldPrint)
      printArray(h_c, n, n); 

  // time many kernel launches and take the average time
  float average_improved_time = 0;
  printf("Timing improved GPU implementation... \n");
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply_improved<<<num_blocks,block_size,block_size * sizeof(float)>>>(d_a, d_b, d_c, n);
    
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_improved_time += time;
  }
  average_improved_time /= num_launches;
  printf(" done! Improved GPU time cost in second: %f\n", average_improved_time / 1000);
  
  // report the effective throughput of each kernel in GFLOPS
  // the effective throughput is measured as the number of floating point operations performed per second:
  // (one mul + one addition) * N^3 operations
  float simple_throughput = (2.0f * n * n * n) / (average_simple_time / 1000.0f) / 1000000000.0f; //time in millisecond
  float improved_throughput = (2.0f * n * n * n) / (average_improved_time / 1000.0f) / 1000000000.0f; //time in millisecond
  float speedup = simple_throughput / improved_throughput;

  printf("Matrix size:  %d x %d \n", n, n);
  printf("Tile size: %d x %d\n", tile_width, tile_width);
 

  printf("Throughput of simple kernel: %.2f GFLOPS\n",simple_throughput);
  printf("Throughput of improved kernel: %.2f GFLOPS\n", improved_throughput);
  printf("Performance improvement: improved_throughput / simple_throughput = %.2f x\n", improved_throughput / simple_throughput );
  printf("Speedup in terms of time cost: %.2f x\n", speedup);

  // destroy the CUDA events
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
