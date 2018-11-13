#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmUtility.h"
#include "timing.h"

int device = 0;

void usage()
{
	printf("Usage: ./programName oldImageFile newImageFile threadsPerBlock\n");
	exit(1);
}

void readData(FILE *oldImageFile, int *numRows, int *numCols, char **header, float *h_dataA, float *h_dataB, float *h_orientation) {
	numRows = (int*)malloc(sizeof(int));
	numCols = (int*)malloc(sizeof(int));
	header = initAra2D(rowsInHeader, 2);
	h_dataA = pgmRead(header, numRows, numCols, oldImageFile);
	h_dataB = (float *)malloc(*numRows*(*numCols)*sizeof(float));
	h_orientation = (float *)malloc(*numRows*(*numCols)*sizeof(float));
}

void runSobel(float *h_dataA, float* h_dataB, int *numCols, int *numRows, int passes, int threadsPerBlock, int shouldPrint){

   // Use card 0  (See top of file to make sure you are using your assigned device.)
   checkCudaErrors(cudaSetDevice(device));

   // To ensure alignment, we'll use the code below to pad rows of the arrays when they are 
   // allocated on the device.
   size_t pitch;
   // allocate device memory for data A
   float* d_dataA;
   checkCudaErrors( cudaMallocPitch( (void**) &d_dataA, &pitch, *numCols * sizeof(float), *numRows));
   
   // copy host memory to device memory for image A
   checkCudaErrors( cudaMemcpy2D( d_dataA, pitch, h_dataA, *numCols * sizeof(float), *numCols * sizeof(float), *numRows,
                             cudaMemcpyHostToDevice) );
   
   
   // repeat for second device array
   float* d_dataB;
   checkCudaErrors( cudaMallocPitch( (void**) &d_dataB, &pitch, *numCols * sizeof(float), *numRows));
   
   // copy host memory to device memory for image B
   checkCudaErrors( cudaMemcpy2D( d_dataB, pitch, h_dataB, *numCols * sizeof(float), *numCols * sizeof(float), *numRows,
                             cudaMemcpyHostToDevice) );
   
   // repeat for orientation array
   float* d_orientation;
   checkCudaErrors( cudaMallocPitch( (void**) &d_orientation, &pitch, *numCols * sizeof(float), *numRows));
   
   // copy host memory to device memory for image B
   checkCudaErrors( cudaMemcpy2D( d_orientation, pitch, h_dataB, *numCols * sizeof(float), *numCols * sizeof(float), *numRows,
                             cudaMemcpyHostToDevice) );
                             
   //***************************
   // setup CUDA execution parameters
   
   int blockHeight;
   int blockWidth;
   
   // When testing with small arrays, this code might be useful. Feel free to change it.
   if (threadsPerBlock > *numCols - 2 ){
   
      blockWidth = 16 * (int) ceil((*numCols - 2) / 16.0); 
      blockHeight = 1;
   } else {
      
      blockWidth = threadsPerBlock;
      blockHeight = 1;
   }
   
   int gridWidth = (int) ceil( (*numCols - 2) / (float) blockWidth);
   int gridHeight = (int) ceil( (*numRows - 2) / (float) blockHeight);
   
   // number of blocks required to process all the data.
   int numBlocks =   gridWidth * gridHeight;
   
   // Each block gets a shared memory region of this size.
   unsigned int shared_mem_size = ((blockWidth + 2) * 4) * sizeof(float); 
   
   printf("blockDim.x=%d blockDim.y=%d    grid = %d x %d\n", blockWidth, blockHeight, gridWidth, gridHeight);
   printf("numBlocks = %d,  threadsPerBlock = %d   shared_mem_size = %d\n", numBlocks, threadsPerBlock,  shared_mem_size);
   
   if(gridWidth > 65536 || gridHeight > 65536) {
      fprintf(stderr, "****Error: a block dimension is too large.\n");
   }
   
   if(threadsPerBlock > 1024) {
      fprintf(stderr, "****Error: number of threads per block is too large.\n");
   }
   
   if(shared_mem_size > 49152) {
      fprintf(stderr, "****Error: shared memory per block is too large.\n");
   }
      
   // Format the grid, which is a collection of blocks. 
   dim3  grid( gridWidth, gridHeight, 1);
   
   // Format the blocks. 
   dim3  threads( blockWidth, blockHeight, 1);
  
   float * temp;
      //execute the kernel
   pgmSobel<<< grid, threads, shared_mem_size >>>( d_dataA, d_dataB, d_orientation, pitch/sizeof(float), *numCols);
      
   // check if kernel execution generated an error
   cudaError_t code = cudaGetLastError();
   if (code != cudaSuccess){
       printf ("Cuda Kerel Launch error -- %s\n", cudaGetErrorString(code));
   }

   //checkCudaErrors( cutStopTimer( timer));
   
   // copy result from device to host
   checkCudaErrors( cudaMemcpy2D( h_dataB, *numCols * sizeof(float), d_dataB, pitch, *numCols * sizeof(float), *numRows,cudaMemcpyDeviceToHost) );
   checkCudaErrors( cudaMemcpy2D( h_orientation, *numCols * sizeof(float), d_orientation, pitch, *numCols * sizeof(float), *numRows,cudaMemcpyDeviceToHost) );
   
   // cleanup memory
   checkCudaErrors(cudaFree(d_dataA));
   checkCudaErrors(cudaFree(d_dataB));
   checkCudaErrors(cudaFree(d_orientation));
}

int main( int argc, char *argv[] )  
{
	FILE *oldImageFile;
	FILE *newImageFile;
	int *numRows;
	int *numCols;
	float *h_dataA;
	float *h_dataB;
	float *h_orientation;
	char **header;

	if( argc == 4 )
	{
		int threadsPerBlock = atoi(argv[1]);
		
		if ((oldImageFile = fopen(argv[1], "r")) && (newImageFile = fopen(argv[2], "w"))) {
			readData(FILE *oldImageFile, int *numRows, int *numCols, char **header, float *h_dataA, float *h_dataB, float *h_orientation);
			runSobel(float *h_dataA, float* h_dataB, int *numCols, int *numRows, int threadsPerBlock);
			int errorCodeWrite = pgmWrite((const char **)header, h_dataB, *numRows, *numCols, newImageFile);
			free(numRows);
			free(numCols);
			freeHeader(header, rowsInHeader);
			free(h_dataA);
			free(h_dataB);
			free(h_orientation);
		}
	}
	
	else {
		if(oldImageFile) {
			fclose(oldImageFile);
		}
		if(newImageFile) {
			fclose(newImageFile);
		}
		usage();
	}
}
