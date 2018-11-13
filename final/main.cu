#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmUtility.h"
#include "pgmProcess.h"
#include "timing.h"

int device = 0;

void usage()
{
	printf("Usage: ./programName oldImageFile newImageFile threadsPerBlock\n");
	exit(1);
}

void runSobel(int *h_dataA, int* h_dataB, float *h_orientation, int *numCols, int *numRows, int threadsPerBlock){

   // To ensure alignment, we'll use the code below to pad rows of the arrays when they are 
   // allocated on the device.
   size_t pitch;
   // allocate device memory for data A
   int* d_dataA;
   cudaMallocPitch( (void**) &d_dataA, &pitch, *numCols * sizeof(int), *numRows);
   
   // copy host memory to device memory for image A
   cudaMemcpy2D( d_dataA, pitch, h_dataA, *numCols * sizeof(int), *numCols * sizeof(int), *numRows,
                             cudaMemcpyHostToDevice);
   
   
   // repeat for second device array
   int* d_dataB;
   cudaMallocPitch( (void**) &d_dataB, &pitch, *numCols * sizeof(int), *numRows);
   
   // copy host memory to device memory for image B
   cudaMemcpy2D( d_dataB, pitch, h_dataB, *numCols * sizeof(int), *numCols * sizeof(int), *numRows,
                             cudaMemcpyHostToDevice);
   
   // repeat for orientation array
   float* d_orientation;
   cudaMallocPitch( (void**) &d_orientation, &pitch, *numCols * sizeof(float), *numRows);
   
   // copy host memory to device memory for image B
   cudaMemcpy2D( d_orientation, pitch, h_dataB, *numCols * sizeof(float), *numCols * sizeof(float), *numRows,
                             cudaMemcpyHostToDevice);
                             
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
  
   //execute the kernel
   pgmSobel<<< grid, threads, shared_mem_size >>>( d_dataA, d_dataB, d_orientation, pitch/sizeof(int), *numCols);
   
   // copy result from device to host
   cudaMemcpy2D( h_dataB, *numCols * sizeof(int), d_dataB, pitch, *numCols * sizeof(int), *numRows,cudaMemcpyDeviceToHost);
   cudaMemcpy2D( h_orientation, *numCols * sizeof(float), d_orientation, pitch, *numCols * sizeof(float), *numRows,cudaMemcpyDeviceToHost);
   
   // cleanup memory
   cudaFree(d_dataA);
   cudaFree(d_dataB);
   cudaFree(d_orientation);
}

int main( int argc, char *argv[] )  
{
	FILE *oldImageFile;
	FILE *newImageFile;
	int *numRows;
	int *numCols;
	int *h_dataA;
	int *h_dataB;
	float *h_orientation;
	char **header;

	if( argc == 4 )
	{
		int threadsPerBlock = atoi(argv[3]);
		
		if ((oldImageFile = fopen(argv[1], "r")) && (newImageFile = fopen(argv[2], "w"))) {
			numRows = (int*)malloc(sizeof(int));
			numCols = (int*)malloc(sizeof(int));
			header = initAra2D(rowsInHeader, 2);
			h_dataA = pgmRead(header, numRows, numCols, oldImageFile);
			h_dataB = (int *)malloc(*numRows*(*numCols)*sizeof(int));
			h_orientation = (float *)malloc(*numRows*(*numCols)*sizeof(float));
			runSobel(h_dataA, h_dataB, h_orientation, numCols, numRows, threadsPerBlock);
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
