#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pgmUtility.h"
#include "pgmProcess.h"
#include "timing.h"

int device = 0;

void usage()
{
	printf("Usage: ./programName oldImageFile newImageFileGPU newImageFileCPU threadsPerBlock blur_value lower_threshold upper_threshold\n");
	exit(1);
}

float runCannyGPU(int *h_dataA, int* h_dataB, float *h_orientation, int *numCols, int *numRows, int threadsPerBlock, double sigma, int lower_thresh, int upper_thresh, double *guassian, int passes){

   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);
   
   // To ensure alignment, we'll use the code below to pad rows of the arrays when they are 
   // allocated on the device.
   size_t pitch;
   // allocate device memory for data A
   int* d_dataA;
   
   //compute gaussian kernel
   double *d_guassian;
   cudaMalloc( (void**) &d_guassian, 9 * sizeof(float));
   cudaMemcpy( d_guassian, guassian, 9 * sizeof(float), cudaMemcpyHostToDevice);
							 
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
	
   // repeat for third device array
   int* d_dataC;
   cudaMallocPitch( (void**) &d_dataC, &pitch, *numCols * sizeof(int), *numRows);
   
   // repeat for fourth device array
   int* d_dataD;
   cudaMallocPitch( (void**) &d_dataD, &pitch, *numCols * sizeof(int), *numRows);
   
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
   
   //execute guassian blur kernel
   guassianBlur<<< grid, threads, shared_mem_size >>>( d_dataA, d_dataB, d_guassian, pitch/sizeof(int), *numCols, *numRows); 
   
   cudaThreadSynchronize();
   
   //execute sobel kernel
   pgmSobel<<< grid, threads, shared_mem_size >>>( d_dataB, d_dataC, d_orientation, pitch/sizeof(int), *numCols, *numRows);
   
   cudaThreadSynchronize();
   
   //execute non-maximum supression kernel
   pgmNonMaximumSupression<<< grid, threads, shared_mem_size >>>(d_dataC, d_dataB, d_orientation, pitch/sizeof(int), *numCols, *numRows);
   //cudaError_t err = cudaGetLastError();
   //printf("Error 1: %s\n", cudaGetErrorString(err));
   
   cudaThreadSynchronize();
   
   //execute Hysterisis Thresholding
   pgmHysterisisThresholdingShared<<< grid, threads >>>(d_dataB, d_dataC, lower_thresh, upper_thresh, pitch/sizeof(int), *numCols, *numRows);
   
   cudaThreadSynchronize();
   
   // copy result from device to host
   cudaMemcpy2D( h_dataB, *numCols * sizeof(int), d_dataC, pitch, *numCols * sizeof(int), *numRows,cudaMemcpyDeviceToHost);
   
   int r;
   printf("Timing simple GPU implementation... \n");
   // record a CUDA event immediately before and after the kernel launch
   cudaEventRecord(launch_begin,0);
   for(r = 0; r < passes; r++) {
	   //execute guassian blur kernel
	   gaussianBlurShared<<< grid, threads, shared_mem_size >>>( d_dataA, d_dataB, d_guassian, pitch/sizeof(int), *numCols, *numRows); 
	   cudaThreadSynchronize();
	   //execute sobel kernel
	   pgmSobelShared<<< grid, threads, shared_mem_size >>>( d_dataB, d_dataC, d_orientation, pitch/sizeof(int), *numCols, *numRows);
	   cudaThreadSynchronize();
	   //execute non-maximum supression kernel
	   pgmNonMaximumSupressionShared<<< grid, threads, shared_mem_size >>>(d_dataC, d_dataB, d_orientation, pitch/sizeof(int), *numCols, *numRows);
	   //cudaError_t err = cudaGetLastError();
	   //printf("Error 1: %s\n", cudaGetErrorString(err));
	   cudaThreadSynchronize();
	   //execute Hysterisis Thresholding
	   pgmHysterisisThresholdingShared<<< grid, threads >>>(d_dataB, d_dataC, lower_thresh, upper_thresh, pitch/sizeof(int), *numCols, *numRows);
   }
   
   cudaEventRecord(launch_end,0);
   cudaEventSynchronize(launch_end);

   // measure the time spent in the kernel
   float time = 0;
   cudaEventElapsedTime(&time, launch_begin, launch_end);
   printf("	done! GPU time cost in second: %f\n", time / 1000 / passes);
   
   //cudaMemcpy2D( h_orientation, *numCols * sizeof(float), d_orientation, pitch, *numCols * sizeof(float), *numRows, cudaMemcpyDeviceToHost);
   // cleanup memory
   cudaFree(d_dataA);
   cudaFree(d_dataB);
   cudaFree(d_dataC);
   cudaFree(d_orientation);
   cudaFree(d_guassian);
   cudaEventDestroy(launch_begin);
   cudaEventDestroy(launch_end);
   
   return time / 1000 / passes;
}

float runCannyCPU(int *dataA, int* dataB, float *orientation, int *numCols, int *numRows, double sigma, int lower_thresh, int upper_thresh, double *guassian, int passes) {
	int *dataC = (int *)malloc(*numRows*(*numCols)*sizeof(int));
	float average_cpu_time = 0;
    clock_t now, then;
	
	pgmGuassianBlurSequential(dataA, dataC, guassian, *numCols, *numRows);
	pgmSobelSqequential(dataC, dataB, orientation, *numCols, *numRows);
	pgmNonMaximumSupressionSequential(dataB, dataC, orientation, *numCols, *numRows);
	pgmHysterisisThresholdingSequential(dataC, dataB, lower_thresh, upper_thresh, *numCols, *numRows);	
	
	printf("Timing CPU implementation...\n");
	int r;
	for(r = 0; r < passes; r++) {
		then = clock();
		pgmGuassianBlurSequential(dataA, dataC, guassian, *numCols, *numRows);
		pgmSobelSqequential(dataC, dataA, orientation, *numCols, *numRows);
		pgmNonMaximumSupressionSequential(dataA, dataC, orientation, *numCols, *numRows);
		pgmHysterisisThresholdingSequential(dataC, dataA, lower_thresh, upper_thresh, *numCols, *numRows);
		now = clock();
		
	    float time = 0;
        time = timeCost(then, now);

        average_cpu_time += time;
	}
	average_cpu_time /= passes;
    printf("	done. CPU time cost in second: %f\n", average_cpu_time);
	
	free(dataC);
	
	return average_cpu_time;
}

int main( int argc, char *argv[] )  
{
	FILE *oldImageFile;
	FILE *newImageFileGPU;
	FILE *newImageFileCPU;
	int *numRows;
	int *numCols;
	int *h_dataA;
	int *h_dataB;
	float *h_orientation;
	char **header;
	
	int *seq_dataB;
	float *seq_orientation;
	
	int passes = 10;
	
	if( argc == 8 )
	{
		int threadsPerBlock = atoi(argv[4]);
		double sigma = atof(argv[5]);
		int lower_thresh = atoi(argv[6]);
		int upper_thresh = atoi(argv[7]);
		double *guassian = computeGaussian(sigma);
		
		if ((oldImageFile = fopen(argv[1], "r")) && (newImageFileGPU = fopen(argv[2], "w")) && (newImageFileCPU = fopen(argv[3], "w"))) {
			numRows = (int*)malloc(sizeof(int));
			numCols = (int*)malloc(sizeof(int));
			header = initAra2D(rowsInHeader, 2);
			h_dataA = pgmRead(header, numRows, numCols, oldImageFile);
			h_dataB = (int *)malloc(*numRows*(*numCols)*sizeof(int));
			h_orientation = (float *)malloc(*numRows*(*numCols)*sizeof(float));
			float GPUTime = runCannyGPU(h_dataA, h_dataB, h_orientation, numCols, numRows, threadsPerBlock, sigma, lower_thresh, upper_thresh, guassian, passes);
			int errorCodeWriteGPU = pgmWrite((const char **)header, h_dataB, *numRows, *numCols, newImageFileGPU);
			
			
			seq_dataB = (int *)malloc(*numRows*(*numCols)*sizeof(int));
			seq_orientation = (float *)malloc(*numRows*(*numCols)*sizeof(float));
			float CPUTime = runCannyCPU(h_dataA, seq_dataB, seq_orientation, numCols, numRows, sigma, lower_thresh, upper_thresh, guassian, passes);
			int errorCodeWriteCPU = pgmWrite((const char **)header, seq_dataB, *numRows, *numCols, newImageFileCPU);
			free(numRows);
			free(numCols);
			freeHeader(header, rowsInHeader);
			free(h_dataA);
			free(h_dataB);
			free(h_orientation);
			free(seq_dataB);
			free(seq_orientation);
			
			printf("speedup: %f \n", CPUTime/GPUTime);
		}
		
		else {
			if(oldImageFile) {
				fclose(oldImageFile);
			}
			if(newImageFileGPU) {
				fclose(newImageFileGPU);
			}
			if(newImageFileCPU) {
				fclose(newImageFileCPU);
			}
			usage();
		}
	}
	
	else {
		usage();
	}
}
