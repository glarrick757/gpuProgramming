#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define R 5
#define C 40 

__device__ int gpu_strlen(char * s)
{
    int i = 0;
    while(s[i++] != '\0')
    {
    }
    return i;
}

__device__ int gpu_isAlpha(char ch)
{
    if((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
        return 1;
    else
        return 0;
}

__global__ void wordCount2( char **a, int **out, int numLine, int maxLineLen )
{
	__extern__ shared char[] line;
	
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	
	line[threadIdx.x] = a[row][col];
	
	if(row < numLine && col < maxLineLen && col < length) {
		alphaValue = gpu_isAlpha(line[row][col]);
		
		if(alphaValue == 1) {
			prevAlphaValue = gpu_isAlpha(line[row][col - 1]);
			if(prevAlphaValue == 1) {
				out[row][col] = 0;
			}
			else {
				out[row][col] = alphaValue;
			}
		}
		else {
			out[row][col] = alphaValue;
		}
	} 
}

void checkErr()
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
        printf ("Cuda error -- %s\n", cudaGetErrorString(code));
}

void printArr( char **a, int lines )
{
    int i;
    for(i=0; i<lines; i++)
    {
        printf("%s\n", a[i]);
    }
}


int main()
{
    int i, j; 
    char **d_in, **h_in, **h_out;
    int h_count_in[R][C], **h_count_out, **d_count_in;

    for(i = 0; i < R; i++)
        for(j = 0; j < C; j ++)
            h_count_in[i][j] = 0;

    //allocate
    h_in = (char **)malloc(R * sizeof(char *));
    h_out = (char **)malloc(R * sizeof(char *));
    h_count_out = (int **)malloc(R * sizeof(int *));
  
    cudaDeviceReset();

    cudaMalloc((void ***)&d_in, sizeof(char *) * R);
    cudaMalloc((void ***)&d_count_in, sizeof(int *) * R);

    //alocate for string data
    for(i = 0; i < R; ++i) 
    {
        cudaMalloc((void **) &h_out[i],C * sizeof(char));
        h_in[i]=(char *)calloc(C, sizeof(char));//allocate or connect the input data to it
        strcpy(h_in[i], "good morning and I'm a good student!");
        cudaMemcpy(h_out[i], h_in[i], strlen(h_in[i]) + 1, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_in, h_out, sizeof(char *) * R,cudaMemcpyHostToDevice);

    //alocate for output occurrence
    for(i = 0; i < R; ++i) 
    {
        cudaMalloc((void **) &h_count_out[i], C * sizeof(int));
        cudaMemset(h_count_out[i], 0, C * sizeof(int));
    }
    cudaMemcpy(d_count_in, h_count_out, sizeof(int *) * R,cudaMemcpyHostToDevice);

    printArr(h_in, R);
    printf("\n\n");
     
    //set up kernel configuration variables
    dim3 grid, block;
    block.x = C;   //NOTE: differs from last lab6 here, Why?
    block.y = 1;
    grid.x  = ceil((float)C / block.x);
    grid.y  = ceil((float)R / block.y); //careful must be type cast into float, otherwise, integer division used
    //printf("grid.x = %d, grid.y=%d\n", grid.x, grid.y );

    //launch kernel
    wordCount2<<<grid, block>>>( d_in, d_count_in, R, C);
    checkErr();

    //copy data back from device to host
    for(i = 0; i < R; ++i) {
        cudaMemcpy(h_count_in[i], h_count_out[i], sizeof(int) * C,cudaMemcpyDeviceToHost);
    }

    printf("Occurrence array obtained from device:\n");

    for(i = 0; i < R; i ++) {
        for(int j = 0; j < C; j ++)
            printf("%4d", h_count_in[i][j]);
        printf("\n");
    }
 
    return 0;
}

