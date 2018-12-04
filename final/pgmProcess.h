#ifndef pgmProcess_h
#define pgmProcess_h
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */ 

__global__ void gaussianBlur(int *d_pixelsIn, int *d_pixelsOut, double *guassian, int floatpitch, int width, int height); 

__global__ void pgmSobel(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width, int height);

__global__ void pgmNonMaximumSupression(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width, int height);

__global__ void pgmHysterisisThresholding(int *d_pixelsIn, int *d_pixelsOut, int lower_thresh, int upper_thresh, int floatpitch, int width, int height);

__device__ __host__ void pushNeighbors(int currentPixelIndex, int lower_thresh, int upper_thresh, int floatpitch, int width, int height, int *stack, int *top, int capacity, int *d_pixelsIn, int *d_pixelsOut);

__device__ __host__ int isFull(int *top, unsigned capacity);

__device__ __host__ int isEmpty(int *top) ;

__device__ __host__ void push(int *stack, int *top, int capacity, int item);

__device__ __host__ int pop(int *stack, int *top, unsigned capacity);

__host__ void pgmSobelSqequential(int *pixelsIn, int *pixelsOut, float *orientation, int width, int height);

__host__ void pgmGuassianBlurSequential(int *pixelsIn, int *pixelsOut, double *guassian, int width, int height);

__host__ void pgmNonMaximumSupressionSequential(int *pixelsIn, int *pixelsOut, float *orientation, int width, int height);

__host__ void pgmHysterisisThresholdingSequential(int *pixelsIn, int *pixelsOut, int lower_thresh, int upper_thresh, int width, int height);
#endif
