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
__global__ void pgmSobel(int *d_pixelsIn, int *d_pixelsOut, float *d_orientation, int floatpitch, int width)

#endif
