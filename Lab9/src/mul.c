
#include "mul.h"


/* Multiply the nxn matrices a and b, placing the result in c. Matrices
 * are passed as 1D arrays of floats, because we don't know the dimensions
 * of the matrices at compile time. The last argument specifies the length
 * of one side of the matrix.
 */
void mul(float c[], float a[], float b[], int n){

   float sum;
   int di, dj, i;
   

   for(di = 0; di < n; di ++){
      for(dj = 0; dj < n; dj ++){
      
         sum=0;
         for(i = 0; i < n; i++){
         
            sum += a[di * n + i] * b[i * n + dj];
         }
         
         c[di * n + dj] = sum;
      }
   }
   
}
