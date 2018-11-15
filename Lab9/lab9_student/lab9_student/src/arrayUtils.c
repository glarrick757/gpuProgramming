/*
 *  matrix.c
 *  MakeMatrix
 *
 */

#include "arrayUtils.h"

#include <stdio.h>
#include <stdlib.h>



void writeArray(float *arr, int rows, int cols, const char * filename){

   int i,j;
   FILE* outfile = fopen(filename, "w");
   
   if(outfile == NULL){
   
      fprintf(stderr, "writeArray(): Couldn\'t open %s for writing.\n", filename);
      return;
   }
   
   // print array dimensions
   fprintf(outfile,"%d %d\n", rows, cols);

   // print array contents
   for(i=0; i<rows; i++){
      for(j=0; j<cols; j++){
      
         fprintf(outfile,"%5.2f ", arr[i*cols + j]);
      }
      fprintf(outfile,"\n");
   }

   fprintf(outfile,"\n");
   fclose(outfile);
}


float * readNewArray(int *rows, int *cols, const char * filename){

   int i,j;
   float f;
   FILE* infile = fopen(filename, "r");
   float * arr;
   
   if(infile == NULL){
   
      fprintf(stderr, "readNewArray(): Couldn\'t open %s for reading.\n", filename);
      return NULL;
   }
   
   // read array dimensions, checking that two values are successfully read.
   if(2 != fscanf(infile," %d %d\n", rows, cols)){
   
      fprintf(stderr, "readNewArray(): Couldn\'t read array dimensions from %s.\n", filename);
      return NULL;      
   }

   //allocate memory
   arr = (float *) malloc( (*rows) * (*cols) * sizeof(float));
   
   if(arr == NULL){
   
      fprintf(stderr, "readNewArray(): Couldn\'t allocate memory.\n");
      return NULL;
   }
   
   // read array contents
   for(i=0; i < *rows; i++){
      for(j=0; j < *cols; j++){
                  
         // Read the value and put it in the array.
         // We could do this in one line, but this way is clearer.
         int status = fscanf(infile," %f", &f );
         arr[i* (*cols) + j] = f;
         
         // check for problems with fscanf()
         if(status != 1){
            
            fprintf(stderr, "readNewArray(): %s lacks sufficient values to fill the matrix.\n",filename);
            free(arr);
            return NULL;
         }
      }
   }

   fclose(infile);
   return arr;
}



void printArray(float *arr, int rows, int cols){

   int i,j;

   for(i=0; i<rows; i++){
      for(j=0; j<cols; j++){
      
         printf("%5.2f ", arr[i*cols + j]);
      }
      printf("\n");
   }

   printf("\n");
}

