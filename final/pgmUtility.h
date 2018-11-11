//
//  pgmUtility.h
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#ifndef cscd439pgm_pgmUtility_h
#define cscd439pgm_pgmUtility_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define rowsInHeader 4      // number of rows in image header
#define maxSizeHeadRow 200  // maximal number characters in one row in the header

int * parseIntArray(char *line, int len);
char ** initAra2D(int rows, int cols);
void freeHeader(char** header, int rows);

/**
 *  Function Name: 
 *      pgmRead()
 *      pgmRead() reads in a pgm image using file I/O, you have to follow the file format. All code in this function are exectured on CPU.
 *      
 *  @param[in,out]  header  holds the header of the pgm file in a 2D character array
 *                          After we process the pixels in the input image, we write the origianl 
 *                          header (or potentially modified) back to a new image file.
 *  @param[in,out]  numRows describes how many rows of pixels in the image.
 *  @param[in,out]  numCols describe how many pixels in one row in the image.
 *  @param[in]      in      FILE pointer, points to an opened image file that we like to read in.
 *  @return         If successful, return all pixels in the pgm image, which is an int **, equivalent to
 *                  a 2D array. Otherwise null.
 *
 */
int * pgmRead( char **header, int *numRows, int *numCols, FILE *in  );

/**
 *  Function Name:
 *      pgmWrite()
 *      pgmWrite() writes headers and pixels into a pgm image using file I/O.
 *                 writing back image has to strictly follow the image format. All code in this function are exectured on CPU.
 *
 *  @param[in]  header  holds the header of the pgm file in a 2D character array
 *                          we write the header back to a new image file on disk.
 *  @param[in]  pixels  holds all pixels in the pgm image, which a 2D integer array.
 *  @param[in]  numRows describes how many rows of pixels in the image.
 *  @param[in]  numCols describe how many columns of pixels in one row in the image.
 *  @param[in]  out     FILE pointer, points to an opened text file that we like to write into.
 *  @return     return 0 if the function successfully writes the header and pixels into file.
 *                          else return -1;
 */
int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out );

#endif
