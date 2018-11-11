#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmUtility.h"
#include "timing.h"

void usage()
{
	printf("Usage: ./programName oldImageFile newImageFile\n");
	exit(1);
}

void runSobel(FILE *oldImageFile, FILE *newImageFile) {
	int *numRows = (int*)malloc(sizeof(int));
	int *numCols = (int*)malloc(sizeof(int));
	char **header = initAra2D(rowsInHeader, 2);
	int *data = pgmRead(header, numRows, numCols, oldImageFile);
	int errorCodeWrite = pgmWrite((const char **)header, data, *numRows, *numCols, newImageFile);
	free(numRows);
	free(numCols);
	freeHeader(header, rowsInHeader);
	free(data);
}

int main( int argc, char *argv[] )  
{
	FILE *oldImageFile;
	FILE *newImageFile;
	
	if( argc == 3 )
	{
		if ((oldImageFile = fopen(argv[1], "r")) && (newImageFile = fopen(argv[2], "w"))) {
			runSobel(oldImageFile, newImageFile);
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
