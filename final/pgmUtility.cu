#include "pgmUtility.h"

char ** initAra2D(int rows, int cols) {
	int i = 0;
	char **arr = (char **)malloc(rows * sizeof(char *)); 
    	for (i=0; i<rows; i++) {
        	arr[i] = (char *)malloc(cols * maxSizeHeadRow); 
	}
	return arr;
}

int * parseIntArray(char *line, int len) {
	int *parseVals = (int *)malloc(sizeof(int)*len);
	const char *s = " ";
	int i;

	//get the first token
	char *token = strtok(line, s);
	   
	//walk through other tokens
	for(i = 0; token != NULL; i++ ) {
		parseVals[i] = atoi(token);
		token = strtok(NULL, s);
	}
	
	return parseVals;
}

void freeHeader(char** header, int rows) {
	int i;
	
	for(i = 0; i < rows; i++) {
		free(header[i]);
	}
	
	free(header);
}

int * pgmRead( char **header, int *numRows, int *numCols, FILE *in  ) {
	int *data;
	int count = 0;
	char *str = (char *)malloc(sizeof(char) * 100);
	int i;
	int j;
	int number;
	
	//read header data
	for(i = 0; i < rowsInHeader && fgets (str, 60, in)!=NULL; i++) {
		if(i == 2) {
			strcpy(header[count], str);
			int *dimensions = parseIntArray(str, 2);
			*numRows = dimensions[1];
			*numCols = dimensions[0];
			free(dimensions);
		}
		
		else {
			strcpy(header[count], str);
		}
		count++;
	} 
	
	data = (int *)malloc(*numRows*(*numCols)*sizeof(int));
	
	//read image data
	while((count - 4) < *numRows) {
		for(j = 0; j < *numCols; j++) {
			fscanf(in, "%d", &number);
			data[(count - 4)*(*numCols) + j] = number;
		}
		count++;
	}
	
	free(str);
	fclose(in);
	
	return data;
}

int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out ) {
	//if(sizeof(*header)/sizeof(char) != rowsInHeader) {
	//	return 0;
	//}

	//print header information
	fprintf(out, "%s", header[0]);
	fprintf(out, "%s", header[1]);
	fprintf(out, "%s", header[2]);
	fprintf(out, "%s\n", "255");
	
	int i;
	int j;
	//print intensity values
	for(i = 0; i < numRows; i++) {
		for(j = 0; j < numCols; j++) {
			fprintf(out, "%d ", pixels[(i * numCols) + j]);
		}
		fprintf(out, "\n");
	}
	
	fclose(out);
	return 1;
}

double *computeGaussian(double sigma) {
	double *guassian = (double *)malloc(sizeof(double) * 9);
	int row, col, x, y;

	for(row = 0; row < 3; row++) {
		for(col = 0; col < 3; col++) {
			x = abs(col - 1);
			y = abs(row - 1); 
			guassian[row * 3 + col] = (1 / (2 * PI * sigma * sigma)) * exp(-1 * (x * x + y * y) / (2 * sigma * sigma));
		}
	}
	
	return guassian;
}
