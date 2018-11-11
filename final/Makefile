myPaint: main.o timing.o pgmUtility.o pgmProcess.o
	nvcc -arch=sm_52 -o myPaint main.o timing.o pgmUtility.o pgmProcess.o -I.

pgmUtility.o: pgmUtility.cu
	nvcc -arch=sm_52 -c pgmUtility.cu

pgmProcess.o: pgmProcess.cu
	nvcc -arch=sm_52 -c pgmProcess.cu

main.o: main.c
	g++ -c -o main.o main.c -I.

timing.o: timing.c
	g++ -c -o timing.o timing.c -I.

clean:
	rm -r *.o myPaint
