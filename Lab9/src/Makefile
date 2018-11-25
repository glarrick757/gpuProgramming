# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = matrix_multiplication.o 
OBJS = mul.o timing.o arrayUtils.o

# make and compile
mmul:$(OBJS) $(GPUOBJS)
	$(NVCC) -o mmul $(OBJS) $(GPUOBJS) 

matrix_multiplication.o: matrix_multiplication.cu
	$(NVCC) -c matrix_multiplication.cu 

mul.o: mul.c
	$(CXX) -c mul.c
timing.o: timing.c
	$(CXX) -c timing.c
arrayUtils.o: arrayUtils.c
	$(CXX) -c arrayUtils.c

clean:
	rm -f *.o
	rm -f mmul
