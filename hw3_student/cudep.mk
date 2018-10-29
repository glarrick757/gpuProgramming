jacobi.o: jacobi.cu /usr/local/cuda-8.0/samples/common/inc/helper_cuda.h \
 /usr/local/cuda-8.0/samples/common/inc/helper_string.h \
 /usr/local/cuda-8.0/samples/common/inc/helper_timer.h \
 /usr/local/cuda-8.0/samples/common/inc/exception.h kernel.h kernel1.h
kernel.o: kernel.cu kernel.h
kernel1.o: kernel1.cu kernel1.h
