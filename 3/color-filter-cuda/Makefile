NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

#OPENCV_LIBPATH=/usr/lib
#OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

OPENCV_LIBPATH=/usr/local/lib
OPENCV_INCLUDEPATH=/usr/local/include

MPI_LIBPATH = /usr/lib/openmpi/lib
MPI_INCLUDEPATH = /usr/lib/openmpi/include
MPI_LIBS = -lmpi_cxx -lmpi


OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_INCLUDEPATH=/usr/local/cuda/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS  = -O3 -Xcompiler -Wall -Xcompiler -Wextra -m64 -D_FORCE_INLINES
NVCC_OPTS += -arch=sm_30 -gencode arch=compute_30,code=sm_30 -g -G

GCC_OPTS  = -O3 -Wall -Wextra -m64

all: main.o func.o Makefile
	$(NVCC) -o box_filter main.o func.o -L $(OPENCV_LIBPATH) -L $(MPI_LIBPATH) $(MPI_LIBS) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h
	mpic++ -c main.cpp $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH) -I $(MPI_INCLUDEPATH)

func.o: func.cu
	nvcc -c func.cu $(NVCC_OPTS)

clean:
	rm -f *.o *.png box_filter
