MPICC = mpicc
CFLAGS = -c -g -Wall -Wshadow
NVCC = nvcc
NVFLAGS = -O2 -D_FORCE_INLINES -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi -arch=sm_30 -gencode arch=compute_30,code=sm_30 -g -G

PROGRAMS = master slave
OBJECT = master.o slave.o

all:
	$(NVCC) $(NVFLAGS) -o master master.cu
	$(NVCC) $(NVFLAGS) -o slave slave.cu

clean:
	rm slave master
