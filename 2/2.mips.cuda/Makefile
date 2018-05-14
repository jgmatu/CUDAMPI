MPICC = mpicc
CFLAGS = -c -g -Wall -Wshadow
NVCC = nvcc
NVFLAGS = -c -arch=sm_20 -gencode arch=compute_20,code=sm_20 -g -G

PROGRAMS = master slave
OBJECT = master.o slave.o

all:
	$(NVCC) $(NVFLAGS) add.cu -o add.o
	$(MPICC) $(CFLAGS) master.c -o master.o
	$(MPICC) $(CFLAGS) slave.c -o slave.o
	mpicc master.o add.o -lstdc++ -lcudart -L/usr/local/cuda/lib64 -o master
	mpicc slave.o add.o -lstdc++ -lcudart -L/usr/local/cuda/lib64 -o slave
	rm slave.o master.o add.o

clean:
	rm slave master
