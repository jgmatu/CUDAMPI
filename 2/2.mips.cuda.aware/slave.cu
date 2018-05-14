#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <mpi.h>
#include <cuda.h>

#define BLOCKS 8
#define THREADS 32

__global__ void __add__(int *array, int *size) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx > *size) return;

      array[*size] = array[0];
      array[*size + 1] = array[0];

      int temp = 0;
      int before = (idx + 1) % *size;
      int after = idx - 1;
      if (after < 0) after = *size - 1;


      temp += array[idx];
      temp += array[before];
      temp += array[after];

      __syncthreads(); // Barrera...
      array[idx] = temp;
}

void slave( int id ) {
      MPI_Status stat;
      int *data;
      int *dev_size;

      cudaMalloc(&dev_size, sizeof(int));
      MPI_Recv(dev_size, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);

      int size = -1;
      cudaMemcpy(&size, dev_size, sizeof(int), cudaMemcpyDeviceToHost);

      cudaMalloc(&data, size * sizeof(int));

      MPI_Recv(data, size, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
      __add__ <<<BLOCKS, THREADS>>>(data, dev_size);
      MPI_Send(data, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
      cudaFree(data);
}


int main(int argc, char* argv[]) {
      int id;
      int numprocs;

      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &id);
      MPI_Barrier(MPI_COMM_WORLD); // Wait all process ready...

      if (id != 0) {
            slave(id);
      }
      MPI_Finalize();
      exit(0);
}
