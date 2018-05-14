#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <mpi.h>
#include "add.h"


void slave( int id ) {
      MPI_Status stat;
      int *data, size;

      MPI_Recv(&size, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
      data = malloc(sizeof(int) * (size + 2));
      if (!data) {
            fprintf(stderr, "No data... %s\n", strerror(errno));
            return;
      }
      MPI_Recv(data, size, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
      data[size] = data[0];
      data[size + 1] = data[0];
      call_device_sum(data, size);
      MPI_Send(&data[1], size, MPI_INT, 0, 0, MPI_COMM_WORLD);
      free(data);
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
