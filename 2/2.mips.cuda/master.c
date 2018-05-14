#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <mpi.h>

#define MAX_INT 50
#define MASTER 1

void initData(int* data, int numb)
{
      for (int i = 0; i < MAX_INT; i++) {
            data[i] = numb;
      }
}

void sendData(int* data, int size, int id) {
      MPI_Send(&size, 1, MPI_INT, id, 0, MPI_COMM_WORLD);
      MPI_Send(data, size, MPI_INT, id, 0, MPI_COMM_WORLD);
}

void receiveData(int* data, int size, int id) {
      MPI_Status stat;

      MPI_Recv(data, size, MPI_INT, id, 0, MPI_COMM_WORLD, &stat);
}

void master(int id, int numprocs) {
      int *data = (int *) malloc(sizeof(int) * MAX_INT);
      if (!data) {
            fprintf(stderr, "Error allocating data %s\n", strerror(errno));
            return;
      }
      initData(data, 8);

      int size = MAX_INT / (numprocs - MASTER);
      for (int i = 0; i < numprocs - MASTER; ++i) {
            sendData(&data[i * size], size, i + MASTER);
      }
      for (int i = 0; i < numprocs - MASTER; ++i) {
            receiveData(&data[i * size], size, i + MASTER);
      }

      fprintf(stderr, "%s\n", "Result...");
      for (int i = 0; i < MAX_INT; i++) {
            fprintf(stderr, " %d ", data[i]);
            if ((i + 1) % 10 == 0) fprintf(stderr, "%s\n", "");
      }
      fprintf(stderr, "%s\n", "");
      free(data);
}

int
main(int argc, char *argv[]) {
      int id;
      int numprocs;

      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &id);

      MPI_Barrier(MPI_COMM_WORLD); // Wait all process ready...
      if (id == 0) {
            master(id, numprocs);
      }
      MPI_Finalize();
      exit(0);
}
