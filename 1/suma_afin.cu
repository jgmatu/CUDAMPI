#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <cuda.h>

#define THREADS 128
#define BLOCKS 16
#define SIZE 2048

__global__ void add(int *array) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx > SIZE) return;

      int temp = 0;
      int before = (idx + 1) % SIZE;
      int after = idx - 1;
      if (after < 0) after = SIZE - 1;


      temp += array[idx];
      temp += array[before];
      temp += array[after];

      __syncthreads(); // Barrera...
      array[idx] = temp;
}

void  init(int* h_v, int numb) {
      for (int i = 0; i < SIZE; i++) {
            h_v[i] = numb;
      }
}

int main( void ) {
      int *result, *h_a;

      int *dev_a;
      int size = SIZE * sizeof(int);

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );
      if (h_a == NULL || result == NULL) {
            fprintf(stderr, "Error allocating memory... %s\n", strerror(errno));
            exit(1);
      }
      memset(result, 0, size);
      init(h_a, 3);

      cudaMalloc(&dev_a, size);

      // se transfieren los datos a memoria de dispositivo...
      cudaMemcpy(dev_a, h_a, size, cudaMemcpyHostToDevice);

      add<<<BLOCKS, THREADS>>>(dev_a);

      // se transfieren los datos del dispositivo a memoria.
      cudaMemcpy(result, dev_a, size, cudaMemcpyDeviceToHost);

      fprintf(stdout, "Result %s\n", "");
      for (int i = 0; i < SIZE; i++) {
            fprintf(stderr, " %d ", result[i]);
            if ((i + 1) % 10 == 0) fprintf(stdout, "%s\n", "");
      }
      fprintf(stdout, "%s\n", "");
      free(h_a), free(result);
      cudaFree(dev_a);
      return 0;
}
