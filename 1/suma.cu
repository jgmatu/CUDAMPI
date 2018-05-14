#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define THREADS 64
#define BLOCKS 16
#define SIZE 512

__global__ void add(int* a, int* b, int* c)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx > SIZE) return;
      c[idx] = a[idx] + b[idx];
}

void  init(int* h_v, int numb) {
      for (int i = 0; i < SIZE; i++) {
            h_v[i] = numb;
      }
}

int main( void ) {
      int *result, *h_a, *h_b;

      int *dev_a, *dev_b, *dev_c;
      int size = SIZE * sizeof(int);

      result = (int*) malloc( size );
      h_a = (int*) malloc( size );
      h_b = (int*) malloc( size );

      init(h_a, 5);
      init(h_b, 5);
      memset(result, 0, size);

      cudaMalloc( &dev_a, size );
      cudaMalloc( &dev_b, size );
      cudaMalloc( &dev_c, size );

      // se transfieren los datos a memoria de dispositivo.
      cudaMemcpy( dev_a, h_a, size, cudaMemcpyHostToDevice );
      cudaMemcpy( dev_b, h_b, size, cudaMemcpyHostToDevice );
      cudaMemset( dev_c, 0, size );

      add<<<BLOCKS, THREADS>>>( dev_a, dev_b, dev_c );

      // se transfieren los datos del dispositivo a memoria.
      cudaMemcpy( result, dev_c, size, cudaMemcpyDeviceToHost );

      for (int i = 0; i < SIZE; i++) {
            fprintf(stdout, " %d ", result[i]);
            if ((i + 1) % 10 == 0) fprintf(stdout, "%s\n", "");
      }
      fprintf(stdout, "%s\n", "");
      free(h_a), free(h_b), free(result);
      cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
      return 0;
}
