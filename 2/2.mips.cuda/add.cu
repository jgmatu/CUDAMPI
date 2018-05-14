#include <cuda.h>
#include <stdio.h>

#define THREADS 16
#define BLOCKS 8


__global__ void __add__(int *array, int *size) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx > *size) return;

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

extern "C" void call_device_sum(int *h_a, int size)
{
      int *dev_a = NULL;
      int *dev_size = NULL;

      cudaMalloc(&dev_size, sizeof(int));
      cudaMemset(dev_size, 0, sizeof(int));
      cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);

      cudaMalloc(&dev_a, (size + 2) * sizeof(int));
      cudaMemset(dev_a, 0, (size + 2) * sizeof(int));
      cudaMemcpy(dev_a, h_a, (size + 2) * sizeof(int), cudaMemcpyHostToDevice);

      __add__ <<<BLOCKS, THREADS>>>(dev_a, dev_size);

      // se transfieren los datos del dispositivo a memoria.
      cudaMemcpy(h_a, dev_a, (size + 2) * sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(dev_a);
      cudaFree(dev_size);
}
