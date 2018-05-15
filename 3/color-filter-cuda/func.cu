//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <algorithm>    // std::max
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

typedef unsigned char uchar;

#define FILTER_WIDTH 3
#define NTHREADS 32

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
      if (err != cudaSuccess) {
            std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            exit(1);
      }
}

__constant__ float filtro[FILTER_WIDTH * FILTER_WIDTH];

__global__ void box_filter(const unsigned char* const inputChannel,
                        unsigned char* const outputChannel,
                        int numRows, int numCols,
                        const float* __restrict__ filter, const int filterWidth)
{
      // Aplicar el filtro a cada pixel de la imagen

      // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
      // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= numCols || idy >= numRows) return;

        int id = idy * numCols + idx;
        redChannel[id] = inputImageRGBA[id].x;
        greenChannel[id] = inputImageRGBA[id].y;
        blueChannel[id] = inputImageRGBA[id].z;
 }

// This kernel takes in three color channels and recombines them
// into one image. The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;

      //make sure we don't try and access memory outside the image
      //by having any threads mapped there return early
      if (idx >= numCols || idy >= numRows) return;
      int id = idy * numCols + idx;

      unsigned char red   = redChannel[id];
      unsigned char green = greenChannel[id];
      unsigned char blue  = blueChannel[id];

      // Alpha should be 255 for no transparency
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);
      outputImageRGBA[id] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
      int sizeImg = sizeof(uchar4) * numRowsImage * numColsImage;
      int sizeFilter = sizeof(uchar4) * filterWidth * filterWidth;

      cudaMalloc(&d_red, sizeImg);
      cudaMalloc(&d_green, sizeImg);
      cudaMalloc(&d_blue, sizeImg);
      cudaMalloc(&d_filter, sizeFilter);

      cudaMemcpy(d_filter, h_filter, sizeFilter, cudaMemcpyHostToDevice);
      cudaMemset(d_red, 0, sizeImg);
      cudaMemset(d_green, 0, sizeImg);
      cudaMemset(d_blue, 0, sizeImg);
}

// Crear el filtro se que va a aplicar (en CPU) y almacenar su tamaño...
void create_filter(float **h_filter, int *filterWidth)
{
      float *filter = (float *) malloc(sizeof(float) * FILTER_WIDTH * FILTER_WIDTH);
      if (!h_filter) {
            std::cerr << "Error creating filter.." << strerror(errno) << '\n';
            exit(1);
      }
      float mask[] = {
            1, 0, 1,
            0, 1, 0,
            1, 0, 1
      };
      for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; ++i) {
            filter[i] = mask[i];
      }
      *h_filter = filter;
      *filterWidth = FILTER_WIDTH;
}

void convolution(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered,
                        unsigned char *d_greenFiltered,
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{
      const dim3 blockSize(NTHREADS, NTHREADS, 1);
      const dim3 gridSize((numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);

      fprintf(stderr, "Num blocks : (x:%d, y:%d, z:%d)\n", gridSize.x, gridSize.y);
      separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_redFiltered, d_greenFiltered, d_blueFiltered);

      recombineChannels<<<gridSize, blockSize>>>(d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA,
                                                   numRows,
                                                   numCols);

	// Separar la imagen en sus canales rgb (separateChannels)
	// aplicar el filtro a cada canal (create_filter) (box_filter)
	// volver a juntar la imagen (recombineChannels)...
}


//Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
      checkCudaErrors(cudaFree(d_red));
      checkCudaErrors(cudaFree(d_green));
      checkCudaErrors(cudaFree(d_blue));
      checkCudaErrors(cudaFree(d_filter));
}
