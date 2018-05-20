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

__global__ void box_filter(const unsigned char* const inputChannel, unsigned char* const outputChannel,
                        int numRows, int numCols, const float* __restrict__ filter, const int filterWidth)
{
      // Aplicar el filtro a cada pixel de la imagen...

      // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
      // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      if (idx >= numCols || idy >= numRows) return;

      float c = 0.0f;
      for (int fx = 0; fx < filterWidth; ++fx) {
            for (int fy = 0; fy < filterWidth; ++fy) {
                  int imagex = idx + fx - filterWidth / 2;
                  int imagey = idy + fy - filterWidth / 2;
                  imagex = min(max(imagex, 0), numCols - 1); // Limit image on borders...
                  imagey = min(max(imagey, 0), numRows - 1); // Limit image on borders...
                  c += (filter[fy * filterWidth + fx] * inputChannel[imagey * numCols + imagex]);
            }
      }
      outputChannel[idy * numCols + idx] = c;
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
__global__ void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
      const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
      const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

      // make sure we don't try and access memory outside the image
      //by having any threads mapped there return early...
      if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

      unsigned char red   = redChannel[thread_1D_pos];
      unsigned char green = greenChannel[thread_1D_pos];
      unsigned char blue  = blueChannel[thread_1D_pos];

      // Alpha should be 255 for no transparency
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);
      outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth)
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
void create_filter(float **d_filter, const float *mask, const int size)
{
      float *h_filter = (float *) malloc(sizeof(float) * size);
      if (!h_filter) {
            std::cerr << "Error creating filter.." << strerror(errno) << '\n';
            exit(1);
      }
      for (int i = 0; i < size; ++i) {
            h_filter[i] = mask[i];
      }

      cudaMalloc(d_filter, sizeof(float) * size);
      cudaMemcpy(*d_filter, h_filter, sizeof(float) * size, cudaMemcpyHostToDevice);
}

void open_mpi_separate_channels(uchar4* const d_inputImageRGBA,
                                    const size_t numRows,
                                    const size_t numCols,
                                    unsigned char *d_red,
                                    unsigned char *d_green,
                                    unsigned char *d_blue)
{
      const dim3 blockSize(NTHREADS, NTHREADS, 1);
      const dim3 gridSize((numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);

      separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red,  d_green, d_blue);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void open_mpi_box_filter(const unsigned char *channel,
                        unsigned char *filter_channel,
                        const size_t numRows,
                        const size_t numCols,
                        float* d_filter,
                        const int filterWidth)
{
      const dim3 blockSize(NTHREADS, NTHREADS, 1);
      const dim3 gridSize((numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);

      box_filter<<<gridSize, blockSize>>>(channel, filter_channel, numRows, numCols, d_filter, filterWidth);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void open_mpi_recombine_channels(const unsigned char* d_redFiltered,
                                    const unsigned char* d_greenFiltered,
                                    const unsigned char* d_blueFiltered,
                                    uchar4* const d_outputImageRGBA,
                                    const size_t numRows,
                                    const size_t numCols)
{
      const dim3 blockSize(NTHREADS, NTHREADS, 1);
      const dim3 gridSize((numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);

      recombineChannels<<<gridSize, blockSize>>>(d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void convolution(uchar4* const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows,
                        const size_t numCols,
                        unsigned char *d_redFiltered,
                        unsigned char *d_greenFiltered,
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{
      const dim3 blockSize(NTHREADS, NTHREADS, 1);
      const dim3 gridSize((numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);

      separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red,  d_green, d_blue);

      // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
      // launching your kernel to make sure that you didn't make any mistakes.
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      box_filter<<<gridSize, blockSize>>>(d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
      box_filter<<<gridSize, blockSize>>>(d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);
      box_filter<<<gridSize, blockSize>>>(d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);

      // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
      // launching your kernel to make sure that you didn't make any mistakes.
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      recombineChannels<<<gridSize, blockSize>>>(d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

//Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
      checkCudaErrors(cudaFree(d_red));
      checkCudaErrors(cudaFree(d_green));
      checkCudaErrors(cudaFree(d_blue));
      checkCudaErrors(cudaFree(d_filter));
}
