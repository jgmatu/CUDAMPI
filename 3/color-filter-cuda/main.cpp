#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <errno.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "timer.h"
#include <mpi.h>

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

const int SIZE_FILTER = 25;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

/*******  DEFINED IN func.cu *********/
void create_filter(float **h_filter, const float *mask, const int size);

void convolution(uchar4 * const d_inputImageRGBA, uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered,
                        unsigned char *d_greenFiltered,
                        unsigned char *d_blueFiltered,
                        const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth);

void open_mpi_separate_channels(uchar4* const d_inputImageRGBA,
                                    const size_t numRows,
                                    const size_t numCols,
                                    unsigned char *d_red,
                                    unsigned char *d_green,
                                    unsigned char *d_blue);

void open_mpi_box_filter(const unsigned char *channel,
                        unsigned char *filter_channel,
                        const size_t numRows,
                        const size_t numCols,
                        float* d_filter,
                        const int filterWidth);

void open_mpi_recombine_channels(const unsigned char* d_redFiltered,
                                    const unsigned char* d_greenFiltered,
                                    const unsigned char* d_blueFiltered,
                                    uchar4* const d_outputImageRGBA,
                                    const size_t numRows,
                                    const size_t numCols);

// ****************************************************************************
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
// ****************************************************************************

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
      if (err != cudaSuccess) {
            std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            exit(1);
      }
}

// Return types are void since any internal error will be handled by quitting
// no point in returning error codes...
// returns a pointer to an RGBA version of the input image
// and a pointer to the single channel grey-scale output
// on both the host and device.
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redFiltered,
                unsigned char **d_greenFiltered,
                unsigned char **d_blueFiltered,
                const std::string &filename)
{
      // make sure the context initializes ok...
      checkCudaErrors(cudaFree(0));

      cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
      if (image.empty()) {
            std::cerr << "Couldn't open file: " << filename << std::endl;
            exit(1);
      }
      cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

      // Allocate memory for the output...
      imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

      // This shouldn't ever happen given the way the images are created
      // at least based upon my limited understanding of OpenCV, but better to check...
      if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
            std::cerr << "Images aren't continuous!! Exiting." << std::endl;
            exit(1);
      }
      *h_inputImageRGBA  = (uchar4 *) imageInputRGBA.ptr<unsigned char>(0);
      *h_outputImageRGBA = (uchar4 *) imageOutputRGBA.ptr<unsigned char>(0);

      const size_t numPixels = numRows() * numCols();

      // Allocate memory on the device for both input and output...
      checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
      checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
      checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

      // copy input array to the GPU...
      checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

      d_inputImageRGBA__  = *d_inputImageRGBA;
      d_outputImageRGBA__ = *d_outputImageRGBA;

      checkCudaErrors(cudaMalloc(d_redFiltered, sizeof(unsigned char) * numPixels));
      checkCudaErrors(cudaMalloc(d_greenFiltered, sizeof(unsigned char) * numPixels));
      checkCudaErrors(cudaMalloc(d_blueFiltered, sizeof(unsigned char) * numPixels));
      checkCudaErrors(cudaMemset(*d_redFiltered, 0, sizeof(unsigned char) * numPixels));
      checkCudaErrors(cudaMemset(*d_greenFiltered, 0, sizeof(unsigned char) * numPixels));
      checkCudaErrors(cudaMemset(*d_blueFiltered, 0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
      cv::Mat output(numRows(), numCols(), CV_8UC4, (void*) data_ptr);

      cv::Mat imageOutputBGR;
      cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);

      // Output the image...
      cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanUp(void)
{
      cudaFree(d_inputImageRGBA__);
      cudaFree(d_outputImageRGBA__);
}

// An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster
//    than the naive implementation in reference_calc.cpp.
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size)
{
	cv::Mat input = cv::imread(input_file);

	// Create an identical image for the output as a placeholder...
	cv::Mat reference = cv::imread(input_file);
	cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size),0);
	cv::imwrite(reference_file, reference);
}

void debug_image(char *filename, unsigned char *pixels)
{
      cv::Mat output(numRows(), numCols(), CV_8UC1, (void*) pixels);

      // Output the image...
      cv::imwrite(filename, output);
}

void debug_devices_data(char *filename, unsigned char *d_data)
{
      int size = sizeof(unsigned char) * numRows() * numCols();
      unsigned char *h_data = (unsigned char*) malloc(size);
      if (!h_data) {
            std::cerr << "Error allocating memory to debug image..." << strerror(errno) << '\n';
            exit(1);
      }
      cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
      debug_image(filename, h_data);
      free(h_data);
}

void sendDataUchar(const unsigned char *data_device, const int size, const int id)
{
      MPI_Send(data_device, size, MPI::UNSIGNED_CHAR, id, 0, MPI_COMM_WORLD);
}

void receiveDataUchar(unsigned char* data_device, const int size, const int id)
{
      MPI_Status stat;

      MPI_Recv(data_device, size, MPI::UNSIGNED_CHAR, id, 0, MPI_COMM_WORLD, &stat);
}

void sendFilter(const float *d_filter, const int size, const int id)
{
      MPI_Send(d_filter, size, MPI::FLOAT, id, 0, MPI_COMM_WORLD);
}

void receiveFilter(float* d_filter, const int size, const int id)
{
      MPI_Status stat;

      MPI_Recv(d_filter, size, MPI::FLOAT, id, 0, MPI_COMM_WORLD, &stat);
}

void master(uchar4 *const d_inputImageRGBA, uchar4 *const d_outputImageRGBA,
                        unsigned char *d_red,
                        unsigned char *d_green,
                        unsigned char *d_blue)
{
      float *d_filter;
      const float mask[] = {
             0.0f,  0.0f, -1.0,   0.0f,  0.0f,
             0.0f, -1.0f, -2.0f, -1.0f,  0.0f,
            -1.0f, -2.0f, 16.0f, -2.0f, -1.0f,
             0.0f, -1.0f, -2.0f, -1.0f,  0.0f,
             0.0f,  0.0f, -1.0, 0.0f, 0.0f
      };

      create_filter(&d_filter, mask, SIZE_FILTER);
      open_mpi_separate_channels(d_inputImageRGBA, numRows(), numCols(), d_red, d_green, d_blue);

      sendDataUchar(d_red, numRows() * numCols(), 1);
      sendFilter(d_filter, SIZE_FILTER, 1);

      sendDataUchar(d_green, numRows() * numCols(), 2);
      sendFilter(d_filter, SIZE_FILTER, 2);

      sendDataUchar(d_blue, numRows() * numCols(), 3);
      sendFilter(d_filter, SIZE_FILTER, 3);

      receiveDataUchar(d_red, numRows() * numCols(), 1);
      receiveDataUchar(d_green, numRows() * numCols(), 2);
      receiveDataUchar(d_blue, numRows() * numCols(), 3);

      open_mpi_recombine_channels(d_red, d_green, d_blue, d_outputImageRGBA, numRows(), numCols());
}



void slave_channel_filter(int id)
{
      unsigned char *d_channel;
      unsigned char *d_channel_filtered;
      float *d_filter;
      unsigned sizeImage = sizeof(unsigned char) * numCols() * numRows();

      fprintf(stderr, "Node channel... : %d\n", id);
      cudaMalloc(&d_channel_filtered, sizeImage);
      cudaMalloc(&d_channel, sizeImage);
      cudaMalloc(&d_filter, sizeof(float) * SIZE_FILTER);
      cudaMemset(d_channel_filtered, 0, sizeImage);

      receiveDataUchar(d_channel, numRows() * numCols(), 0);
      receiveFilter(d_filter, SIZE_FILTER, 0);

      open_mpi_box_filter(d_channel, d_channel_filtered, numRows(), numCols(), d_filter, SIZE_FILTER);

      sendDataUchar(d_channel_filtered, numRows() * numCols(), 0);

      char debug_msg[512];
      sprintf(debug_msg, "channel_%d.png", id);
      debug_devices_data(debug_msg, d_channel_filtered);
      cudaFree(d_channel);
}

int mpi_comm_devices(int argc, char* argv[],
                        uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA,
                        unsigned char *d_redFiltered,
                        unsigned char *d_greenFiltered,
                        unsigned char *d_blueFiltered)
{
      int id;
      int numprocs;

      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &id);

      MPI_Barrier(MPI_COMM_WORLD); // Wait all process ready...
      if (id == 0) {
            master(d_inputImageRGBA, d_outputImageRGBA, d_redFiltered, d_greenFiltered, d_blueFiltered);
      } else {
            slave_channel_filter(id);
      }
      return id;
}

void write_results(uchar4 *h_outputImageRGBA, uchar4 *d_outputImageRGBA, std::string output_file) {
      size_t numPixels = numRows() * numCols();

      checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
      postProcess(output_file, h_outputImageRGBA);
}

/*******  Begin main *********/
int main(int argc, char **argv) {
      uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
      uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
      unsigned char *d_redFiltered, *d_greenFiltered, *d_blueFiltered;


      std::string input_file;
      std::string output_file;
      std::string reference_file;

      switch (argc)
      {
            case 2:
            input_file = std::string(argv[1]);
            output_file = "output.png";
            break;

            case 3:
            input_file  = std::string(argv[1]);
            output_file = std::string(argv[2]);
            break;

            default:
            std::cerr << "Usage: ./box_filter input_file [output_filename]" << std::endl;
            exit(1);
      }
      // Load the image and give us our input and output pointers
      preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA, &d_redFiltered, &d_greenFiltered, &d_blueFiltered,
            input_file);

      GpuTimer timer;
      timer.Start();

      int id = mpi_comm_devices(argc, argv, d_inputImageRGBA, d_outputImageRGBA, d_redFiltered, d_greenFiltered, d_blueFiltered);

      timer.Stop();

      cudaDeviceSynchronize();
      checkCudaErrors(cudaGetLastError());

      int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
      if (err < 0) {
            std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
            exit(1);
      }

      if (id == 0) {
            write_results(h_outputImageRGBA, d_outputImageRGBA, output_file);
      }

      checkCudaErrors(cudaFree(d_redFiltered));
      checkCudaErrors(cudaFree(d_greenFiltered));
      checkCudaErrors(cudaFree(d_blueFiltered));
      cleanUp();
      MPI_Finalize();
      return 0;
}
