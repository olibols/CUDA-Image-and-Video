#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "blur.h"
#include "../image.h"

#define TILE_SIZE 32

__global__ void CudaBoxBlur(const int radius, const int width, const int height, CIVL::Pixel* input, CIVL::Pixel* output);

namespace CIVL { namespace Blur{

    Image BoxBlur(Image image, int radius) {
        int width = image.width;
        int height = image.height;
        Pixel* image_data = image.pixels.data();

        // Allocate memory on the GPU
        Pixel* d_input;
        Pixel* d_output;
        cudaMalloc(&d_input, width * height * sizeof(Pixel));
        cudaMalloc(&d_output, width * height * sizeof(Pixel));

        // Copy the image data to the GPU
        cudaMemcpy(d_input, image_data, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Calculate the number of blocks and threads
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

        // Call the kernel
        CudaBoxBlur<<<numBlocks, threadsPerBlock>>>(radius, width, height, d_input, d_output);

        // Copy the image data back to the CPU
        cudaMemcpy(image_data, d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

        // Free the memory on the GPU
        cudaFree(d_input);
        cudaFree(d_output);

        return image;
    }
} }

// Cuda code for the Gaussian Blur
__global__ void CudaBoxBlur(const int radius, const int width, const int height, CIVL::Pixel* input, CIVL::Pixel* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int pixelIndex = y * width + x;

        int r = 0, g = 0, b = 0, a = 0;
        int numPixels = 0;

        // Iterate over a square region around the current pixel
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int curX = x + j;
                int curY = y + i;

                // Check if the current pixel is within the bounds of the image
                if (curX >= 0 && curX < width && curY >= 0 && curY < height) {
                    int curIndex = curY * width + curX;
                    CIVL::Pixel curPixel = input[curIndex];

                    r += curPixel.r;
                    g += curPixel.g;
                    b += curPixel.b;
                    a += curPixel.a;

                    numPixels++;
                }
            }
        }

        // Compute the average of the pixels in the square region and store in output image
        output[pixelIndex].r = r / numPixels;
        output[pixelIndex].g = g / numPixels;
        output[pixelIndex].b = b / numPixels;
        output[pixelIndex].a = a / numPixels;
    }
}