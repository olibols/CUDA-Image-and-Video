#include "edge.h"
#define TILE_SIZE 32

__global__ void CudaSobel(const int threshold, const int width, const int height, CIVL::Pixel* input, CIVL::Pixel* output);

namespace CIVL{
    namespace Edge{
        Image Sobel(Image image, int threshold){
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
            CudaSobel<<<numBlocks, threadsPerBlock>>>(threshold, width, height, d_input, d_output);

            // Copy the image data back to the CPU
            cudaMemcpy(image_data, d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

            // Free the memory on the GPU
            cudaFree(d_input);
            cudaFree(d_output);

            return image;
        }
    }
}

__device__ unsigned char grayscalePixel(const CIVL::Pixel& pixel) {
    // Compute grayscale value from RGB components
    return static_cast<unsigned char>(0.2989f * pixel.r + 0.5870f * pixel.g + 0.1140f * pixel.b);
}

__global__ void CudaSobel(const int threshold, const int width, const int height, CIVL::Pixel* input, CIVL::Pixel* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // If on the edge of the image, set the pixel to black
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int index = y * width + x;

        // Convert pixel to grayscale
        unsigned char grayscale = grayscalePixel(input[index]);

        // Apply Sobel operator to estimate gradient in x and y directions
        // (using grayscale value instead of RGB components)
        int gx = -grayscalePixel(input[(y - 1) * width + (x - 1)]) + grayscalePixel(input[(y - 1) * width + (x + 1)])
                 - 2 * grayscalePixel(input[y * width + (x - 1)]) + 2 * grayscalePixel(input[y * width + (x + 1)])
                 - grayscalePixel(input[(y + 1) * width + (x - 1)]) + grayscalePixel(input[(y + 1) * width + (x + 1)]);

        int gy = -grayscalePixel(input[(y - 1) * width + (x - 1)]) - 2 * grayscalePixel(input[(y - 1) * width + x]) - grayscalePixel(input[(y - 1) * width + (x + 1)])
                 + grayscalePixel(input[(y + 1) * width + (x - 1)]) + 2 * grayscalePixel(input[(y + 1) * width + x]) + grayscalePixel(input[(y + 1) * width + (x + 1)]);

        // Compute magnitude of gradient and threshold to produce binary edge map
        float mag = sqrtf(gx * gx + gy * gy);
        output[index].r = mag > threshold ? 255 : 0;
        output[index].g = output[index].r;
        output[index].b = output[index].r;
        output[index].a = 255;
    }
}