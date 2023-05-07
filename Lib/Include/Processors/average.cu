#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "average.h"
#include "../image.h"

#define TILE_SIZE 32

__global__ void CudaWeightedAverage(const float amount, const int width, const int height, CIVL::Pixel* input1, CIVL::Pixel* input2, CIVL::Pixel* output);
__global__ void CudaScreenBlend(const int height, const int width, CIVL::Pixel* input1, CIVL::Pixel* input2, CIVL::Pixel* output);

namespace CIVL{ namespace Average {

    Image WeightedAverage(Image image1, Image image2, float amount){
        int width = image1.width;
        int height = image1.height;

        Pixel* image1_data = image1.pixels.data();
        Pixel* image2_data = image2.pixels.data();

        Pixel* d_input1;
        Pixel* d_input2;
        Pixel* d_output;

        cudaMalloc(&d_input1, width * height * sizeof(Pixel));
        cudaMalloc(&d_input2, width * height * sizeof(Pixel));
        cudaMalloc(&d_output, width * height * sizeof(Pixel));

        cudaMemcpy(d_input1, image1_data, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, image2_data, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Calculate the number of blocks and threads
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

        // Call the kernel
        CudaWeightedAverage<<<numBlocks, threadsPerBlock>>>(amount, width, height, d_input1, d_input2, d_output);

        // Copy the image data back to the CPU
        cudaMemcpy(image1_data, d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

        // Free the memory on the GPU
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);

        return image1;
    }

    Image Average(Image image1, Image image2){
        return WeightedAverage(image1, image2, 0.5);
    }

    Image ScreenBlend(Image image1, Image image2){
        int width = image1.width;
        int height = image1.height;

        Pixel* image1_data = image1.pixels.data();
        Pixel* image2_data = image2.pixels.data();

        Pixel* d_input1;
        Pixel* d_input2;
        Pixel* d_output;

        cudaMalloc(&d_input1, width * height * sizeof(Pixel));
        cudaMalloc(&d_input2, width * height * sizeof(Pixel));
        cudaMalloc(&d_output, width * height * sizeof(Pixel));

        cudaMemcpy(d_input1, image1_data, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, image2_data, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Calculate the number of blocks and threads
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

        // Call the kernel
        CudaScreenBlend<<<numBlocks, threadsPerBlock>>>(width, height, d_input1, d_input2, d_output);

        // Copy the image data back to the CPU
        cudaMemcpy(image1_data, d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

        // Free the memory on the GPU
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);

        return image1;
    }
}}

__device__ CIVL::Pixel multiplyPixel(const CIVL::Pixel& pixel, float amount) {
    CIVL::Pixel outPixel;
    outPixel.r = pixel.r * amount;
    outPixel.g = pixel.g * amount;
    outPixel.b = pixel.b * amount;
    outPixel.a = 255;

    return outPixel;
}

__device__ CIVL::Pixel addPixel(const CIVL::Pixel& pixel1, const CIVL::Pixel& pixel2){
    CIVL::Pixel outPixel;
    outPixel.r = pixel1.r + pixel2.r;
    outPixel.g = pixel1.g + pixel2.g;
    outPixel.b = pixel1.b + pixel2.b;
    outPixel.a = 255;

    return outPixel;
}

__global__ void CudaScreenBlend(const int width, const int height, CIVL::Pixel* input1, CIVL::Pixel* input2, CIVL::Pixel* output){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pixelIndex = y * width + x;

    float srcR = input1[pixelIndex].r / 255.0f;
    float srcG = input1[pixelIndex].g / 255.0f;
    float srcB = input1[pixelIndex].b / 255.0f;
    float srcA = input1[pixelIndex].a / 255.0f;

    float dstR = input2[pixelIndex].r / 255.0f;
    float dstG = input2[pixelIndex].g / 255.0f;
    float dstB = input2[pixelIndex].b / 255.0f;
    float dstA = input2[pixelIndex].a / 255.0f;

    float outR = 1.0f - (1.0f - srcR) * (1.0f - dstR);
    float outG = 1.0f - (1.0f - srcG) * (1.0f - dstG);
    float outB = 1.0f - (1.0f - srcB) * (1.0f - dstB);
    float outA = 1.0f - (1.0f - srcA) * (1.0f - dstA);

    output[pixelIndex].r = static_cast<unsigned char>(outR * 255.0f);
    output[pixelIndex].g = static_cast<unsigned char>(outG * 255.0f);
    output[pixelIndex].b = static_cast<unsigned char>(outB * 255.0f);
    output[pixelIndex].a = static_cast<unsigned char>(outA * 255.0f);
}

__global__ void CudaWeightedAverage(const float amount, const int width, const int height, CIVL::Pixel* input1, CIVL::Pixel* input2, CIVL::Pixel* output){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pixelIndex = y * width + x;

    CIVL::Pixel pixel1 = multiplyPixel(input1[pixelIndex], (1.0f - amount));
    CIVL::Pixel pixel2 = multiplyPixel(input2[pixelIndex], amount);

    output[pixelIndex] = addPixel(pixel1, pixel2);
}

