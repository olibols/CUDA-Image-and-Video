//
// Created by olibo on 07/11/2022.
//

#include <stdio.h>
#include <random>
#include <thread>
#include "image.h"
#include "image_encoder.h"
#include "Processors/blur.h"

int main(){
    CIVL::Image image = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\test.png");

    // Calculate average time for 1000 blurs
    double averageTime = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
        CIVL::Image blurredImage = CIVL::Blur::BoxBlur(image, 10);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    averageTime = elapsed.count() / 1000;

    printf("Average time for 1000 blurs: %f seconds", averageTime);

    // Save image
    CIVL::Image blurredImage = CIVL::Blur::BoxBlur(image, 10);
    CIVL::ImageEncoder::EncodeImage("C:\\Users\\olibo\\Downloads\\test2.png", blurredImage);
}