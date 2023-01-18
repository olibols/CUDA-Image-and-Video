//
// Created by olibo on 07/11/2022.
//

#include <stdio.h>
#include <random>
#include <thread>
#include "image.h"
#include "image_encoder.h"
#include "Processors/blur.h"
#include "Processors/edge.h"

int main(){
    CIVL::Image image = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\test.png");

    CIVL::Image sobelImage = CIVL::Edge::Sobel(image , 200);
    CIVL::Image blurredImage = CIVL::Blur::BoxBlur(sobelImage, 10);
    CIVL::ImageEncoder::EncodeImage("C:\\Users\\olibo\\Downloads\\test2.png", blurredImage);
}